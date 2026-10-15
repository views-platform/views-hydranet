"""R4 — do the body and gate gradients actually CONFLICT on the shared trunk?

WHY THIS EXISTS. R2/R3 measured LOSS VALUES: the arm's body term is 35.2x its gate term where
the control's is 0.593x. The method review (reports/2026-10-15_loss_balance_method_review.md)
established that a loss-value ratio is NOT the quantity that governs allocation:

  1. Only ~11.3% of this model's parameters are shared between the body and gate paths. Each has
     its own ~400k-parameter decoder (`*_reg` / `*_class`); they share the encoder, the
     bottleneck, the ConvLSTM cells and their BatchNorm.
  2. The optimiser is AdamW, whose update is invariant to a constant on the gradient. So a
     per-task loss constant is EXACTLY invisible on that task's PRIVATE parameters (verified
     numerically: max|diff| = 0.0 at scale 1 vs 65.5 over 60 steps). It acts only by tilting the
     SUMMED gradient's DIRECTION on the shared subset.
  3. GradNorm (Chen et al. 2018) states the mechanism as imbalance "manifest[ing] as imbalances
     between backpropagated gradients" -- gradient norms, not loss values.

So the allocation story has a precondition nobody has tested: the two gradients must actually
conflict on the shared parameters. If they are near-orthogonal or aligned, there is nothing to
allocate, the 65x measured nothing, and the fork's answer is N.

PRE-COMMITTED READING (reports/2026-10-15_loss_balance_method_review.md section 4.1):
  cos(grad body, grad gate) on the shared subset clearly NEGATIVE  -> conflict is real, A/B are
      aimed at something that exists.
  cos >= 0, or shared-trunk gradient NORMS already comparable despite the 65x loss gap
      -> nothing to allocate; the diagnosis measured a quantity with no causal path. Ruling -> N.

SCOPE, stated not discovered.
  * eval() mode, as R2/R3: dropout off, BN from buffers. train() would rewrite the artifacts' BN
    running stats (C-184/C-328) and the artifacts cannot be retrained (C-317).
  * Production backprops the WHOLE window in one backward (`w_loss.backward()`,
    training_engine.py:1467) over a graph with no state detach, i.e. full-window BPTT. Holding
    that graph while taking six separate per-task backwards does not fit, so the BPTT is CHUNKED:
    `h` is carried with grad inside a chunk of K steps and detached at the boundary. This drops
    the >K-step component of the gradient. The training-loop audit measured gradient reaching
    ~118 steps, so K=1 is NOT production. Mitigation: the cosine is reported for several K and
    the verdict is taken only from a sign that is STABLE across K. A sign that moves with K is
    itself the finding and voids the test.
  * The sampler's RNG stream is not production's, so the windows are from the same distribution
    but are not the same windows (same caveat as R2/R3). `random_flips` off.
  * Deliberately a WET sibling of r2/r3 rather than an edit: their numbers are cited in OUTCOME.md
    and their provenance stays fixed.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate

sys.path.insert(0, str(Path(__file__).parent.parent / "2026-10-12_underfire_diagnosis"))
from vehicle import TARGETS, grid_rowcol, pull  # noqa: E402

SP = Path(__file__).parent
DIAG = SP.parent / "2026-10-12_underfire_diagnosis"
MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
CONTROL_PT = Path("/home/simon/Documents/views_hydranet_vehicle_20261012/"
                  "calibration_model_20261012_033112.pt")
ARM_PT = (MODELS / "amber_kestrel/data/generated/calibration_delivery_20261013/"
          "artifact/calibration_model_20261013_010739.pt")
N_SAMPLED_LESSONS = 4          # 4 lessons x 3 windows = 12 windows per (model, K)
CHUNKS = (1, 8, 24)            # K: BPTT chunk lengths to test the verdict's stability
POS_WEIGHT = 2.0


def load_hp(name: str) -> dict:
    path = MODELS / name / "configs" / "config_hyperparameters.py"
    spec = importlib.util.spec_from_file_location(f"hp_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_hp_config()


def load_weights(pt: Path, expect_dist: str):
    from views_hydranet.utils.utils import choose_model

    cfg = json.loads(pt.with_suffix(".pt.config.json").read_text())
    if cfg.get("output_distribution") != expect_dist:
        raise SystemExit(f"VOID: {pt.name} is {cfg.get('output_distribution')!r}, "
                         f"expected {expect_dist!r}.")
    sha = hashlib.sha256(pt.read_bytes()).hexdigest()
    shaf = pt.with_suffix(".pt.sha256")
    if shaf.exists() and shaf.read_text().split()[0] != sha:
        raise SystemExit(f"VOID: sha256 mismatch for {pt.name}.")
    model = choose_model(cfg, torch.device("cpu"))
    model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    model.to("cpu").eval()
    print(f"  {pt.name}  sha {sha[:16]}…  dist={cfg['output_distribution']}")
    return model


def shared_params(model):
    """The parameters BOTH paths touch: everything not in a `*_reg` or `*_class` head."""
    out = [(n, p) for n, p in model.named_parameters()
           if "_reg" not in n and "_class" not in n and p.requires_grad]
    if not out:
        raise SystemExit("VOID: no shared parameters identified.")
    return out


def build_handler(config: dict):
    from views_hydranet.utils.feature_scaler import FeatureScaler
    from views_hydranet.utils.volume_handler import VolumeHandler

    df = pull((121, 504), DIAG / "pull_land_all3_121_504.parquet")
    df = df[(df.month_id >= 121) & (df.month_id <= 456)].copy()
    r, c = grid_rowcol(df.priogrid_id.to_numpy())
    df["row"], df["col"] = r, c
    df["c_id"] = 0
    df = df.rename(columns=dict(zip(TARGETS, config["regression_targets"])))
    return VolumeHandler.from_df(FeatureScaler(config).fit_transform(df), config)


def flat_grad(task_loss, shared, retain):
    """d(task_loss)/d(shared params), flattened. Missing grads -> zeros."""
    gs = torch.autograd.grad(task_loss, [p for _, p in shared],
                             retain_graph=retain, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g, (_, p) in zip(gs, shared)])


def measure(model, config, handler, family, K):
    """-> per-task shared-trunk gradient vectors, summed over the window, per window."""
    from views_hydranet.distributions.family_loss import FamilyLoss
    from views_hydranet.train.training_engine import _SequenceIndices
    from views_hydranet.utils.curriculum import CurriculumLearner
    from views_hydranet.utils.volume_sampler import VolumeSampler
    from views_hydranet.utils.weighted_bce_loss import WeightedBCEWithLogitsLoss

    sampler = VolumeSampler(handler, config)
    planner = CurriculumLearner(config, handler)
    gate_fn = WeightedBCEWithLogitsLoss(pos_weight=POS_WEIGHT)
    body_fn = FamilyLoss(family)
    npar = body_fn.n_params
    shared = shared_params(model)
    wpl = config["windows_per_lesson"]
    lessons = np.unique(np.linspace(0, config["total_lessons"] - 1, N_SAMPLED_LESSONS).astype(int))

    rows, names = [], None
    for li in lessons:
        for wi in range(wpl):
            target, threshold = planner.get_lesson(int(li) * wpl + wi)
            sh = sampler.get_batch(target, threshold, batch_size=1)[0][0]
            tt = sh.to_pytorch(torch.device("cpu"), include_identities=False)
            idx = _SequenceIndices([n for n in sh.channel_map if n in sh.tensor_cols], config)
            names = [f"body/{n}" for n in idx.reg_names] + [f"gate/{n}" for n in idx.cls_names]
            n_task = idx.n_reg + idx.n_cls
            acc = [torch.zeros(sum(p.numel() for _, p in shared)) for _ in range(n_task)]
            h = model.init_hTtime(hidden_channels=model.base, H=tt.shape[-2], W=tt.shape[-1])
            steps = tt.shape[1] - 1
            for start in range(0, steps, K):
                stop = min(start + K, steps)
                chunk = [None] * n_task
                for i in range(start, stop):
                    t1 = tt[:, i + 1]
                    y_reg, y_cls = t1[:, idx.reg], t1[:, idx.cls]
                    o = model(tt[:, i][:, idx.feat], h)
                    h = o.h_next
                    for j in range(idx.n_reg):
                        p = o.reg[:, j * npar:(j + 1) * npar].permute(0, 2, 3, 1)
                        term = body_fn(p, y_reg[:, j])
                        chunk[j] = term if chunk[j] is None else chunk[j] + term
                    for j in range(idx.n_cls):
                        term = gate_fn(o.cls[:, j], y_cls[:, j])
                        k = idx.n_reg + j
                        chunk[k] = term if chunk[k] is None else chunk[k] + term
                for t in range(n_task):
                    acc[t] += flat_grad(chunk[t], shared, retain=(t < n_task - 1))
                h = h.detach()
            rows.append(torch.stack(acc).numpy() / steps)
    return np.asarray(rows), names, sum(p.numel() for _, p in shared)


def report(tag, G, names, n_shared):
    """G: [W, n_task, n_shared]."""
    nr = sum(1 for n in names if n.startswith("body/"))
    print(f"\n{'=' * 94}\n{tag}   ({G.shape[0]} windows · {n_shared:,} shared params)\n{'=' * 94}")
    nrm = np.linalg.norm(G, axis=2)                                   # [W, n_task]
    print(f"  {'task':<22}{'||grad|| on shared':>20}{'as-run coeff':>14}{'coeff*||grad||':>16}")
    for t, n in enumerate(names):
        c = 0.5 if n.startswith("body/") else 1.0
        print(f"  {n:<22}{np.median(nrm[:, t]):>20.4e}{c:>14.1f}{c * np.median(nrm[:, t]):>16.4e}")
    gb = 0.5 * G[:, :nr].sum(1)
    gg = 1.0 * G[:, nr:].sum(1)
    nb, ng = np.linalg.norm(gb, axis=1), np.linalg.norm(gg, axis=1)
    print(f"\n  AS-RUN shared-trunk gradient-norm ratio  ||0.5*sum body|| / ||sum gate||"
          f"  = {np.median(nb / ng):.3f}")
    print("     (compare the LOSS-VALUE ratio R2 measured: control 0.593, arm 35.2)")
    cos = (gb * gg).sum(1) / (nb * ng)
    print(f"\n  cos( grad body , grad gate ) on the shared trunk:"
          f"  median {np.median(cos):+.4f}   mean {cos.mean():+.4f}")
    print(f"     range [{cos.min():+.4f}, {cos.max():+.4f}]   "
          f"windows with cos < 0: {100 * (cos < 0).mean():.0f}%")
    print("\n  PAIRWISE cosines, median over windows (body rows x gate cols):")
    print("    " + " " * 20 + "".join(f"{n.split('/')[1][3:9]:>10}" for n in names[nr:]))
    for a in range(nr):
        cells = []
        for b in range(nr, len(names)):
            c = ((G[:, a] * G[:, b]).sum(1)
                 / (np.linalg.norm(G[:, a], axis=1) * np.linalg.norm(G[:, b], axis=1)))
            cells.append(f"{np.median(c):>+10.4f}")
        print(f"    {names[a]:<20}" + "".join(cells))
    print("\n  body-vs-body pairwise (do the three targets fight each other?):")
    for a in range(nr):
        for b in range(a + 1, nr):
            c = ((G[:, a] * G[:, b]).sum(1)
                 / (np.linalg.norm(G[:, a], axis=1) * np.linalg.norm(G[:, b], axis=1)))
            print(f"    {names[a]:<22} vs {names[b]:<22} {np.median(c):>+8.4f}")
    return dict(norm=nrm, cos=cos, ratio=nb / ng)


ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
print("loading vehicles …")
ctrl = load_weights(CONTROL_PT, "nb")
arm = load_weights(ARM_PT, "truncated_nb")
ctrl_cfg, arm_cfg = load_hp("violet_visitor"), load_hp("amber_kestrel")
if ctrl_cfg.get("target_weights") or arm_cfg.get("target_weights"):
    raise SystemExit("VOID: a config carries target_weights (C-87).")

ns = shared_params(ctrl)
tot = sum(p.numel() for p in ctrl.parameters())
sh = sum(p.numel() for _, p in ns)
print(f"\nshared parameters: {sh:,} of {tot:,} = {100 * sh / tot:.2f}% of the model")
print("  " + ", ".join(n for n, _ in ns[:6]) + f", … ({len(ns)} tensors)")

print("\nbuilding the calibration train volume …")
handler = build_handler(ctrl_cfg)

from views_hydranet.distributions import resolve_family  # noqa: E402

out = {}
for K in CHUNKS:
    print(f"\n\n######################  BPTT CHUNK K = {K}  ######################")
    Gc, names, nsh = measure(ctrl, ctrl_cfg, handler, resolve_family("nb"), K)
    out[f"ctrl_K{K}"] = report(f"CONTROL (nb)   K={K}", Gc, names, nsh)
    Ga, names, nsh = measure(arm, arm_cfg, handler, resolve_family("truncated_nb"), K)
    out[f"arm_K{K}"] = report(f"ARM (truncated_nb)   K={K}", Ga, names, nsh)

print(f"\n\n{'=' * 94}\nVERDICT TABLE — is the sign stable across K?\n{'=' * 94}")
print(f"  {'':<10}{'cos(body,gate)':>18}{'||0.5 body||/||gate||':>24}")
for K in CHUNKS:
    for tag in ("ctrl", "arm"):
        d = out[f"{tag}_K{K}"]
        print(f"  {tag + ' K=' + str(K):<10}{np.median(d['cos']):>+18.4f}"
              f"{np.median(d['ratio']):>24.3f}")
np.savez(SP / "r4_result.npz", **{f"{k}_{m}": v[m] for k, v in out.items()
                                  for m in ("norm", "cos", "ratio")})
print("\nwrote r4_result.npz")
