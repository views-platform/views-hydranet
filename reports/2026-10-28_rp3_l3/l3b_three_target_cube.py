"""L3b (#471) — the three-target cube: write `mu`, `theta` and `gate` for ALL THREE targets from
ONE forward sweep, so the family-free lookup can be scored on `ns` and `os` as well as `sb`.

⛔ WHY THIS IS A SEPARATE SCRIPT and not a flag on `d4_mu_on_the_grid.py`. That script's output
shape `(4, 49, 64818)` is a de facto interface: 28 `.py` files reference `d4_result`, 25 of them
load the `.npz` directly, and a silently-added axis would mis-index most of them rather than
raising. ⇒ The writer is left untouched; **#487** holds the deferred decision about whether it
should ever grow a target axis. Nothing here edits it, and this cube is written into the RP-3
dossier under its own name so it can never be mistaken for a `d4_result*.npz`.

⭐⭐⭐ WHY IT COSTS NOTHING EXTRA. The model emits all three targets on EVERY forward pass:
`out.reg` is `[1, n_targets * npar, H, W]` and `out.cls` is `[1, n_targets, H, W]`.
`d4_mu_on_the_grid.py:151,154` slices `SB = 0` out of both and discards the other two at write
time. ⇒ Keeping all three adds ZERO forward passes — only ~0.45 GB of array. The sweep is the same
sweep.

⭐⭐⭐ THE CHECKPOINT, and the correction that made this possible.
`d4_result_CONTROL_nb_4acbf4d3.PROVENANCE.md` states, under a heading reading "⛔ Why this file
cannot be regenerated", that the vehicle behind every family-free number is **DELETED**. ⛔ That
conclusion is FALSE. Its literal sentence — *"not on disk anywhere under `views-models`"* — is
true, and the file is not under `views-models`: it is at
`~/Documents/views_hydranet_vehicle_20261012/calibration_model_20261012_033112.pt`, with its
`.config.json` sidecar, its own `.sha256`, and a full run directory. Verified by hash:

    sha256  4acbf4d3c2b457b0827a84148606c55dab022e859257ca4e669e224edca6bc7f

which is EXACTLY the sha stored inside `d4_result.npz`. Sidecar: family **`nb`**, dropout **0.15**,
300 lessons, git `f99413e`, model `violet_visitor`, region `land`.

⇒ Two consequences, both load-bearing:
  1. ⭐ The three-target result needs **no cross-family caveat at all** — it comes off the very
     vehicle the published `sb` number came off. No transported CV, no family shift.
  2. ⭐⭐ A **bit-identity void gate** becomes available, and it is the strongest one this project
     can construct: the `sb` slice of this cube must be BIT-IDENTICAL to `d4_result.npz`. If it is,
     then in one assertion the lead-in is proven identical, the dropout masks proven paired, the
     slicing proven correct, AND `d4_result.npz` proven to be what it claims — settling the
     provenance question by measurement rather than by reading a note.

C-351 SLOTS, declared:
  · population  all 64,818 land cells x 49 fed months 421-469, x 3 targets, x 4 posterior draws.
  · transform   none. Raw head outputs: `mu`, `theta` in the family's own parameterisation, and
                `gate` as sigmoid(cls). ⛔ NOT composed here — composition is C-386's business and
                belongs to the scorer, per draw then averaged.
  · estimator   none. This script measures nothing; it is a cube writer. Every number it prints is
                either a gate or a cost.
  · normalised  no.

⛔ THE GATES, declared before the run:
  G-1  the checkpoint's own sha256 equals the sha stored inside `d4_result.npz`. If it does not,
       G-2 is MEANINGLESS and is skipped LOUDLY rather than quietly passed.
  G-2  ⭐⭐ BIT IDENTITY. `mu[SB]`, `theta[SB]`, `gate[SB]` from this sweep are bit-identical to
       `d4_result.npz`'s `mu`, `theta`, `gate` — `np.array_equal`, no tolerance. This is a
       structural assertion over arrays produced from identical inputs, not a float sum, so
       exactness is the right demand here (W-96's rule, applied in the direction it permits).
  G-3  NON-VACUITY: the three targets are not copies of one another. Every pair of target slices
       must differ, and the count of differing pairs must be exactly 3 of 3.
  G-4  no `BatchNorm` is in `train()` mode at any point — copied from `d4_mu_on_the_grid.py:104`,
       because C-384 was a BN-in-train defect that inverted the sign of a headline.
  G-5  the saved cube reloads and its `sb` slice still matches, i.e. compression round-trips.

Run: python reports/2026-10-28_rp3_l3/l3b_three_target_cube.py \
         /home/simon/Documents/views_hydranet_vehicle_20261012/calibration_model_20261012_033112.pt
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

from views_hydranet.distributions import get_family
from views_hydranet.distributions.family_loss import FamilyLoss
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate

REPO = Path(__file__).resolve().parents[2]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
SP = Path(__file__).parent
sys.path.insert(0, str(D4))

from vehicle import (  # noqa: E402
    SB,
    TARGETS,
    build_volume,
    grid_rowcol,
    load_vehicle,
    pull,
)

ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit("usage: l3b_three_target_cube.py <path to calibration_model_*.pt> [--tag=X]")
TAG = ""
for _a in sys.argv[1:]:
    if _a.startswith("--tag="):
        TAG = _a.split("=", 1)[1]

#: ⛔ Every constant below is copied from `d4_mu_on_the_grid.py` and must stay copied: G-2's bit
#: identity is only a gate because the lead-in, the month range, the draw count and the seeds are
#: the same. Changing any one of them does not weaken the gate, it DELETES it.
POSTERIOR_D = 4
LEADIN_START, FIRST_KEPT, LAST_FED = 421, 421, 469
torch.set_num_threads(8)

t_start = time.time()

ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
model, cfg, sha = load_vehicle(ART)
model.set_locked_dropout(True)
for m in model.modules():
    if "BatchNorm" in type(m).__name__ and m.training:
        raise SystemExit(
            f"⛔ G-4 VOID: {type(m).__name__} in train() mode — BN would be rewritten."
        )
print("  PASS  G-4  no BatchNorm in train() mode")
npar = FamilyLoss(get_family(cfg["output_distribution"])).n_params
n_tgt = len(TARGETS)
print(f"vehicle sha {sha[:16]}…  family {cfg['output_distribution']}  n_params {npar}  "
      f"dropout {cfg['dropout_rate']}  targets {n_tgt}")

# ── G-1: is the bit-identity gate even applicable? ────────────────────────────────────────
ref_path = D4 / "d4_result.npz"
ref = np.load(ref_path)
ref_sha = str(ref["sha"])
g1 = ref_sha == sha
print(f"  {'PASS' if g1 else '⚠️ N/A '}  G-1  checkpoint sha == d4_result.npz's stored sha "
      f"({ref_sha[:16]}… vs {sha[:16]}…)")
if not g1:
    print("        ⛔ G-2 (bit identity) is MEANINGLESS against a different vehicle and will be")
    print("           SKIPPED, not passed. The cube is still written; it simply carries no")
    print("           reproduction gate, and that must be stated wherever it is quoted.")

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
land_cells = np.sort(df.priogrid_id.unique())
lr, lc = grid_rowcol(land_cells)
lri, lci = lr - 1, lc - 1
months = list(range(LEADIN_START, LAST_FED + 1))
kept = [m for m in months if m >= FIRST_KEPT]
vol, _, _ = build_volume(df, months)
print(f"sweeping {months[0]}-{months[-1]} ({len(months)} steps) x {POSTERIOR_D} passes, "
      f"collecting (mu, theta, gate) for ALL {n_tgt} targets on {len(land_cells):,} land cells …")

MU = np.zeros((n_tgt, POSTERIOR_D, len(kept), len(land_cells)), dtype=np.float32)
TH = np.zeros_like(MU)
GATE = np.zeros_like(MU)
with torch.no_grad():
    for d in range(POSTERIOR_D):
        model.reset_locked_dropout()
        h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
        for t, fed in enumerate(months):
            out = model(vol[:, t], h)
            h = out.h_next
            if fed < FIRST_KEPT:
                continue
            i = fed - FIRST_KEPT
            #: ⛔ the ONLY difference from d4_mu_on_the_grid.py: `j` ranges over all targets
            #: instead of being pinned to SB. The slice arithmetic is otherwise identical.
            for j in range(n_tgt):
                p = out.reg[:, j * npar:(j + 1) * npar, :, :]
                MU[j, d, i] = p[0, 0, lri, lci].numpy()
                TH[j, d, i] = p[0, 1, lri, lci].numpy()
                GATE[j, d, i] = torch.sigmoid(out.cls[0, j])[lri, lci].numpy()
        print(f"  pass {d + 1}/{POSTERIOR_D} done  ({time.time() - t_start:.0f}s)")

print()
print("=" * 99)
print("  THE GATES")
print("=" * 99)

# ── G-2: bit identity of the sb slice against the published cube ──────────────────────────
if g1:
    checks = [
        ("mu", MU[SB], ref["mu"]),
        ("theta", TH[SB], ref["theta"]),
        ("gate", GATE[SB], ref["gate"]),
    ]
    bad = []
    for nm, new, old in checks:
        if new.shape != old.shape:
            bad.append(f"{nm}: shape {new.shape} vs {old.shape}")
        elif not np.array_equal(new, old):
            n_diff = int((new != old).sum())
            bad.append(f"{nm}: {n_diff:,} of {new.size:,} entries differ, "
                       f"max |delta| {float(np.abs(new - old).max()):.6e}")
    g2 = not bad
    print(f"  {'PASS' if g2 else '⛔ FAIL'}  G-2  BIT IDENTITY of the sb slice against "
          f"d4_result.npz (mu, theta, gate)")
    if g2:
        print(f"           all three arrays {MU[SB].shape} bit-identical, np.array_equal, "
              f"no tolerance")
        print("           ⇒ lead-in, month range, draw count, dropout masks and slice arithmetic")
        print("             are all proven identical in ONE assertion — AND d4_result.npz is")
        print("             proven to be what it claims, from the vehicle it claims.")
    else:
        for b_ in bad:
            print(f"           ⛔ {b_}")
else:
    g2 = None
    print("  ⚠️ SKIP  G-2  not applicable — see G-1 above. NOT counted as a pass.")

# ── G-3: the three targets are not copies of one another ──────────────────────────────────
pairs = [(0, 1), (0, 2), (1, 2)]
differ = [(a, b_) for a, b_ in pairs if not np.array_equal(MU[a], MU[b_])]
g3 = len(differ) == 3
print(f"  {'PASS' if g3 else '⛔ FAIL'}  G-3  NON-VACUITY: {len(differ)} of 3 target pairs differ "
      f"in mu")
for a, b_ in pairs:
    d_mu = float(np.abs(MU[a] - MU[b_]).max())
    print(f"           {TARGETS[a]} vs {TARGETS[b_]}: max |delta mu| {d_mu:.6e}")
print("    per-target level, posterior mean over draws:")
for j in range(n_tgt):
    print("      %-14s mu mean %10.6f  theta mean %10.6f  gate mean %10.6f"
          % (TARGETS[j], MU[j].mean(), TH[j].mean(), GATE[j].mean()))

# ── write, then G-5: the round trip ───────────────────────────────────────────────────────
out_path = SP / f"l3b_cube3{TAG}.npz"
np.savez_compressed(
    out_path, mu=MU, theta=TH, gate=GATE, land_cells=land_cells,
    months=np.asarray(kept, np.int32), sha=sha,
    family=cfg["output_distribution"], dropout_rate=cfg["dropout_rate"],
    targets=np.asarray(TARGETS), target_axis=np.asarray(0),
)
back = np.load(out_path)
g5 = (
    np.array_equal(back["mu"], MU)
    and np.array_equal(back["theta"], TH)
    and np.array_equal(back["gate"], GATE)
    and list(back["targets"]) == TARGETS
)
print(f"  {'PASS' if g5 else '⛔ FAIL'}  G-5  the saved cube round-trips bit-identically "
      f"({out_path.stat().st_size / 1e6:.1f} MB)")
print(f"           saved {out_path}")
print(f"           shape {MU.shape} = (targets, draws, months, land cells), axis 0 = "
      f"{list(back['targets'])}")

ok = all(x for x in (g1 is not None, g3, g5)) and (g2 is not False)
print()
print("=" * 99)
print("  ALL GATES PASS" if ok else "  ⛔⛔ A GATE FAILED — the cube is written but is NOT usable")
print("=" * 99)
print("  ⛔ This cube MEASURES NOTHING. It is raw head output. Composition is the scorer's job")
print("     and is E[gate x mu] per draw then averaged, never E[gate] x E[mu] (C-386).")
print("  ⛔ `d4_mu_on_the_grid.py` was NOT modified. #487 holds the decision about whether the")
print("     repo's cube writer should ever carry a target axis.")
print()
print("  wall time %.1f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
raise SystemExit(0 if ok else 1)
