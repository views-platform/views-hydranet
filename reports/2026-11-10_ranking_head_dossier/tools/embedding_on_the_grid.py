"""embedding_on_the_grid.py — dump the 32-channel decoder embedding the magnitude head reads.

⭐ **NO NEW INFRASTRUCTURE.** The vehicle loader, the volume builder, the grid mapping, the data
pull, the sweep window, the thread cap, the determinism proof and the BatchNorm refusal are all
taken from `score_on_the_grid.py`, which is committed and whose `mu` field is the pinned comparator
(`+0.5154 ± 0.0397`, `calm_lander`). What is new here is **one forward hook**, nothing else.

**What it collects and why.** `HydraBNrecurrentUnet_06_LSTM4.py:642-643` is the whole magnitude
pathway's last step::

    H1_reg   = self.dec_conv4_head1_reg(H1_reg)   # Conv2d(32 -> n_params, 3x3)
    out_reg1 = self._reg_activation(H1_reg)        # softplus | relu | identity — MONOTONE

⇒ The body's ordering of cells is the rank of a **single linear functional** of the 32-channel map
entering that conv, because every activation after it is monotone and `ρ_event` is rank-only. This
tool stores that map (`emb`) and the conv's pre-activation output (`mu_pre`), so the question *"is
the ordering limited by the embedding or by the 289 coefficients that read it?"* can be asked on a
laptop, with the trunk frozen.

⛔⛔ **TEACHER FORCING, `C-399`, Tier 1.** `vehicle.build_volume` writes the **observed** fatality
series into the input channels for every month, and this loop steps over that volume, so the model
is handed observed history at every step. Every file carries `input_mode="teacher_forced"` and may
**never** be compared to the delivered autoregressive figures (the body's `+0.3127`, the gate's
`+0.4418`). The matched comparator is the teacher-forced `mu` row, `+0.5154 ± 0.0397`.

⛔ **The hook is VERIFIED, not assumed.** Which of `head1/head2/head3` serves target `SB` is a
convention, and a wrong guess would silently dump another target's embedding. So on the first kept
step this tool applies the model's own `_reg_activation` to the hook's output and requires
**bit-equality** with `out.reg`'s `mu` channel. A mismatch refuses and writes nothing.

**Population and provenance, per `C-351`:** all land cells of the `region='land'` pull
(`land_cells`, written into the output), log1p inputs, one target (`sb`), recurrent state carried
from `LEADIN_START` with the lead-in discarded. Only fed months `>= EMB_FIRST_KEPT` are stored —
the sweep is identical, the retention window is not, because the embedding is 32x the size of a
scalar field and the folds need only months 457..469.

Usage::

    python embedding_on_the_grid.py <calibration_model_*.pt> --tag CTL_EMB_calm_lander

Writes `results/emb_grid_<tag>.npz` with `emb [n_months, 32, n_land]`,
`mu_pre [n_months, n_land]`, `gate [n_months, n_land]`, `months`, `land_cells`, `sha`,
`input_mode`. Exit 0 on success; a refusal exits non-zero and writes nothing.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPORTS = _HERE.parents[1]
_VEHICLE = _REPORTS / "2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(_VEHICLE))

from vehicle import SB, TARGETS, build_volume, grid_rowcol, load_vehicle, pull  # noqa: E402

from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate  # noqa: E402

#: ⛔ Identical to `score_on_the_grid.py` so the two runs are the same sweep over the same vehicle.
LEADIN_START, LAST_FED = 421, 469
#: The RETENTION window, which is the only thing that differs. `e1_data.FOLDS` scores months
#: 463..469 on six-month fit windows, so forecast months 457..469 are needed; `G-ALIGN` reads the
#: forecast for month `m` at fed index `m-1`, so fed 456..468 suffices. 450 leaves margin.
EMB_FIRST_KEPT = 450
N_THREADS = 8
#: ⛔ `input_mode` is a stored fact, not a flag. `C-399`.
INPUT_MODE = "teacher_forced"
#: The layer whose INPUT is the embedding and whose OUTPUT is the pre-activation score.
_HEAD_CONV = "dec_conv4_head1_reg"


def _collect(model, cfg, vol, months, lri, lci, npar: int):
    """-> (emb, mu_pre, gate, kept). emb [n_kept, 32, n_land]; the rest [n_kept, n_land]."""
    conv = getattr(model, _HEAD_CONV)
    cap: dict[str, torch.Tensor] = {}

    def _hook(_mod, inp, out):
        cap["in"] = inp[0].detach()
        cap["out"] = out.detach()

    handle = conv.register_forward_hook(_hook)
    kept = [m for m in months if m >= EMB_FIRST_KEPT]
    n_in = conv.in_channels
    emb = np.zeros((len(kept), n_in, lri.size), dtype=np.float32)
    mu_pre = np.zeros((len(kept), lri.size), dtype=np.float32)
    gate = np.zeros_like(mu_pre)
    verified = False
    try:
        with torch.no_grad():
            h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
            for t, fed in enumerate(months):
                out = model(vol[:, t], h)
                h = out.h_next
                if fed < EMB_FIRST_KEPT:
                    continue
                if not verified:
                    # ⛔ The head-to-target check. `_reg_activation` is the model's own, so this
                    # proves the hook sits on the conv whose output BECOMES the mu we pinned.
                    mine = model._reg_activation(cap["out"])[0, 0]
                    theirs = out.reg[:, SB * npar : (SB + 1) * npar][0, 0]
                    if not torch.equal(mine, theirs):
                        raise SystemExit(
                            f"REFUSED: the hook on {_HEAD_CONV} does not reproduce target "
                            f"{TARGETS[SB]}'s mu — max |Δ| "
                            f"{float((mine - theirs).abs().max()):.3e}. The head-to-target "
                            "convention is not what this tool assumed; nothing is written."
                        )
                    print(
                        f"  hook VERIFIED: _reg_activation({_HEAD_CONV}) is bit-identical to "
                        f"out.reg's {TARGETS[SB]} mu ({cap['in'].shape[1]} embedding channels)"
                    )
                    verified = True
                i = fed - EMB_FIRST_KEPT
                emb[i] = cap["in"][0, :, lri, lci].numpy()
                mu_pre[i] = cap["out"][0, 0, lri, lci].numpy()
                gate[i] = torch.sigmoid(out.cls[0, SB])[lri, lci].numpy()
    finally:
        handle.remove()
    return emb, mu_pre, gate, kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "artifact", type=Path, help="a calibration_model_*.pt WITH its .pt.config.json"
    )
    ap.add_argument(
        "--tag", required=True, help="output suffix; ⛔ never reuse one, it overwrites"
    )
    args = ap.parse_args()

    torch.set_num_threads(N_THREADS)
    t0 = time.time()
    print(__doc__.split("Usage::")[0])
    print("=" * 100)
    print(f"  EMBEDDING ON THE GRID · tag={args.tag} · input_mode={INPUT_MODE}")
    print("=" * 100)

    out_path = _HERE.parent / "results" / f"emb_grid_{args.tag}.npz"
    if out_path.exists():
        print(f"  ⛔ REFUSED: {out_path.name} exists. A --tag collision overwrites it.")
        return 2

    ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
    model, cfg, sha = load_vehicle(args.artifact)

    from views_hydranet.architectures.locked_dropout import LockedDropout

    n_do = sum(1 for m in model.modules() if isinstance(m, LockedDropout))
    for m in model.modules():
        if isinstance(m, LockedDropout):
            m.locked = False
            m.eval()
    print(f"  one deterministic pass: {n_do} LockedDropout modules set to eval()")
    for m in model.modules():
        if "BatchNorm" in type(m).__name__ and m.training:
            print(f"  ⛔ REFUSED: {type(m).__name__} is in train() mode — BN would be rewritten.")
            return 2

    from views_hydranet.distributions import get_family
    from views_hydranet.distributions.family_loss import FamilyLoss

    npar = FamilyLoss(get_family(cfg["output_distribution"])).n_params
    print(f"  vehicle sha {sha} · family {cfg['output_distribution']} · n_params {npar}")

    df = pull((121, 504), _VEHICLE / "pull_land_all3_121_504.parquet")
    land_cells = np.sort(df.priogrid_id.unique())
    lr, lc = grid_rowcol(land_cells)
    lri, lci = lr - 1, lc - 1
    months = list(range(LEADIN_START, LAST_FED + 1))
    vol, _land, _pg = build_volume(df, months)
    print(
        f"  sweeping {months[0]}-{months[-1]} ({len(months)} sequential steps) over "
        f"{len(land_cells):,} land cells · retaining fed {EMB_FIRST_KEPT}-{LAST_FED}"
    )

    h0 = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
    with torch.no_grad():
        a, b = model(vol[:, 0], h0).reg, model(vol[:, 0], h0).reg
    if not torch.equal(a, b):
        print(
            "  ⛔ REFUSED: two forwards on identical input differ by "
            f"{float((a - b).abs().max()):.3e} — a LockedDropout is still sampling."
        )
        return 2
    print("  determinism VERIFIED: two forwards on identical input are bit-identical")

    emb, mu_pre, gate, kept = _collect(model, cfg, vol, months, lri, lci, npar)

    # ⛔ Non-degeneracy on BOTH stored fields, before anything is saved. §0 item 2 of the
    # evaluation report: a delivery served 0.0 for 2,330,712 cells and every structural check
    # passed. An all-zero embedding channel is the same failure one level down, so the count is
    # per channel and the worst is printed.
    d_mu = float(np.unique(mu_pre).size) / float(mu_pre.size)
    per_ch = np.array([np.unique(emb[:, c, :]).size for c in range(emb.shape[1])])
    dead = int((per_ch <= 1).sum())
    print(
        f"\n  mu_pre distinct {d_mu:.6f} of {mu_pre.size:,} · "
        f"range [{mu_pre.min():+.4f}, {mu_pre.max():+.4f}]"
    )
    print(
        f"  embedding: {emb.shape[1]} channels · distinct values per channel "
        f"min {per_ch.min():,} median {int(np.median(per_ch)):,} max {per_ch.max():,} · "
        f"constant channels {dead}"
    )
    if d_mu <= 1e-6:
        print("  ⛔ REFUSED: mu_pre is constant. Nothing is written.")
        return 2
    if dead == emb.shape[1]:
        print("  ⛔ REFUSED: every embedding channel is constant. Nothing is written.")
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        emb=emb,
        mu_pre=mu_pre,
        gate=gate,
        months=np.asarray(kept, np.int32),
        land_cells=land_cells,
        sha=sha,
        input_mode=INPUT_MODE,
        family=cfg["output_distribution"],
        target=TARGETS[SB],
        head_conv=_HEAD_CONV,
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"\n  saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(
        f"  ⛔ input_mode={INPUT_MODE} is stored IN the file (C-399). The matched comparator is "
        "the teacher-forced mu row, +0.5154 ± 0.0397 — never the delivered +0.3127."
    )
    print(f"  cost: {time.time() - t0:.0f}s wall · {rss:.2f} GB peak RSS · {N_THREADS} threads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
