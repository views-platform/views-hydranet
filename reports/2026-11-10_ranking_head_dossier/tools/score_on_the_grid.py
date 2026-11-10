"""score_on_the_grid.py — write ONE scalar field per cell per month, for the ρ_event readout.

⭐ **NO NEW INFRASTRUCTURE.** Every moving part is imported from a committed instrument, which is
the discipline `2026-11-08_ordering_discrimination/tools/mu_ordering.py` set and the reason it
could
be trusted: the vehicle loader, the volume builder, the grid mapping and the data pull all come
from
`../../2026-10-12_underfire_diagnosis/vehicle.py`, and the forward loop is the same sequential
recurrent sweep `d4_mu_on_the_grid.py` runs. What is new here is **which channel is collected** and
**the output contract**, nothing else.

⛔⛔ **WHY THIS EXISTS, AND WHY `body_mean_dump_dir` CANNOT DO IT.** The one in-package seam that
writes a per-cell field to disk refuses a head with no registered distribution family
(`hydranet_inference.py:428-430`) — and a ranking head has none; it emits one bare score. The seam
also sits *inside* the posterior-pass loop (`:1738`), so switching it on builds the draw cube that
OOM-killed both EXP-02 smoke arms on **host RAM** at ~21 GiB of 31. A ranking readout needs one
number per cell and **zero draws**, so paying for a cube would be paying for the thing that breaks.
⚠️ §9c `N9` of `2026-09-30_evaluation_learning_report.md` records the same gap from the other side:
the body's `mu` field is **not on disk for any delivered run** — `S8`/#568. ⇒ One tool, both jobs:
`--channel mu` is that missing field, `--channel latent` is the ranking score.

⛔⛔ **THE TEACHER-FORCING LABEL IS THE MOST IMPORTANT THING IN THIS FILE.** `vehicle.build_volume`
writes the **observed** fatality series into the model's input channels for every month
(`vehicle.py:398-404`), and this loop steps over that volume — so at every step the model is handed
the observed history, **not its own predictions**. That is **`C-399`, Tier 1**, verbatim: the
frozen
cube every RP-3 number was computed from had exactly this property, and it measured teacher-forced
AUC **0.99777** against the delivered autoregressive **0.72340**, with no decay in horizon.

⇒ The defect `C-399` records is **not the mode — it is the carry.** Teacher-forced at h1 is the
same input regime the body is *trained* in, so it is the right setting for the one question phase 3
asks — *does a patch-trained score sort the GLOBAL grid?* — because it changes the patch→global
axis
alone and leaves the rollout axis fixed. ⛔ What may never happen is comparing a field from this
tool to the **delivered** autoregressive numbers (the body's `+0.3127`, the gate's `+0.4418`). So
every file this tool writes carries `input_mode="teacher_forced"`, and
`rho_event_readout.py` refuses to put it in the same table as a delivered row.

⭐ **`G-ALIGN` comes free and is not optional.** The gate is collected on every run whatever
`--channel` asks for, because the **delivered** gate at h1 is on disk for all 13 origins. Comparing
them fixes the month offset **by measurement** rather than by reading the code — the convention of
whether feeding month `t` yields the forecast for `t` or `t+1` is silent if wrong, and it is one
month of displacement away from a plausible wrong answer.

**Population and provenance, per `C-351`:** all land cells of the `region='land'` pull
(`land_cells`, written into the output), log1p inputs, one target (`sb`), the recurrent state
carried
from `LEADIN_START` with the lead-in discarded. Cost is **measured and printed**, never estimated.

Usage::

    python score_on_the_grid.py <calibration_model_*.pt> --channel latent --tag RANKING_ARM
    python score_on_the_grid.py <calibration_model_*.pt> --channel mu     --tag CTL_NB
    python score_on_the_grid.py <calibration_model_*.pt> --channel gate   --tag CTL_GATE

Writes `results/score_grid_<tag>.npz` with `score [n_months, n_land]`, `gate [n_months, n_land]`,
`months`, `land_cells`, `sha`, `channel`, `input_mode`, `n_passes`. Exit 0 on success; a refusal
exits non-zero and writes nothing.
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

#: The lead-in and the kept window, identical to `d4_mu_on_the_grid.py` so the two are comparable.
LEADIN_START, FIRST_KEPT, LAST_FED = 421, 421, 469
#: ⛔ 8 of 12 threads. The first launch of `d4` was uncapped, took all 12 cores and made the laptop
#: unusable mid-video-call. Wall time is dominated by the SEQUENTIAL step count anyway.
N_THREADS = 8
#: ⛔ `input_mode` is a stored fact, not a flag: see the module docstring. `C-399`.
INPUT_MODE = "teacher_forced"

_CHANNELS = ("latent", "mu", "gate")


def _collect(model, cfg, vol, months, lri, lci, channel: str, npar: int, n_passes: int):
    """-> (field, gate) each [n_passes, n_kept, n_land]. One sequential sweep per pass."""
    kept = [m for m in months if m >= FIRST_KEPT]
    field = np.zeros((n_passes, len(kept), lri.size), dtype=np.float32)
    gate = np.zeros_like(field)
    with torch.no_grad():
        for d in range(n_passes):
            model.reset_locked_dropout()
            h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
            for t, fed in enumerate(months):
                out = model(vol[:, t], h)
                h = out.h_next
                if fed < FIRST_KEPT:
                    continue
                i = fed - FIRST_KEPT
                if channel == "latent":
                    lat = getattr(out, "reg_latent", None)
                    if lat is None:
                        raise SystemExit(
                            "REFUSED: this model emits no `reg_latent`, so there is no "
                            "pre-activation score to collect. --channel latent needs a head that "
                            "exposes it; `mu` and `gate` do not."
                        )
                    field[d, i] = lat[0, SB, lri, lci].numpy()
                elif channel == "mu":
                    field[d, i] = out.reg[:, SB * npar : (SB + 1) * npar][0, 0, lri, lci].numpy()
                else:
                    field[d, i] = torch.sigmoid(out.cls[0, SB])[lri, lci].numpy()
                gate[d, i] = torch.sigmoid(out.cls[0, SB])[lri, lci].numpy()
    return field, gate, kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "artifact", type=Path, help="a calibration_model_*.pt WITH its .pt.config.json"
    )
    ap.add_argument("--channel", choices=_CHANNELS, required=True)
    ap.add_argument(
        "--tag", required=True, help="output suffix; ⛔ never reuse one, it overwrites"
    )
    ap.add_argument(
        "--passes",
        type=int,
        default=1,
        help="MC-dropout passes. ⚠️ Default 1 DETERMINISTIC pass: averaging posterior passes of a "
        "SCALE-FREE score is not the same object as averaging means — the mean of orderings is a "
        "different ordering. >1 enables locked dropout and averages, as d4 does for `mu`.",
    )
    args = ap.parse_args()

    torch.set_num_threads(N_THREADS)
    t0 = time.time()
    print(__doc__.split("Usage::")[0])
    print("=" * 100)
    print(
        f"  SCORE ON THE GRID · channel={args.channel} · tag={args.tag} · "
        f"passes={args.passes} · input_mode={INPUT_MODE}"
    )
    print("=" * 100)

    out_path = _HERE.parent / "results" / f"score_grid_{args.tag}.npz"
    if out_path.exists():
        print(
            f"  ⛔ REFUSED: {out_path.name} exists. A --tag collision overwrites a field that "
            "may be the only copy — `d4`'s control npz was produced by a checkpoint that no "
            "longer exists on disk, and an unconditional write destroyed it once already."
        )
        return 2

    ReproducibilityGate.lock_entropy(np_seed=42, torch_seed=42)
    model, cfg, sha = load_vehicle(args.artifact)
    if args.passes > 1:
        model.set_locked_dropout(True)
    else:
        from views_hydranet.architectures.locked_dropout import LockedDropout

        n_do = sum(1 for m in model.modules() if isinstance(m, LockedDropout))
        for m in model.modules():
            if isinstance(m, LockedDropout):
                m.locked = False
                m.eval()
        print(f"  --passes 1: {n_do} LockedDropout modules set to eval()")
    for m in model.modules():
        if "BatchNorm" in type(m).__name__ and m.training:
            print(f"  ⛔ REFUSED: {type(m).__name__} is in train() mode — BN would be rewritten.")
            return 2

    # ⚠️ Resolved for EVERY channel, not only for `mu`, so the printed line is a fact about the
    # vehicle rather than about this code path. An earlier version computed it only under `mu` and
    # printed "family nb · n_params 1" on a gate run — nb has 2. A sentence about what exists,
    # wrong, which is the failure mode `2026-09-30_evaluation_learning_report.md` F.5 is about.
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
        f"  sweeping {months[0]}-{months[-1]} ({len(months)} sequential steps) × "
        f"{args.passes} pass(es) over {len(land_cells):,} land cells …"
    )

    if args.passes == 1:
        # ⛔ PROVE dropout is off before spending the pass: `locked = False` alone clears the flag
        # without calling .eval(), so a module already in train() would resample a FRESH mask every
        # forward — MORE noise, not less. The C-324 inert-knob signature, pointed the other way.
        h0 = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
        with torch.no_grad():
            a, b = model(vol[:, 0], h0).reg, model(vol[:, 0], h0).reg
        if not torch.equal(a, b):
            print(
                "  ⛔ REFUSED: two forwards on identical input differ by "
                f"{float((a - b).abs().max()):.3e} — a LockedDropout is still sampling, so "
                "'1 deterministic pass' would be one arbitrary draw."
            )
            return 2
        print("  determinism VERIFIED: two forwards on identical input are bit-identical")

    field, gate, kept = _collect(
        model, cfg, vol, months, lri, lci, args.channel, npar, args.passes
    )
    score = field.mean(0).astype(np.float64)
    gate_m = gate.mean(0).astype(np.float64)

    # ⛔ Non-degeneracy, printed before anything is saved: §0 item 2 of the evaluation report — a
    # delivery served 0.0 for all 2,330,712 cells and EVERY structural check passed.
    distinct = float(np.unique(score).size) / float(score.size)
    print(
        f"\n  distinct values {distinct:.6f} of {score.size:,} · "
        f"range [{score.min():+.4f}, {score.max():+.4f}] · "
        f"gate distinct {np.unique(gate_m).size / gate_m.size:.6f}"
    )
    if distinct <= 1e-6:
        print("  ⛔ REFUSED: the field is constant. Nothing is written.")
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        score=score,
        gate=gate_m,
        months=np.asarray(kept, np.int32),
        land_cells=land_cells,
        sha=sha,
        channel=args.channel,
        input_mode=INPUT_MODE,
        n_passes=np.int32(args.passes),
        family=cfg["output_distribution"],
        target=TARGETS[SB],
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"\n  saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(
        f"  ⛔ input_mode={INPUT_MODE} is stored IN the file (C-399). It may not be compared "
        "to the delivered autoregressive figures."
    )
    print(f"  cost: {time.time() - t0:.0f}s wall · {rss:.2f} GB peak RSS · {N_THREADS} threads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
