"""P0 — the zero attribution measured on the PRODUCTION posterior, not inferred from one pass.

T0 (views-hydranet#396) attributed the composed zero mass analytically from a SINGLE deterministic
forward pass: `composed_zero = where(g <= tau, 1, P_NB(0))`. Production emits something else:

  * S = D x K = 4 x 4 = 16 columns (`hydranet_inference.py:1578`), not one;
  * each of the D passes runs with **MC-dropout ON** — `model.eval()` plus
    `set_locked_dropout(True)` and one `reset_locked_dropout()` per pass (ADR-057), so every pass
    has its own gate AND its own (mu, theta). `vehicle.load_vehicle` calls `.eval()` only, which
    makes dropout the identity, so T0 measured the posterior MEAN pass, not a posterior draw;
  * the zero mass is a count of REALISED draws, not an analytic P(0).

So T0's attribution is an expectation evaluated at one point, and the emitted object is a 16-column
sample. `t0_verify_on_emitted_draws.py` already found the consequence: the archive's own draws give
a 0.3750 positive-draw fraction against T0's 0.4714 (opponent's B7, the ~0.1 accounting gap).

This script walks that gap one axis at a time, so whatever closes it is named:

  A0 1 pass, dropout OFF, analytic P(0)       <- T0 as published, reproduced here
  A  1 MC-dropout pass, analytic P(0)         <- adds the dropout switch only
  B  D=4 MC-dropout passes, analytic P(0)     <- adds MC averaging over the D axis
  C  D=4 MC-dropout passes, 16 realised draws <- adds sampling; THIS is what production emits

and reports the headline the composition fix rests on: of the emitted zeros, what share is the
BODY's own draw and what share is the GATE's mask. T0 inferred 94% for the body; rung C
measures it.

⛔ WHAT THIS DOES **NOT** SETTLE. The archive's 0.3750 is one forecast step (h1) of a free-running
roll at a different origin; every rung here is a 35-step teacher-forced sweep over 421-455. The two
are therefore NOT on a common basis, and the residue between rung C and 0.3750 is reported as
UNMATCHED — indicative of the free-running roll's contribution, not a measurement of it. Settling
B7 needs these rungs run on the archive's own origin and horizon.

⛔ NOT REPRODUCIBLE RUN-TO-RUN. `LockedDropout.forward` draws its mask with `torch.rand_like`
from the GLOBAL torch RNG (`locked_dropout.py:71`) — no generator is threaded in, and nothing here
seeds it. Two runs of this script therefore give different dropout masks: measured across two runs,
the body's share of the emitted zeros was 99.0% then 99.1% (stable), but the dropout delta A0->A
was -0.063 then -0.049 and the gate-kill rate 4.6% then 3.9%. Read the single-axis deltas as a
range, not a point, and read the per-pass sd printed below as the honest error bar. The same gap
applies to production inference, where only the FAMILY draw is seeded
(`hydranet_inference.py:1660`, from `config["torch_seed"]`) while the dropout mask is not —
worth an issue, not chased here.

⚠️ IN-SAMPLE. Months 421-456 are inside the calibration training partition 121-456 (prereg
Amendment: the leakage caveat). That is accepted: the question is the mechanism of composition,
not skill, and both arms of the attribution are computed on the same cell-steps.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from views_hydranet.distributions.composition import compose_samples
from views_hydranet.distributions.nb_core import NBCore
from views_hydranet.distributions.registry import resolve_family

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import (  # noqa: E402
    GATE_THRESHOLD,
    SB,
    active_cells_top100,
    build_volume,
    grid_rowcol,
    load_vehicle,
    pull,
)

ART = sys.argv[1] if len(sys.argv) > 1 else None
if ART is None:
    raise SystemExit("usage: p0_zero_attribution.py <path to calibration_model_*.pt>")
SP = Path(__file__).parent
POOL_LO, POOL_HI = 421, 456
WINDOW = list(range(POOL_LO, POOL_HI + 1))

# The production posterior width, from the delivered config (views-models violet_visitor).
POSTERIOR_D = 4          # n_posterior_samples (MC-dropout passes, model uncertainty)
POSTERIOR_K = 4          # n_head_samples (per-cell family draws, outcome uncertainty)
TORCH_SEED = 42          # the arm's configured torch_seed; the stream differs from the pod run's
#                          (different T and origin), so MC error across passes is reported below.
ARCHIVE_POSITIVE = 0.3750   # t0_verify_on_emitted_draws.py, h1, free-running — NOT matched (above)

model, cfg, sha = load_vehicle(ART)
family = resolve_family(cfg["output_distribution"])
if family is None:
    raise SystemExit(f"VOID: no family for output_distribution={cfg['output_distribution']!r}")
if not hasattr(model, "set_locked_dropout"):
    raise SystemExit("VOID: the vehicle has no set_locked_dropout; cannot replicate MC-dropout.")
df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
cells = active_cells_top100(df, POOL_LO, POOL_HI)
vol, land, pg_grid = build_volume(df, WINDOW)
r, c = grid_rowcol(cells)
ri, ci = r - 1, c - 1
n_steps = vol.shape[1] - 1
npar = family.n_params


def sweep(n_passes: int, dropout: bool):
    """-> (gate [P,T,100], analytic body P(0) [P,T,100], log1p body draws [P,T,100,K]).

    ADR-057: dropout submodules go into train() with a LOCKED mask while the rest of the net stays
    in eval(), so BatchNorm running stats are untouched. A bare `model.train()` here would rewrite
    BN buffers across every lead-in step and confound the run — the C-184/#289 trap, which my own
    T2 nearly walked into. Asserted, not assumed.
    """
    model.set_locked_dropout(dropout)
    for m in model.modules():
        if "BatchNorm" in type(m).__name__ and m.training:
            raise SystemExit(f"VOID: {type(m).__name__} in train() mode — BN would be rewritten.")
    g_o = np.zeros((n_passes, n_steps, len(cells)), dtype=np.float64)
    pz_o = np.zeros_like(g_o)
    bd_o = np.zeros((n_passes, n_steps, len(cells), POSTERIOR_K), np.float32)
    with torch.no_grad():
        for d in range(n_passes):
            if dropout:
                model.reset_locked_dropout()   # one fresh locked mask per posterior sample
            h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
            for t in range(n_steps):
                out = model(vol[:, t], h)
                h = out.h_next
                g_o[d, t] = torch.sigmoid(out.cls[0, SB])[ri, ci].numpy()
                # out.reg is ALREADY activated: the head applies the family's activation
                # (`_family_activation`, HydraBNrecurrentUnet_06_LSTM4.py:119) because the config
                # carries reg_activation=None. Do NOT softplus again.
                prm = out.reg[:, SB * npar:(SB + 1) * npar, :, :].permute(0, 2, 3, 1)[0][ri, ci]
                pz_o[d, t] = torch.exp(NBCore.log_prob_zero(prm[..., 0], prm[..., 1])).numpy()
                # Production's per-(pass, step) sub-generator (sampling.py:99) — independent
                # streams so a step's draws depend only on its own params.
                seed_t = (TORCH_SEED + d * 1_000_003 + t * 10_007) & 0x7FFF_FFFF_FFFF_FFFF
                bd_o[d, t] = torch.log1p(
                    family.sample(prm, POSTERIOR_K, torch.Generator().manual_seed(seed_t))
                ).numpy()
    return g_o, pz_o, bd_o


def analytic_positive(gate, pz) -> float:
    """threshold_gate's composed POSITIVE mass, median over cell-steps (T0's own statistic)."""
    return 1 - float(np.median(np.where(gate < GATE_THRESHOLD, 1.0, pz)))


G_off, PZ_off, _ = sweep(1, dropout=False)        # rung A0 — reproduces T0
G, PZ, BODY = sweep(POSTERIOR_D, dropout=True)    # rungs A / B / C

# --- compose exactly as production does: threshold_gate keeps the cell where gate >= tau ---
emitted = compose_samples(
    torch.from_numpy(BODY), torch.from_numpy(G).float(), "threshold_gate", GATE_THRESHOLD
).numpy()
kept = G >= GATE_THRESHOLD                                               # [D,T,100]
body_zero = BODY == 0.0                                                  # [D,T,100,K]
emit_zero = emitted == 0.0
gate_zero = emit_zero & ~body_zero          # zero ONLY because the mask killed it

a0 = analytic_positive(G_off, PZ_off)
per_pass_analytic = np.array([analytic_positive(G[d], PZ[d]) for d in range(POSTERIOR_D)])
a = float(per_pass_analytic.mean())
b = analytic_positive(G, PZ)
cc = 1 - float(emit_zero.mean())

print("=" * 78)
print("P0 — zero attribution on the PRODUCTION posterior (S = %d x %d = %d)"
      % (POSTERIOR_D, POSTERIOR_K, POSTERIOR_D * POSTERIOR_K))
print("=" * 78)
print("  cells %d · steps %d (months %d-%d, IN-SAMPLE) · tau %.2f · sha %s…"
      % (len(cells), n_steps, WINDOW[0], WINDOW[-2], GATE_THRESHOLD, sha[:12]))
print()
print("  RUNG A0  1 pass, dropout OFF, analytic   positive mass %.4f   <- T0 reproduced" % a0)
print("  RUNG A   1 dropout pass, analytic        positive mass %.4f   (mean of %d passes: %s)"
      % (a, POSTERIOR_D, " ".join("%.4f" % v for v in per_pass_analytic)))
print("  RUNG B   D=%d dropout passes, analytic    positive mass %.4f" % (POSTERIOR_D, b))
print("  RUNG C   D=%d dropout, %d realised draws  positive mass %.4f   <- production emits this"
      % (POSTERIOR_D, POSTERIOR_D * POSTERIOR_K, cc))
print("           reference: 0.4062 delivered · %.4f archive (h1, UNMATCHED) · "
      "0.7639 observed (0.6944 honest)" % ARCHIVE_POSITIVE)
print()
print("  ATTRIBUTION of the emitted zeros (rung C, all %d columns):"
      % (POSTERIOR_D * POSTERIOR_K))
print("    emitted zero fraction            %.4f" % emit_zero.mean())
print("    of those, the BODY's own draw    %.1f%%   <- T0 inferred 94%%"
      % (100 * body_zero.sum() / emit_zero.sum()))
print("    of those, the GATE's mask alone  %.1f%%" % (100 * gate_zero.sum() / emit_zero.sum()))
print("    cell-steps killed by the gate    %.1f%%  (dropout OFF: %.1f%%)"
      % (100 * (~kept).mean(), 100 * (G_off < GATE_THRESHOLD).mean()))
print()
print("  MC error across the D=%d passes (emitted zero fraction per pass):" % POSTERIOR_D)
per_pass = emit_zero.reshape(POSTERIOR_D, -1).mean(1)
print("    " + "  ".join("%.4f" % v for v in per_pass) + "   sd %.4f" % per_pass.std(ddof=1))
print()
print("  WHAT MOVED THE NUMBER (one axis at a time, positive mass):")
print("    the dropout switch   (A0->A)     %+.4f" % (a - a0))
print("    MC averaging over D  (A->B)      %+.4f" % (b - a))
print("    realised draws       (B->C)      %+.4f" % (cc - b))
print("    residue to the archive           %+.4f  ⛔ UNMATCHED (h1 free-running vs 35-step "
      "teacher-forced) — indicative only, B7 NOT settled" % (ARCHIVE_POSITIVE - cc))
print()
print("  INCIDENTAL — what dropout does to the two halves:")
print("    gate median        OFF %.4f   ON %.4f" % (np.median(G_off), np.median(G)))
print("    body P(0) median   OFF %.4f   ON %.4f" % (np.median(PZ_off), np.median(PZ)))
np.savez(SP / "p0_result.npz", cells=cells, G=G, PZ=PZ, BODY=BODY, G_off=G_off, PZ_off=PZ_off,
         sha=sha, D=POSTERIOR_D, K=POSTERIOR_K, tau=GATE_THRESHOLD)
print("\nwrote p0_result.npz")
