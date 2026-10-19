"""P1 — the MATCHED h=1 comparison against the archive's own draws (closes B7).

T0's 0.4714 and the archive's 0.3750 were never the same quantity: analytic vs sampled, months
421-455 in-sample vs month 457 out-of-sample, and — found while planning this — computed with two
different statistics. This measures both sides on ONE statistic at the SAME months.

WHY h=1 IS THE ONLY MATCHABLE SLICE (traced 2026-10-12):
  * `origin_N` = base origin month 456+N, forecasting 457+N .. 492+N. So h=1 exists at months
    457-469 across the 13 origins, all out-of-sample, 100 cells x 16 draws each.
  * h=1 is teacher-forced in BOTH the fed input AND the recurrent state.
    `hydranet_inference.py:1318` feeds the real observed input at the seed step (`t == origin`);
    the switch to fed-back prediction is `:1383-1388`, reached only for `t > origin`. The code
    states it at `:1219-1221` — "h=1 is untouched by construction". A teacher-forced sweep plus
    one step therefore IS production's h=1.
  * Pooling h=1 ACROSS origins is safe (13 distinct months). Pooling anything else is not:
    stride is 1, so adjacent origins share 35 of 36 months and would double-count at mixed
    horizons.
  * WEIGHT-MATCHED, not substituted: t0_result.npz's sha == PROVENANCE == .pt.sha256 == 4acbf4d3….
    Amendment 1's config-identity caveat does not apply to this comparison.

⛔ THE STATISTIC IS FIXED AND SINGLE. Median over cells of the per-cell positive-draw fraction —
exactly what the prereg's 0.3750 / 0.4062 / 0.7639 column already is (0.3750 = 6/16 confirms the
1/16 quantisation of a per-cell draw fraction). The pooled mean is printed as a SEPARATE column and
is never compared against the median one. Mixing those two is what manufactured the phantom B7.

LEAD-IN LENGTH IS MEASURED, NOT ASSUMED. Production digests the full partition from month 121
before the origin; T0 digested only 36 months (421-456) from a zero state. Since the cell state
carries a spatial map and drives placement (M50/M54/M60), that is a candidate explanation for the
gap, so both lead-ins are run and reported.

⚠️ ONE KNOWN DEPARTURE FROM PRODUCTION. Production calls `predict()` once per origin, each with its
own `reset_locked_dropout()`, so each origin gets a fresh mask. This sweeps once per pass and reads
all 13 origins off it, so the 13 share a mask. Reported per-origin so the spread is visible.
"""
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

from views_hydranet.distributions.nb_core import NBCore
from views_hydranet.distributions.registry import resolve_family
from views_hydranet.infrastructure.reproducibility_gate import ReproducibilityGate

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
ARCHIVE = sys.argv[2] if len(sys.argv) > 2 else None
if ART is None or ARCHIVE is None:
    raise SystemExit("usage: p1_matched_h1.py <calibration_model_*.pt> <predictions_*.tar.zst>")
SP = Path(__file__).parent

POSTERIOR_D, POSTERIOR_K = 4, 4
NP_SEED, TORCH_SEED = 42, 42        # Correction 4: seed the GLOBAL rng, as the manager does
BASE_ORIGIN = 456                   # origin_N = BASE_ORIGIN + N  (run.log: "Seq 0: origin 456")
N_ORIGINS = 13
LEADIN_STARTS = (121, 421)          # faithful (production digests from 121) and T0-style
LAST_FED = BASE_ORIGIN + N_ORIGINS - 1          # 468: feeding it predicts 469 = origin_12's h1

# ⛔ The one statistic. Both sides go through this function and nothing else.
def median_per_cell_positive(draws_cells_by_s: np.ndarray) -> float:
    """[n_cells, S] emitted draws -> median over cells of the per-cell positive-draw fraction."""
    return float(np.median((draws_cells_by_s > 0).mean(axis=1)))


ReproducibilityGate.lock_entropy(np_seed=NP_SEED, torch_seed=TORCH_SEED)
model, cfg, sha = load_vehicle(ART)
family = resolve_family(cfg["output_distribution"])
if family is None:
    raise SystemExit(f"VOID: no family for output_distribution={cfg['output_distribution']!r}")
if not hasattr(model, "set_locked_dropout"):
    raise SystemExit("VOID: the vehicle has no set_locked_dropout; cannot replicate MC-dropout.")
npar = family.n_params

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
cells = active_cells_top100(df, 421, 456)
r, c = grid_rowcol(cells)
ri, ci = r - 1, c - 1

# ---------- the archive side: read it, never recall it ----------
print("reading the archive's own h=1 draws …")
import zstandard  # noqa: E402

with open(ARCHIVE, "rb") as fh:
    raw = zstandard.ZstdDecompressor().stream_reader(fh).read()
archive = {}
with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
    root = tf.getnames()[0].split("/")[0]
    for n in range(N_ORIGINS):
        d = f"{root}/origin_{n}/lr_sb_best"
        y = np.load(io.BytesIO(tf.extractfile(f"{d}/y_pred.npy").read()))
        ids = np.load(io.BytesIO(tf.extractfile(f"{d}/identifiers.npz").read()))
        time_, unit = np.asarray(ids["time"]).ravel(), np.asarray(ids["unit"]).ravel()
        h1 = time_.min()
        if h1 != BASE_ORIGIN + 1 + n:
            raise SystemExit(f"VOID: origin_{n} h1 is month {h1}, expected {BASE_ORIGIN + 1 + n}.")
        order = {u: i for i, u in enumerate(cells)}
        sel = (time_ == h1) & np.isin(unit, cells)
        rows, units = y[sel], unit[sel]
        if len(rows) != len(cells):
            raise SystemExit(f"VOID: origin_{n} has {len(rows)} of {len(cells)} cells at h1.")
        archive[n] = (h1, rows[np.argsort([order[u] for u in units])])
del raw
print(f"  13 origins, h1 months {archive[0][0]}-{archive[N_ORIGINS - 1][0]}, "
      f"{archive[0][1].shape[1]} draws/cell")


def sweep(leadin_start: int, n_passes: int, dropout: bool):
    """Teacher-forced sweep; read gate, (mu,theta) and K draws at each origin's h=1.

    ADR-057: dropout submodules go to train() with a LOCKED mask while the net stays in eval(), so
    BatchNorm running stats are untouched. A bare model.train() would rewrite BN buffers on every
    lead-in step — the C-184/#289 trap. Asserted, not assumed.
    """
    model.set_locked_dropout(dropout)
    for m in model.modules():
        if "BatchNorm" in type(m).__name__ and m.training:
            raise SystemExit(f"VOID: {type(m).__name__} in train() mode — BN would be rewritten.")
    months = list(range(leadin_start, LAST_FED + 1))
    vol, _, _ = build_volume(df, months)
    g = np.zeros((n_passes, N_ORIGINS, len(cells)))
    pz = np.zeros_like(g)
    dr = np.zeros((n_passes, N_ORIGINS, len(cells), POSTERIOR_K), np.float32)
    with torch.no_grad():
        for d in range(n_passes):
            if dropout:
                model.reset_locked_dropout()
            h = torch.zeros(1, cfg["total_hidden_channels"], *vol.shape[-2:])
            for t, fed in enumerate(months):
                out = model(vol[:, t], h)
                h = out.h_next
                if fed < BASE_ORIGIN:
                    continue                      # lead-in: state only, nothing read
                n = fed - BASE_ORIGIN             # feeding 456+n predicts 457+n = origin_n's h1
                g[d, n] = torch.sigmoid(out.cls[0, SB])[ri, ci].numpy()
                # out.reg is ALREADY activated (the head applies the family's activation,
                # HydraBNrecurrentUnet_06_LSTM4.py:119, since reg_activation=None). No softplus.
                prm = out.reg[:, SB * npar:(SB + 1) * npar, :, :].permute(0, 2, 3, 1)[0][ri, ci]
                pz[d, n] = torch.exp(NBCore.log_prob_zero(prm[..., 0], prm[..., 1])).numpy()
                seed_t = (TORCH_SEED + d * 1_000_003 + t * 10_007) & 0x7FFF_FFFF_FFFF_FFFF
                dr[d, n] = family.sample(
                    prm, POSTERIOR_K, torch.Generator().manual_seed(seed_t)
                ).numpy()
    return g, pz, dr


def compose(g, dr):
    """threshold_gate: keep the cell where gate >= tau, else zero it. -> [P,N,cells,K] counts."""
    return np.where((g < GATE_THRESHOLD)[..., None], 0.0, dr)


results = {}
for start in LEADIN_STARTS:
    n_fed = LAST_FED - start + 1
    print(f"\nsweeping lead-in {start}-{LAST_FED} ({n_fed} steps x {POSTERIOR_D + 1} passes) …")
    results[start] = {
        "off": sweep(start, 1, dropout=False),
        "on": sweep(start, POSTERIOR_D, dropout=True),
    }

print("\n" + "=" * 86)
print("P1 — MATCHED h=1: reconstruction vs the archive's own draws")
print("=" * 86)
print("  statistic: MEDIAN over %d cells of the per-cell positive-draw fraction" % len(cells))
print("  weights 4acbf4d3… (archive and vehicle are the SAME artifact) · tau %.2f · S=%d"
      % (GATE_THRESHOLD, POSTERIOR_D * POSTERIOR_K))

obs = {}
for n in range(N_ORIGINS):
    h1 = archive[n][0]
    w = df[(df.month_id == h1) & (df.priogrid_id.isin(cells))]
    obs[n] = float((w.ged_sb_best > 0).mean())

print("\n  PER ORIGIN (median per-cell positive fraction)")
print("  %-7s %-6s %9s %9s %11s %11s" % ("origin", "month", "observed", "ARCHIVE",
                                          "recon(121)", "recon(421)"))
for n in range(N_ORIGINS):
    row = [median_per_cell_positive(
        compose(results[s]["on"][0], results[s]["on"][2])[:, n].transpose(1, 0, 2).reshape(
            len(cells), -1)) for s in LEADIN_STARTS]
    print("  %-7d %-6d %9.4f %9.4f %11.4f %11.4f"
          % (n, archive[n][0], obs[n], median_per_cell_positive(archive[n][1]), row[0], row[1]))

print("\n  POOLED over the 13 h=1 slices")
arch_all = np.concatenate([archive[n][1] for n in range(N_ORIGINS)], axis=0)
obs_pooled = float(np.mean(list(obs.values())))
print("  %-34s %s" % ("observed truth (13 months)", "%.4f" % obs_pooled))
print("  %-34s %.4f   <- ground truth, read from the archive"
      % ("ARCHIVE, emitted draws", median_per_cell_positive(arch_all)))
for s in LEADIN_STARTS:
    g_on, pz_on, dr_on = results[s]["on"]
    g_off, pz_off, _ = results[s]["off"]
    em = compose(g_on, dr_on).transpose(2, 0, 1, 3).reshape(len(cells), -1)
    rec = median_per_cell_positive(em)
    ana_on = float(np.median(1 - np.where(g_on < GATE_THRESHOLD, 1.0, pz_on).mean((0, 1))))
    ana_off = float(np.median(1 - np.where(g_off < GATE_THRESHOLD, 1.0, pz_off).mean((0, 1))))
    print("  lead-in %d:" % s)
    print("    %-32s %.4f" % ("rung A0 dropout OFF, analytic", ana_off))
    print("    %-32s %.4f" % ("rung B  D=4 dropout, analytic", ana_on))
    print("    %-32s %.4f   residue vs archive %+.4f"
          % ("rung C  D=4, 16 realised draws", rec,
             median_per_cell_positive(arch_all) - rec))

print("\n  SECOND COLUMN, NOT COMPARABLE TO THE ABOVE — pooled mean positive fraction")
print("    archive %.4f" % (arch_all > 0).mean())
for s in LEADIN_STARTS:
    g_on, _, dr_on = results[s]["on"]
    print("    reconstruction, lead-in %d  %.4f" % (s, (compose(g_on, dr_on) > 0).mean()))

print("\n  REFERENCE, in-sample 421-455 (P0): rung C 0.4290 pooled / 0.4259 median-over-cells")
np.savez(SP / "p1_result.npz", cells=cells, sha=sha,
         archive=np.stack([archive[n][1] for n in range(N_ORIGINS)]),
         months=np.array([archive[n][0] for n in range(N_ORIGINS)]),
         observed=np.array([obs[n] for n in range(N_ORIGINS)]),
         **{f"g_on_{s}": results[s]["on"][0] for s in LEADIN_STARTS},
         **{f"pz_on_{s}": results[s]["on"][1] for s in LEADIN_STARTS},
         **{f"dr_on_{s}": results[s]["on"][2] for s in LEADIN_STARTS},
         **{f"g_off_{s}": results[s]["off"][0] for s in LEADIN_STARTS},
         **{f"pz_off_{s}": results[s]["off"][1] for s in LEADIN_STARTS})
print("\nwrote p1_result.npz")
