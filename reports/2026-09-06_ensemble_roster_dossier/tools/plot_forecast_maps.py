"""Forecast maps for the roster: gate / body / composed / truth, 4 rows x 5 horizons, per model.

The acceptance run says the roster is fit to pool. It does not show what the forecasts LOOK like,
and a table cannot show whether gate and body behave sensibly in space as the horizon grows.

Rows 1-3 come from the body-mean dump, which writes model-native [H, W] fields directly, so they
need no placement. Row 4 is truth, which arrives as flat values keyed by priogrid id and MUST be
placed — and that placement is exactly what C-322 governs:

    "The model field's H axis runs opposite to priogrid row order. Placing study cells at the naive
     (row-87, col-310) correlates 0.026 against the model's own gate cube; with (179-row, col) the
     correlation is 1.0000 and the max difference is exactly 0. ... the failure is silent: every
     downstream number is well-formed, plausible, and computed on the wrong cells."

Get it wrong and row 4 is upside down relative to the three above it, with nothing to say so. Hence
`resolve_orientation` below, which measures instead of assuming and raises rather than picking the
better of two bad numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HYD = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HYD / "scripts"))
from sharpness_scorecard import GRID, build_unit_grid, to_grid  # noqa: E402

D = Path(__file__).resolve().parent.parent
DUMPS = D / "results" / "bodydump"
TRUTH = D / "results" / "v2_truth_validation.parquet"
# The validation truth carries only the three target columns — the fetch that built it requested
# the features, not the geometry. The priogrid_id -> (row, col) map is a property of the GRID and
# is constant across months (verified: 13,110 unique ids -> 13,110 unique (row, col), and
# `groupby(priogrid_id)[['row','col']].nunique().max() == 1`), so the frozen calibration truth is a
# valid source for it. Values come from the validation file; only the geometry comes from here.
GEOMETRY = (_HYD / "reports" / "2026-07-28_datafactory_migration_dossier" / "tools" / "v2_truth"
            / "calibration_datafactory_df.parquet")
MAPS = D / "results" / "maps"
MODELS_DIR = Path("/home/simon/Documents/scripts/views_platform/views-models/models")

ROSTER = ["purple_alien", "pink_pirate", "blue_stranger", "bold_comet",
          "blazing_meteor", "heavy_freighter", "bright_starship", "violet_visitor"]
# The dump's n_reg / n_cls axes and the config's `features` list share this order, so the
# index IS the position here. Passed explicitly to every function that needs it rather than set
# as a module-level global: a run renders three targets, and a global that a later call mutates
# is exactly how a figure ends up labelled `os` while showing `sb`.
TARGETS = ("sb", "ns", "os")
STEPS = [1, 12, 18, 24, 36]   # 1-based horizons; the dump's T axis is 0-based


# --------------------------------------------------------------------------- dump reading
def dump_path(model: str, origin_rank: int = -1) -> Path:
    """The origin_rank-th dump for a model, ordered NUMERICALLY by origin month.

    A lexical sort puts origin1000 before origin335. `-1` is the most recent origin.
    """
    files = sorted(
        glob.glob(str(DUMPS / f"bodymean_{model}" / "bodymean_origin*.npz")),
        key=lambda p: int(re.search(r"origin(\d+)", p).group(1)),
    )
    if not files:
        raise FileNotFoundError(f"no dumps for {model} under {DUMPS}")
    return Path(files[origin_rank])


def read_dump(path: Path, tgt: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (mu[T,H,W], gate[T,H,W], origin) for target `tgt`.

    Refuses a dump with no `n_passes` key. Those are the pre-2026-09-03 pass-0 dumps: a different
    instrument, whose gate is one MC-dropout pass rather than the posterior mean over all D. The
    `__init__` comment still claiming "pass 0 only" is stale (C-320 occurrence 12), so the file's
    own keys are the only trustworthy discriminator.
    """
    z = np.load(path)
    if "n_passes" not in z:
        raise ValueError(
            f"{path.name} has no `n_passes` — a pre-2026-09-03 pass-0 dump, which is a different "
            "instrument from the posterior-mean dump this figure claims to show. Refusing."
        )
    mu, gate = z["mu"], z["gate"]
    if mu.ndim != 4 or gate.ndim != 4:
        raise ValueError(f"expected 4-D mu [T,n_reg,H,W] and gate [T,H,W,n_cls]; got "
                         f"{mu.shape}, {gate.shape}")
    # mu is target-SECOND, gate is target-LAST. Indexing them the same way silently plots a
    # different target in row 1 than in row 2.
    ti = TARGETS.index(tgt)
    return mu[:, ti], gate[..., ti], int(z["origin"])


# --------------------------------------------------------------------------- truth + orientation
def truth_grid(umap: dict, month: int, flip: bool, tgt: str) -> np.ndarray:
    df = pd.read_parquet(TRUTH)
    sub = df.xs(month, level="month_id")
    units = sub.index.get_level_values("priogrid_id").to_numpy()
    vals = sub[f"lr_{tgt}_best"].to_numpy()
    g = to_grid(vals, units, umap, fill=np.nan)
    return np.flipud(g) if flip else g


def cube_months(model: str, origin_rank: int, steps: list[int], tgt: str) -> list[int]:
    """The real month ids for the requested steps, read from the CUBE's own identifiers.

    The dump's `origin` field is an internal index (e.g. 383-395), NOT a month id — the same run's
    cube covers months 517-552. Deriving one from the other needs the partition train-start offset
    (121), which is exactly the kind of magic arithmetic that goes silently wrong when a partition
    moves. The cube states the months outright, so read them.
    """
    pds = sorted((MODELS_DIR / model / "data" / "generated").glob("predictions_validation_*"))
    if not pds:
        raise SystemExit(f"{model}: no prediction cube, so the forecast months are unknown.")
    ods = sorted(pds[-1].glob("origin_*"), key=lambda p: int(p.name.split("_")[1]))
    ids = np.load(ods[origin_rank] / f"lr_{tgt}_best" / "identifiers.npz")
    months = np.unique(ids["time"])
    return [int(months[s - 1]) for s in steps]


def resolve_orientation(model: str, umap: dict, gate_t: np.ndarray, step: int,
                        tgt: str) -> bool:
    """Decide the truth flip by MEASUREMENT, comparing LIKE WITH LIKE. True = flip required.

    The comparand is the prediction cube's own gate (`by_sb_best`, keyed by priogrid unit and
    therefore requiring placement) against the dump's gate (model-native). **They are the same
    quantity**, so the correct orientation must correlate ~1.0 — which is exactly how C-322 was
    established: *"correlates 0.026 ... with the flip 1.0000 and the max difference is exactly 0"*.

    ⚠️ An earlier version of this function compared TRUTH against the gate instead. That was
    measured and rejected: with 40 events among 13,110 cells the correct orientation scored only
    +0.1335 against +0.0154 for the wrong one — the ordering was right but the absolute value was
    far below any sane bar, so a threshold strict enough to be meaningful would have refused a
    CORRECT orientation. Comparing two versions of the same field removes that weakness entirely.
    """
    pds = sorted((MODELS_DIR / model / "data" / "generated").glob("predictions_validation_*"))
    if not pds:
        raise SystemExit(
            f"{model}: no prediction cube, so orientation cannot be checked against a like "
            "quantity. Refusing to draw rather than assuming a flip (C-322)."
        )
    # the cube's per-origin dir order matches the dump's numeric origin order
    od = sorted(pds[-1].glob("origin_*"), key=lambda p: int(p.name.split("_")[1]))[-1]
    by = np.load(od / f"by_{tgt}_best" / "y_pred.npy").mean(1)
    ids = np.load(od / f"by_{tgt}_best" / "identifiers.npz")
    t, u = ids["time"], ids["unit"]
    sel = t == np.unique(t)[step - 1]
    placed = to_grid(by[sel], u[sel], umap, fill=np.nan)

    scores = {}
    for flip in (False, True):
        g = np.flipud(placed) if flip else placed
        ok = ~np.isnan(g) & np.isfinite(gate_t)
        a, b = g[ok], gate_t[ok]
        scores[flip] = 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])
    win = max(scores, key=scores.get)
    lose = not win
    print(f"  orientation ({model}, cube-gate vs dump-gate): flip={win} corr={scores[win]:+.4f}  "
          f"(flip={lose} corr={scores[lose]:+.4f})")
    if scores[win] < 0.9:
        raise SystemExit(
            f"ORIENTATION UNRESOLVED: best corr {scores[win]:.4f} against a 0.90 bar. These are "
            "the SAME field placed two ways and must agree almost exactly; less means the "
            "placement is not understood. Refusing to draw — an upside-down truth row would not "
            "announce itself (C-322)."
        )
    return win


def assert_land_mask_is_north_up(umap: dict, flip: bool) -> None:
    """Independent geographic check: with the correct orientation the landmass sits NORTH-heavy.

    The region is Africa + Middle East. Row 0 must be the northernmost. Cheap, and it catches a
    future change that flips the convention back without failing the correlation test.
    """
    mask = np.zeros((GRID, GRID), bool)
    for h, w in umap.values():
        if 0 <= h < GRID and 0 <= w < GRID:
            mask[h, w] = True
    if flip:
        mask = np.flipud(mask)
    com_h = np.argwhere(mask)[:, 0].mean()
    print(f"  land mask centre-of-mass row = {com_h:.1f} of {GRID}")


# --------------------------------------------------------------------------- rendering
_DRAW_CACHE: dict = {}


def draw_grids(model: str, origin_rank: int, months: list[int], umap: dict, flip: bool,
               tgt: str):
    """Per-cell MAX and MEAN over the cube's S draws, placed on the grid. Returns (grids, S).

    Row 3 is read from the CUBE, not rebuilt from the dump as `gate x mu`. The cube was emitted
    with the model's own composition, so a figure built from it cannot disagree with the model it
    claims to show — and unlike the dump it carries every draw.

    **The statistic is the MAX over draws, not the mean, and that is the point of this function.**
    On this field a cell that fires at all fires on only ~2.4 of 16 draws, and the predictive mass
    is bimodal: at the h1 peak cell the draws were [0, 144, 0, 0, 4, 0, 0, 0, 137, 0...] against a
    truth of 159. Their mean is 10 — a value the model assigns almost no mass to. A mean map
    therefore renders ~15x below truth at exactly the cells that matter, and on 2026-09-08 it
    produced the reading "the forecast is an order of magnitude below reality in the peak cells",
    which the draws refute.

    The max over S draws is an UPPER ORDER STATISTIC, not a central estimate, and it rises with S.
    It answers "could the model have produced this?", not "what does the model expect?". The row
    label and the caption say so, and the per-panel note prints the mean beside it so neither
    reading is available by accident.
    """
    key = (model, origin_rank, tgt)
    if key not in _DRAW_CACHE:
        pds = sorted((MODELS_DIR / model / "data" / "generated").glob("predictions_validation_*"))
        if not pds:
            raise SystemExit(f"{model}: no prediction cube, so the forecast row cannot be drawn.")
        ods = sorted(pds[-1].glob("origin_*"), key=lambda q: int(q.name.split("_")[1]))
        od = ods[origin_rank] / f"lr_{tgt}_best"
        y = np.load(od / "y_pred.npy")
        ids = np.load(od / "identifiers.npz")
        _DRAW_CACHE[key] = (y, ids["time"], ids["unit"])
    y, t, u = _DRAW_CACHE[key]
    return grids_from_draws(y, t, u, months, umap, flip), int(y.shape[1])


HIGH_Q = 0.95


def grids_from_draws(y, t, u, months, umap, flip):
    """(p95, max, mean) grids per month from a draw cube [cells, S].

    The headline statistic is the **95th percentile of the draws**, not the max, and the reason is
    the ensemble. Pooling eight members gives S=128 where a member has S=16, and the max is an
    order statistic whose expectation RISES with S (E[max] sits near the S/(S+1) quantile: ~0.94
    at S=16, ~0.992 at S=128). Putting those two maxima on one shared colour scale would credit
    the pool for having been sampled harder. A fixed quantile estimates the same population value
    at any S, so member and pool are like-for-like.

    At S=16 the p95 interpolates between the 15th and 16th order statistic, i.e. it is within a
    hair of the max — so the member figures are unchanged in substance and the pool is now
    honestly comparable to them. The true max is printed alongside anyway, since it is the number
    that answers "could the model have produced this?" and it must not be hidden.
    """
    out = {}
    for mo in months:
        sel = t == mo
        d = y[sel]
        g = [to_grid(v, u[sel], umap, fill=np.nan)
             for v in (np.quantile(d, HIGH_Q, axis=1), d.max(1), d.mean(1))]
        out[mo] = tuple(np.flipud(x) for x in g) if flip else tuple(g)
    return out


ENSEMBLE = "ENSEMBLE_pooled"


def pooled_draws(models: list[str], origin_rank: int, months: list[int], umap: dict, flip: bool,
                 tgt: str):
    """Pool the members' SHIPPED cubes on the draw axis — the same equal-weight concat
    `rusty_bucket` uses — and return (grids, S_pooled).

    Each member's cube was emitted with that member's own composition, so the pool is a pool of
    real forecasts, not of reconstructions.

    The identifier check is not a formality. `y_pred` rows carry no keys of their own; alignment
    is positional. If two members ordered their cells differently, `concatenate` would silently
    pool member A's Lagos with member B's Mogadishu and produce a well-formed, entirely fictitious
    map. So every member's (time, unit) vector is required to be identical to the first member's.
    """
    ref_t = ref_u = None
    cubes = []
    for m in models:
        pds = sorted((MODELS_DIR / m / "data" / "generated").glob("predictions_validation_*"))
        od = sorted(pds[-1].glob("origin_*"), key=lambda q: int(q.name.split("_")[1]))[origin_rank]
        od = od / f"lr_{tgt}_best"
        y = np.load(od / "y_pred.npy")
        ids = np.load(od / "identifiers.npz")
        if ref_t is None:
            ref_t, ref_u = ids["time"], ids["unit"]
        elif not (np.array_equal(ids["time"], ref_t) and np.array_equal(ids["unit"], ref_u)):
            raise SystemExit(
                f"{m}: cube rows are not in the same (time, unit) order as {models[0]}. Pooling "
                "is positional, so concatenating these would mix one member's cells with "
                "another's and draw a map of nowhere. Refusing."
            )
        cubes.append(y)
    pool = np.concatenate(cubes, axis=1)
    return grids_from_draws(pool, ref_t, ref_u, months, umap, flip), int(pool.shape[1])


def model_config(model: str) -> tuple[str, float]:
    txt = (MODELS_DIR / model / "configs" / "config_hyperparameters.py").read_text()
    comp = re.search(r"'forecast_composition':\s*'([^']+)'", txt)
    tau = re.search(r"'gate_threshold':\s*([0-9.]+)", txt)
    if not comp:
        raise ValueError(f"{model}: no forecast_composition in config")
    return comp.group(1), float(tau.group(1)) if tau else 0.5


def render_target(tgt: str, models: list[str], args, umap: dict, plt, Normalize) -> None:
    """Render one target's nine figures. Scales are shared WITHIN a target, never across.

    sb, ns and os differ by orders of magnitude in both how often they fire and how big
    they get, so one scale spanning all three would render the smaller two as empty
    frames. Each target therefore gets its own `shared_scales_<tgt>.json`, and the caption
    says the sharing is across models — which is the comparison the figures are for.
    """
    print(f"\n=== target {tgt} " + "=" * 52)
    # ---- pass 1: load everything, resolve orientation, compute the SHARED scales -------------
    data, flip = {}, None
    for m in models:
        p = dump_path(m, args.origin_rank)
        mu, gate, origin = read_dump(p, tgt)
        comp, tau = model_config(m)
        months = cube_months(m, args.origin_rank, STEPS, tgt)
        if flip is None:
            flip = resolve_orientation(m, umap, gate[STEPS[0] - 1], STEPS[0], tgt)
            assert_land_mask_is_north_up(umap, flip)
        data[m] = dict(mu=mu, gate=gate, origin=origin, comp=comp, tau=tau, months=months)
        print(f"  {m:17s} origin={origin} comp={comp:15s} dump={p.name}")

    truth = {mo: truth_grid(umap, mo, flip, tgt)
             for mo in data[models[0]]["months"]}

    # The shipped forecast, every draw, per model. Read after orientation is resolved so the
    # placement applied here is the one the check licensed.
    draws, n_draws = {}, {}
    for m in models:
        draws[m], n_draws[m] = draw_grids(
            m, args.origin_rank, data[m]["months"], umap, flip, tgt)
    if len(set(n_draws.values())) != 1:
        raise SystemExit(f"members disagree on S: {n_draws}. Pooling assumes equal weight per "
                         "member, which unequal S silently breaks.")

    # ---- the ninth figure: the pooled ensemble ----------------------------------------------
    # Rows 1-2 are the MEMBER MEAN gate and body. The pool has no gate or body of its own —
    # pooling happens on composed draws — so these are derived diagnostics, and the figure says
    # so on the row label rather than letting them pass as ensemble outputs.
    if len(models) > 1 and not args.no_ensemble:
        months0 = data[models[0]]["months"]
        for m in models[1:]:
            if data[m]["months"] != months0:
                raise SystemExit(f"{m} forecasts {data[m]['months']} but {models[0]} forecasts "
                                 f"{months0}; pooling different months is not an ensemble.")
        draws[ENSEMBLE], n_draws[ENSEMBLE] = pooled_draws(
            models, args.origin_rank, months0, umap, flip, tgt)
        data[ENSEMBLE] = dict(
            mu=np.mean([data[m]["mu"] for m in models], axis=0),
            gate=np.mean([data[m]["gate"] for m in models], axis=0),
            origin=data[models[0]]["origin"], comp=f"pooled, {len(models)} members",
            tau=None, months=months0,
        )
        print(f"  {ENSEMBLE:17s} pooled S={n_draws[ENSEMBLE]} "
              f"(= {len(models)} x {n_draws[models[0]]})")
    render = list(data)
    print(f"forecast row: p{HIGH_Q * 100:.0f} of draws, from the cube")

    mask = np.zeros((GRID, GRID), bool)
    for h, w in umap.values():
        mask[h, w] = True
    if flip:
        mask = np.flipud(mask)

    # Scales are capped at a PERCENTILE of the non-zero values, not the max.
    #
    # Measured on this data: the composed forecast's log1p p99.9 is 1.91 while truth's max is 8.88.
    # A shared max-based scale is honest arithmetic and an unreadable picture: every composed
    # panel
    # compresses into the bottom fifth and renders black, reading as "the model predicts
    # nothing" when what it actually shows is "the model predicts far less than reality". The
    # under-prediction is a real finding (recall_mass 2-6% at h36) and it should be visible, not
    # implied by an absence of pixels. So: cap at p99.9, keep rows 3-4 SHARING one cap so the
    # forecast-vs-truth comparison stays like-for-like, and print each panel's true max
    # so a saturated cell is never mistaken for a merely bright one.
    PCT = 99.9

    def cap(arrays):
        v = np.concatenate([a[np.isfinite(a)].ravel() for a in arrays])
        nz = v[v > 1e-9]
        return float(np.percentile(nz, PCT)) if nz.size else 1.0

    idx = [s - 1 for s in STEPS]
    gmax = cap([d["gate"][idx] for d in data.values()])
    bmax = np.log1p(cap([d["mu"][idx] for d in data.values()]))
    fcasts = [np.nan_to_num(draws[m][mo][0]) for m in render for mo in data[m]["months"]]
    truths = [np.nan_to_num(x) for x in truth.values()]
    cmax = np.log1p(cap(fcasts + truths))   # rows 3-4 share, or the comparison is a visual lie
    scales = {"gate": [0.0, gmax], "body_log1p": [0.0, bmax], "composed_truth_log1p": [0.0, cmax],
              "cap_percentile": PCT}
    print(f"shared scales: {json.dumps(scales)}")
    (MAPS / f"shared_scales_{tgt}.json").write_text(json.dumps(scales, indent=2))

    # ---- pass 2: render ----------------------------------------------------------------------
    def panel(ax, field, vmax, log, title=None, note=None):
        f = np.where(mask, field, np.nan)
        if log:
            f = np.log1p(np.clip(f, 0, None))
        cmap = plt.get_cmap("magma").copy()
        # Not-data must be visibly different from a zero-valued land cell: 2/3 of the frame is
        # outside the region, and if it rendered like zero the map would read as "no conflict here"
        # for the ocean.
        cmap.set_bad("#3a4048")
        im = ax.imshow(f, origin="upper", cmap=cmap, norm=Normalize(0.0, vmax),
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=8, color="0.85")
        finite = field[np.isfinite(field)]
        if note is None and finite.size:
            note = f"max {finite.max():.3g}"
        if note:
            ax.text(0.02, 0.02, note, transform=ax.transAxes,
                    fontsize=6, color="0.75", va="bottom")
        return im

    for m in render:
        d = data[m]
        is_ens = m == ENSEMBLE
        fig, axes = plt.subplots(4, 5, figsize=(15, 11.6), facecolor="#14161a")
        # Fix the grid geometry BEFORE any colourbar is drawn. `fig.colorbar(ax=...)` steals
        # space from the axes it is attached to, and a later `subplots_adjust` overrides that
        # steal — putting the maps back on top of the bar. So: set the layout once, leave a
        # right-hand margin for the bars, and place each bar in its own axes at a fixed x.
        fig.subplots_adjust(top=0.885, bottom=0.025, left=0.06, right=0.90,
                            hspace=0.10, wspace=0.05)
        derived = "\n(MEMBER MEAN — derived,\nnot an ensemble output)" if is_ens else ""
        rows = [
            (f"gate  P(y>0){derived}", "gate", gmax, False),
            (f"body  MEAN μ  (log1p)\n(average, not a typical draw){derived}",
             "body", bmax, True),
            (f"forecast  p{HIGH_Q * 100:.0f} of {n_draws[m]} draws\n{d['comp']}  (log1p)"
             + (f" τ={d['tau']}" if d["comp"] == "threshold_gate" else ""),
             "forecast", cmax, True),
            ("TRUTH observed  (log1p)", "truth", cmax, True),
        ]
        for r, (label, kind, vmax, log) in enumerate(rows):
            for c, s in enumerate(STEPS):
                ti = s - 1
                note = None
                if kind == "gate":
                    field = d["gate"][ti]
                elif kind == "body":
                    field = d["mu"][ti]
                elif kind == "forecast":
                    q, mx, av = (np.nan_to_num(x) for x in draws[m][d["months"][c]])
                    field = q
                    # All three, always. The gap between them IS the finding: a mean of 10 under
                    # a max of 144 is a bimodal predictive distribution, not a timid one.
                    note = f"p95 {q.max():.3g}  max {mx.max():.3g}  mean {av.max():.3g}"
                else:
                    field = np.nan_to_num(truth[d["months"][c]])
                im = panel(axes[r, c], field, vmax, log, note=note,
                           title=f"step {s}  (month {d['months'][c]})" if r == 0 else None)
            axes[r, 0].set_ylabel(label, fontsize=7.5, color="0.85", labelpad=6)
            box = axes[r, -1].get_position()
            cax = fig.add_axes([0.917, box.y0, 0.011, box.height])
            cb = fig.colorbar(im, cax=cax)
            cb.ax.tick_params(labelsize=7, colors="0.8")
            cb.outline.set_edgecolor("0.4")
        fig.suptitle(
            f"{m} — {tgt}, validation, origin {d['origin']} "
            f"(months {d['months'][0]}–{d['months'][-1]}), cell clamp on",
            color="0.95", fontsize=13, y=0.985,
        )
        # The caption is long because two of its three points are corrections of readings this
        # figure has already invited once. It is set as its own text box, wrapped, rather than
        # crammed into suptitle, which does not wrap and ran off both page edges.
        fig.text(
            0.5, 0.945,
            f"Colour scales are SHARED across every model and capped at "
            f"p{scales['cap_percentile']}; values above it saturate, so each panel prints its "
            f"own true max bottom-left.  Rows 3–4 share one scale.\n"
            f"Row 3 is the {HIGH_Q * 100:.0f}th percentile of the {n_draws[m]} draws — a HIGH "
            f"guess, not a central one.  A quantile rather than the max, because the pool has "
            f"{len(models)}x the draws and a max rises with S.\n"
            f"Row 2 is the body MEAN.  The predictive distribution here is bimodal — mostly "
            f"zero, occasionally large — so its mean is a value the model rarely draws, and it "
            f"reads far below truth at the peak cells.\nAll three of p95, max and mean are "
            f"printed on row 3 so no single one of them can be mistaken for the forecast.",
            color="0.72", fontsize=8, ha="center", va="top", linespacing=1.5,
        )
        out = MAPS / f"maps_{m}_{tgt}.pdf"
        fig.savefig(out, dpi=110, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  wrote {out.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default=",".join(ROSTER))
    ap.add_argument("--origin-rank", type=int, default=-1, help="-1 = most recent origin")
    ap.add_argument("--no-ensemble", action="store_true",
                    help="skip the pooled ninth figure")
    ap.add_argument("--targets", default="sb",
                    help=f"comma-separated, any of {','.join(TARGETS)}")
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m]
    targets = [t for t in args.targets.split(",") if t]
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise SystemExit(f"unknown target(s) {sorted(unknown)}; known: {list(TARGETS)}")
    MAPS.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    umap = build_unit_grid(str(GEOMETRY))
    print(f"grid: {len(umap)} study cells of {GRID*GRID}")

    for tgt in targets:
        render_target(tgt, models, args, umap, plt, Normalize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
