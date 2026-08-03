"""S3 (#233, Epic #230): ex-ante risk stratum + origin-block-bootstrap Giacomini–White test.

ADDITIVE to the v2 ruler — it does NOT touch `score_v2_horizons.py`'s `_metric_row` (so the
h=1 frozen-T=0 anchor is preserved by construction). It composes the frozen `crps_ensemble` and the
rollout ruler's gather/support/truth machinery to score the mixture (Epic #230) against the NB
baseline on a covariate-stratified proper CRPS + a conditional predictive-ability verdict.

Correctness guards (upfront review, 2026-08-01):
  * C-248 (Tier 1): `exante_stratum` reads PRE-ORIGIN truth ONLY (months [m0-window, m0-1] < m0):
    is structurally independent of the scored-horizon outcome. Never stratify on the current truth.
  * C-253 (Tier 2): `gw_stratified_test` uses an ORIGIN-BLOCK bootstrap (resample whole origins),
    robust to the origin×cell panel's serial + spatial dependence; plain per-obs resampling
    understates the SE → false significance.
  * C-252 (Tier 3): the test consumes per-cell CRPS VECTORS; `score_gw_v2` drops the (N,S) cubes.
  * C-254: the GW verdict binds on the number of origins P — reported as `n_origins`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def exante_stratum(truth, support, target, *, window: int = 12, target_col: "str | None" = None):
    """Ex-ante high-risk stratum: True where the cell had ANY conflict in the recent pre-origin
    window [m0-window, m0-1] (strictly BEFORE the origin m0 — so before the scored outcome at
    m0+h-1). `truth` is a DataFrame or parquet path with (month_id, priogrid_id, lr_<target>_best);
    `support` is the list of (m0, unit) keys. Returns a bool array aligned to `support`.

    C-248: the returned mask is a function of pre-origin truth ONLY; it never reads month >= m0, so
    permuting the scored-horizon outcome cannot change it (see the leakage regression test)."""
    df = truth if isinstance(truth, pd.DataFrame) else pd.read_parquet(truth)
    col = target_col or f"lr_{target}_best"
    if "month_id" not in df.columns or "priogrid_id" not in df.columns:
        df = df.reset_index()  # real v2 truth parquet keys (month_id, priogrid_id) live in the index
    lookup = df.set_index(["month_id", "priogrid_id"])[col].to_dict()
    out = np.zeros(len(support), dtype=bool)
    for i, (m0, u) in enumerate(support):
        m0 = int(m0)
        total = 0.0
        for mm in range(m0 - window, m0):  # [m0-window, m0-1]  — strictly pre-origin (< m0)
            total += float(lookup.get((mm, u), 0.0))
            if total > 0:
                break
        out[i] = total > 0
    return out


def _bootstrap_mean_ci(values, group_ids, n_boot, seed, ci, resample):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if resample == "origin":
        uniq = np.unique(group_ids)
        arrs = [values[np.where(group_ids == g)[0]] for g in uniq]
        ng = len(uniq)
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, ng, ng)  # resample WHOLE origins with replacement
            boots[b] = np.concatenate([arrs[i] for i in pick]).mean()
    elif resample == "obs":
        n = values.size
        boots = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    else:
        raise ValueError(f"resample must be 'origin' or 'obs', got {resample!r}")
    a = (1.0 - ci) / 2.0
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1.0 - a)), boots


def gw_stratified_test(
    loss_nb,
    loss_mix,
    stratum,
    origin_ids,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
    resample: str = "origin",
):
    """Conditional predictive-ability verdict on the high-risk stratum. `loss_*` are per-cell CRPS
    vectors (NOT cubes — C-252); `stratum` selects the ex-ante high-risk cells; `origin_ids`
    clusters the bootstrap (C-253). The differential d = crps_nb - crps_mix is POSITIVE for mixture
    has the lower (better) CRPS. Returns the mean differential, the %-reduction vs NB, an
    origin-block-bootstrap CI + 2-sided p-value, and the effective origin count (C-254)."""
    loss_nb = np.asarray(loss_nb, dtype=float)
    loss_mix = np.asarray(loss_mix, dtype=float)
    stratum = np.asarray(stratum, dtype=bool)
    origin_ids = np.asarray(origin_ids)
    d = (loss_nb - loss_mix)[stratum]  # + => mixture LOWER crps => better
    oids = origin_ids[stratum]
    nb_s = loss_nb[stratum]
    if d.size == 0:
        raise ValueError("empty stratum — no cells to score")
    point = float(d.mean())
    denom = float(nb_s.mean())
    pct = point / denom if denom > 0 else float("nan")
    lo, hi, boots = _bootstrap_mean_ci(d, oids, n_boot, seed, ci, resample)
    p = 2.0 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return dict(
        mean_diff=point,
        pct_reduction=pct,
        ci_lo=lo,
        ci_hi=hi,
        p_boot=float(min(p, 1.0)),
        n_stratum=int(stratum.sum()),
        n_origins=int(np.unique(oids).size),
        significant=bool(lo > 0 or hi < 0),
    )


def score_gw_v2(
    nb_entry,
    mix_entry,
    targets=("sb", "ns", "os"),
    horizons=(1, 18, 36),
    truth_parquet=None,
    window: int = 12,
    n_boot: int = 2000,
    seed: int = 0,
):
    """Driver-facing scorer: gather the NB and mixture arms, compute per-cell CRPS with the frozen
    `crps_ensemble`, build the ex-ante stratum, and run `gw_stratified_test` per (target, horizon).
    Additive — reuses the frozen primitives verbatim, retains only per-cell CRPS vectors, drops the
    (N,S) cubes (C-252). `*_entry` = (label, pred_dir, lr_tmpl, by_tmpl_or_None)."""
    import sys
    from pathlib import Path

    hn = Path(__file__).resolve().parents[3]  # repo root (reports/<dossier>/tools/ → 3 up); relative, not hardcoded (avoids C-247)
    for sub in (
        "reports/2026-07-17_lodestar_eval_dossier/tools",
        "reports/2026-07-25_t0_rollout_skill_dossier/tools",
    ):
        p = str(hn / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from lodestar_score import crps_ensemble  # noqa: E402
    from rollout_skill_score import _support_keys, _truth_map, gather_all_horizons  # noqa: E402

    if truth_parquet is None:
        truth_parquet = str(
            hn / "reports/2026-07-28_datafactory_migration_dossier/tools/v2_truth"
            "/calibration_datafactory_df.parquet"
        )
    horizons = tuple(horizons)
    rows = []
    for tgt in targets:
        lr_full = "lr_" + tgt + "_best"
        support = sorted(
            set.intersection(
                _support_keys(nb_entry[1], nb_entry[2].format(t=tgt), horizons),
                _support_keys(mix_entry[1], mix_entry[2].format(t=tgt), horizons),
            )
        )
        if not support:
            raise ValueError(f"[{tgt}] EMPTY fixed support across the NB/mixture arms — FAIL")
        stratum = exante_stratum(truth_parquet, support, tgt, window=window)
        origin_ids = np.array([m0 for (m0, _u) in support])
        months = {m0 + h - 1 for (m0, _u) in support for h in horizons}
        tmap = _truth_map(truth_parquet, lr_full, months)
        g_nb = gather_all_horizons(
            nb_entry[1],
            nb_entry[2].format(t=tgt),
            nb_entry[3].format(t=tgt) if nb_entry[3] else None,
        )
        g_mix = gather_all_horizons(
            mix_entry[1],
            mix_entry[2].format(t=tgt),
            mix_entry[3].format(t=tgt) if mix_entry[3] else None,
        )
        for h in horizons:
            truth = np.array([tmap[(m0 + h - 1, u)] for (m0, u) in support], dtype=float)
            c_nb = crps_ensemble(truth, np.stack([g_nb[(m0, h, u)][0] for (m0, u) in support]))
            c_mix = crps_ensemble(truth, np.stack([g_mix[(m0, h, u)][0] for (m0, u) in support]))
            r = gw_stratified_test(c_nb, c_mix, stratum, origin_ids, n_boot=n_boot, seed=seed)
            r.update(
                target=tgt,
                h=h,
                crps_stratum_nb=float(c_nb[stratum].mean()),
                crps_stratum_mix=float(c_mix[stratum].mean()),
                stratum_frac=float(stratum.mean()),
            )
            rows.append(r)
        del g_nb, g_mix  # C-252: cubes freed; only per-cell CRPS retained above
    return rows
