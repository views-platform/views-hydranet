"""Cross-seed assembly for Wave 1 — applies the decision rule, never averages over disagreement.

The rule is M48's, reused rather than reinvented, and its limit is stated rather than hidden: a
paired sign-flip at n=4 has a floor of 1/16 = 0.0625 and CANNOT reach p <= 0.05. So the standard is

    SUPPORTED  4/4 seeds agree in sign AND |mean effect| exceeds the seed spread (sd)
    CONTESTED  3/4, or the effect sits inside the seed spread
    (no p-values are claimed, because none is reachable at this n)

Per-seed values are always printed next to the verdict. A mean that hides a 3/1 split is exactly
the failure `aggregate_seeds.py` refuses by design.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escalation import arm_rows as esc_rows  # noqa: E402
from subset_ap import arm_table as sub_table  # noqa: E402
from wave1_data import RAW, RESULTS, build_unit_grid, load_origins, load_truth  # noqa: E402

SEEDS = ["fullzero_fortytwo", "fullzero_fortythree", "fullzero_fortyfour", "fullzero_fortyfive"]
ARMS = [
    ("none", "identity"),
    ("hidden", "identity_freezehidden"),
    ("cell", "identity_freezecell"),
    ("all", "identity_freezeall"),
]
HS = (1, 6, 12, 18, 24, 36)


def score_row(model, label, h, target="sb"):
    f = RESULTS / f"score_{model}_{label}.csv"
    if not f.exists():
        return None
    with open(f) as fh:
        for r in csv.DictReader(fh):
            if r.get("target") == target and r.get("h") == str(h):
                return r
    return None


def verdict(deltas):
    """Apply the rule to a list of per-seed deltas. Returns (text, mean, sd, n)."""
    d = [x for x in deltas if x is not None and not np.isnan(x)]
    if len(d) < 2:
        return (
            f"insufficient seeds (n={len(d)})",
            (d[0] if d else float("nan")),
            float("nan"),
            len(d),
        )
    a = np.array(d)
    pos, neg = int((a > 0).sum()), int((a < 0).sum())
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    if len(a) == 4 and (pos == 4 or neg == 4) and abs(mean) > sd:
        return f"SUPPORTED ({pos}/4 positive)", mean, sd, len(a)
    if pos == len(a) or neg == len(a):
        return (
            f"CONTESTED — all same sign but |mean| {abs(mean):.4f} <= sd {sd:.4f}",
            mean,
            sd,
            len(a),
        )
    return f"CONTESTED — signs split {pos}+/{neg}-", mean, sd, len(a)


def main() -> int:
    out = [
        "# Wave 1 — cross-seed findings",
        "",
        "Decision rule: 4/4 sign agreement AND |mean| > seed sd. No p-values are claimed — a",
        "paired sign-flip at n=4 floors at 1/16 = 0.0625 and cannot reach 0.05.",
        "",
    ]
    have = [s for s in SEEDS if all((RESULTS / f"score_{s}_{lb}.csv").exists() for _, lb in ARMS)]
    out.append(
        f"Seeds with all four arms complete: **{len(have)}/4** — {', '.join(have) or 'none'}"
    )
    out.append("")

    for metric, better in (
        ("AP", "higher"),
        ("Brier", "lower"),
        ("crps_events", "lower"),
        ("size_ratio", "higher"),
    ):
        out += [
            f"## {metric} ({better} is better) — each freeze arm minus `none`",
            "",
            "| arm | h | "
            + " | ".join(s.replace("fullzero_", "") for s in SEEDS)
            + " | mean | sd | verdict |",
            "|---" * (5 + len(SEEDS)) + "|",
        ]
        for arm, lb in ARMS[1:]:
            for h in (18, 36):
                ds = []
                for s in SEEDS:
                    a, b = score_row(s, lb, h), score_row(s, "identity", h)
                    ds.append(float(a[metric]) - float(b[metric]) if a and b else None)
                txt, mean, sd, _ = verdict(ds)
                cells = [f"{d:+.4f}" if d is not None else "—" for d in ds]
                out.append(
                    f"| {arm} | {h} | "
                    + " | ".join(cells)
                    + f" | {mean:+.4f} | {sd:.4f} | {txt} |"
                )
        out.append("")

    if have:
        origins = load_origins()
        umap = build_unit_grid(str(RAW))
        tm = load_truth(origins, HS)
        out += [
            "## C.4 — onset vs continuation AP, freeze arm minus `none`",
            "",
            "| arm | h | universe | "
            + " | ".join(s.replace("fullzero_", "") for s in have)
            + " | mean | verdict |",
            "|---" * (4 + len(have)) + "|",
        ]
        cache = {}
        for s in have:
            for _a, lb in ARMS:
                cache[(s, lb)] = sub_table(RESULTS / f"bodymean_{s}_{lb}", origins, umap, tm)
        for arm, lb in ARMS[1:]:
            for h in (18, 36):
                for uni in ("cont", "onset"):
                    ds = [
                        cache[(s, lb)][h][f"ap_{uni}"] - cache[(s, "identity")][h][f"ap_{uni}"]
                        for s in have
                    ]
                    txt, mean, _sd, _ = verdict(ds)
                    out.append(
                        f"| {arm} | {h} | {uni} | "
                        + " | ".join(f"{d:+.4f}" for d in ds)
                        + f" | {mean:+.4f} | {txt} |"
                    )
        out.append("")

        out += [
            "## C.2/C.3 — dispersion of predicted change, and direction skill",
            "",
            "| arm | h | measure | "
            + " | ".join(s.replace("fullzero_", "") for s in have)
            + " | mean | verdict |",
            "|---" * (4 + len(have)) + "|",
        ]
        ecache = {}
        for s in have:
            for _a, lb in ARMS:
                ecache[(s, lb)] = esc_rows(RESULTS / f"bodymean_{s}_{lb}", origins, umap, tm)
        for arm, lb in ARMS[1:]:
            for h in (18, 36):
                for key in ("dispersion", "rho"):
                    ds = [ecache[(s, lb)][h][key] - ecache[(s, "identity")][h][key] for s in have]
                    txt, mean, _sd, _ = verdict(ds)
                    out.append(
                        f"| {arm} | {h} | {key} | "
                        + " | ".join(f"{d:+.4f}" for d in ds)
                        + f" | {mean:+.4f} | {txt} |"
                    )
        out.append("")

    p = RESULTS / "FINDINGS.md"
    p.write_text("\n".join(out) + "\n")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
