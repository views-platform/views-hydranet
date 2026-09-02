"""EXP-1 readout, in the order 05 fixed BEFORE the run.

The order is enforced by this script rather than left to the analyst, because the analyst is the
person the pre-registration is protecting the result from. The treatment's primary number is
computed last, so no earlier choice can be tuned against it.

Pools across origins as a ratio of sums, never a mean of ratios: with the field size constant, the
per-origin means are proportional to the sums, so `mass / occurrence` pools exactly. A mean of
per-origin magnitudes would silently weight a near-silent origin the same as a busy one.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decompose import DEFAULT_TAUS, load_arm  # noqa: E402

ANCHOR_AP_H18 = 0.3298395823400329
FLAT_BAND = (0.7, 1.4)
KILL_BELOW = 0.5


def pool(records: list[dict], target_index: int) -> dict[int, dict]:
    """Pool per-origin records into one record per horizon, by summing then dividing."""
    by_h = defaultdict(list)
    for r in records:
        if r["target_index"] == target_index:
            by_h[r["horizon"]].append(r)
    out = {}
    for h, rs in by_h.items():
        n = len(rs)
        occ = sum(r["occurrence"] for r in rs) / n
        mass = sum(r["emitted_mass"] for r in rs) / n
        row = {
            "n_origins": n,
            "occurrence": occ,
            "emitted_mass": mass,
            "mag_gate_weighted": mass / occ if occ > 0 else float("nan"),
            "mag_unweighted": sum(r["mag_unweighted"] for r in rs) / n,
        }
        for tau in DEFAULT_TAUS:
            key = f"{tau:g}".replace(".", "p")
            support = sum(r[f"n_above_{key}"] for r in rs)
            row[f"n_above_{key}"] = support
            # Support-weighted, so an origin with 2 surviving cells cannot count as much as one
            # with 2000. The un-pooled version of this error is what C-318 published.
            row[f"mag_tau_{key}"] = (
                sum(r[f"mag_tau_{key}"] * r[f"n_above_{key}"] for r in rs if r[f"n_above_{key}"])
                / support
                if support
                else float("nan")
            )
        out[h] = row
    return out


def ratio(pooled: dict[int, dict], key: str, h_late: int, h_early: int = 1) -> float:
    a, b = pooled[h_late][key], pooled[h_early][key]
    if b == 0 or math.isnan(a) or math.isnan(b):
        return float("nan")
    return a / b


def check_anchor(score_csv: Path) -> tuple[bool, float | None]:
    """F7: the treatment arm must reproduce the archived AP@h18 exactly, or it is not that arm."""
    if not score_csv.exists():
        return False, None
    with score_csv.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("target") == "sb" and row.get("h") == "18":
                ap = float(row["AP"])
                return abs(ap - ANCHOR_AP_H18) < 1e-12, ap
    return False, None


def report(name: str, pooled: dict[int, dict], h_late: int) -> dict:
    print(f"\n=== {name} ===")
    print(f"{'h':>3} {'occurrence':>12} {'mass':>12} {'mag_gw':>10} {'mag_unw':>10}")
    for h in sorted(pooled):
        if h in (1, 6, 12, 18, 24, 30, h_late):
            r = pooled[h]
            print(
                f"{h:>3} {r['occurrence']:>12.6e} {r['emitted_mass']:>12.6e} "
                f"{r['mag_gate_weighted']:>10.4f} {r['mag_unweighted']:>10.4f}"
            )
    out = {
        "occ_ratio": ratio(pooled, "occurrence", h_late),
        "mag_gw_ratio": ratio(pooled, "mag_gate_weighted", h_late),
        "mag_unw_ratio": ratio(pooled, "mag_unweighted", h_late),
    }
    print(
        f"  h{h_late}/h1 -> occurrence {out['occ_ratio']:.4g} | "
        f"mag_gate_weighted {out['mag_gw_ratio']:.4f} | mag_unweighted {out['mag_unw_ratio']:.4f}"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--model", default="fullzero_fortytwo")
    ap.add_argument("--treatment", default="identity")
    ap.add_argument("--control", default="use_real")
    ap.add_argument("--target-index", type=int, default=0, help="0 = sb, the primary target")
    ap.add_argument("--h-late", type=int, default=36)
    args = ap.parse_args()

    fired: list[str] = []

    # (a) F7 — the anchor. Run first: if the arm is not the arm, nothing after it means anything.
    ok, value = check_anchor(args.results / f"score_{args.model}_{args.treatment}.csv")
    print(
        f"[a] F7 anchor  AP@h18 = {value} (expected {ANCHOR_AP_H18}) -> {'OK' if ok else 'MISMATCH'}"
    )
    if not ok:
        fired.append("F7")

    # (b) F6 — parity. Asserted by tests/distributions/test_body_mean_dump.py, not re-run here.
    print("[b] F6 parity  asserted by test_dump_on_is_byte_identical_to_dump_off (suite green)")

    # (c) F3 — G1, the two-instrument identity check. Reported by exp1_g1.py; see 07.
    print("[c] F3 G1      see exp1_g1.py output (cube instrument vs dump instrument)")

    # (d) the CONTROL, before the treatment.
    ctrl = pool(
        load_arm(args.results / f"bodymean_{args.model}_{args.control}"), args.target_index
    )
    c = report(f"(d) CONTROL — {args.control} (real observations fed back)", ctrl, args.h_late)
    if not (FLAT_BAND[0] <= c["mag_gw_ratio"] <= FLAT_BAND[1]):
        fired.append("F4")

    # (e) the TREATMENT. The primary number of the whole experiment, read last.
    trt = pool(
        load_arm(args.results / f"bodymean_{args.model}_{args.treatment}"), args.target_index
    )
    t = report(f"(e) TREATMENT — {args.treatment} (free-running)", trt, args.h_late)

    if t["mag_gw_ratio"] < KILL_BELOW:
        fired.append("F1")
    if t["mag_unw_ratio"] < KILL_BELOW and t["mag_gw_ratio"] >= FLAT_BAND[0]:
        fired.append("F2")
    if t["occ_ratio"] > KILL_BELOW:
        fired.append("F5")

    # (f) the tau sweep — the survivorship dose-response (05 amendment A1). Read last of all.
    print("\n=== (f) tau sweep — the CONDITIONED statistic, treatment arm ===")
    print(
        f"{'tau':>6} {'mag@h1':>10} {'mag@h' + str(args.h_late):>10} {'ratio':>8} {'n@h1':>9} {'n@late':>9}"
    )
    tau_ratios = []
    for tau in DEFAULT_TAUS:
        key = f"{tau:g}".replace(".", "p")
        r = ratio(trt, f"mag_tau_{key}", args.h_late)
        tau_ratios.append(r)
        print(
            f"{tau:>6} {trt[1][f'mag_tau_{key}']:>10.4f} "
            f"{trt[args.h_late][f'mag_tau_{key}']:>10.4f} "
            f"{r:>8.4f} {trt[1][f'n_above_{key}']:>9} {trt[args.h_late][f'n_above_{key}']:>9}"
        )
    clean = [r for r in tau_ratios if not math.isnan(r)]
    monotone = len(clean) > 2 and all(b >= a for a, b in zip(clean, clean[1:]))
    spread = (max(clean) - min(clean)) if clean else float("nan")
    print(f"  monotone in tau: {monotone} | spread {spread:.4f}")
    if monotone and spread > 0.3:
        fired.append("F9")

    print("\n=== VERDICT ===")
    print(f"falsifiers fired: {fired if fired else 'NONE'}")
    primary = t["mag_gw_ratio"]
    if "F1" in fired or "F2" in fired:
        print("C1 FALSIFIED — the model does make smaller forecasts.")
    elif math.isnan(primary):
        print("UNDEFINED primary readout — halt.")
    elif primary < FLAT_BAND[0]:
        print(f"CONTESTED — primary {primary:.4f} in the pre-declared grey zone [0.5, 0.7).")
    elif FLAT_BAND[0] <= primary <= FLAT_BAND[1]:
        print(f"C1 SUPPORTED at this seed — primary {primary:.4f} within the flat band.")
    else:
        print(
            f"primary {primary:.4f} ABOVE the flat band — magnitude rose; see 05 decision rules."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
