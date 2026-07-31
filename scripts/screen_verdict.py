#!/usr/bin/env python
"""screen_verdict.py — apply the LOCKED T=0 body-calibration screen thresholds to an audit JSON.

Mechanically turns `audit_run.py` output into the pre-registered PASS/KILL verdict of the
2026-07-02 T=0 Body-Calibration Screen (dossier `05_analysis_plan.md` §4, LOCK 3). No human
judgment: reads step "0" of lr_sb_best + lr_ns_best (os is reported, NOT gated) and fires the
locked criteria. The thresholds below are CONSTANTS anchored to the measured 300L floor — they do
not move (LOCK 3); changing them is a lock violation.

Usage:
  screen_verdict.py <audit.json>                     # P1 single-seed screen verdict
  screen_verdict.py --confirm <j1.json,j2,j3>        # P2: S-criterion must hold on all 3 seeds
"""

from __future__ import annotations

import argparse
import json
import math

# LOCKED §4 thresholds (dossier 2026-07-02 05_analysis_plan.md), gating channels sb + ns only.
CH = ("lr_sb_best", "lr_ns_best")
S1_MAX = {"lr_sb_best": 67.0, "lr_ns_best": 72.0}  # %mass_zeros ≤  (floor − 10 pts)
S2_SPEARMAN_MIN = {"lr_sb_best": 0.46, "lr_ns_best": 0.33}  # spearman ≥ (floor + 0.15)
S2_RANGECOMP_MAX = {"lr_sb_best": 4.2, "lr_ns_best": 3.1}  # range_comp ≤ (0.6 × floor)
GUARD_CRPS_MAX = {"lr_sb_best": 0.60, "lr_ns_best": 0.28}  # CRPS ≤ (1.2 × floor)
GUARD_AP_MIN = {"lr_sb_best": 0.35, "lr_ns_best": 0.30}  # gate AP ≥ (0.8 × floor)
K2_RATIO_MED_MIN = 0.10  # ratio_med < this on BOTH channels ⇒ timid prophet
K1_PREDMAX_CAP = 1e7  # any-step pred_max > this ⇒ unstable
_DECISION_FIELDS = (
    "frac_zero_mass",
    "spearman_pos",
    "range_compression",
    "ratio_med",
    "crps",
    "gate_ap",
    "pred_max",
)


def _step0(blob: dict, ch: str) -> dict:
    return blob[ch]["0"]


def _k1(blob: dict) -> list[str]:
    """K1 unstable AT T=0: non-finite decision field, or pred_max > 1e7, at STEP 0 (sb/ns).

    This is a T=0 calibration screen — the verdict is judged at step 0 ONLY. A ROLLOUT-step
    explosion (the C-113 autoregressive bloom, which even the floor exhibits mildly) is a SEPARATE
    axis and must NOT gate the T=0 verdict; it is captured as `rollout_pred_max` info in the metric
    snapshot, not fired as a kill. K1 here catches a body that has genuinely diverged AT T=0 (e.g.
    count_mean-style ~1e16 at step 0).
    """
    fired = []
    for ch in CH:
        r0 = _step0(blob, ch)
        for f in _DECISION_FIELDS:
            v = r0.get(f)
            if v is None or not math.isfinite(v):
                fired.append(f"K1:{ch}:{f}=non-finite@T0")
        if (r0.get("pred_max", 0.0) or 0.0) > K1_PREDMAX_CAP:
            fired.append(f"K1:{ch}:pred_max={r0['pred_max']:.3g}>1e7@T0")
    return fired


def _s1(blob: dict) -> dict:
    """S1 zero-smear win on a channel: %mass_zeros ≤ threshold."""
    return {ch: (100.0 * _step0(blob, ch)["frac_zero_mass"]) <= S1_MAX[ch] for ch in CH}


def _s2(blob: dict) -> dict:
    """S2 dynamic-range win on a channel: spearman ≥ min AND range_comp ≤ max."""
    out = {}
    for ch in CH:
        r0 = _step0(blob, ch)
        out[ch] = (
            r0["spearman_pos"] >= S2_SPEARMAN_MIN[ch]
            and r0["range_compression"] <= S2_RANGECOMP_MAX[ch]
        )
    return out


def _guardrails(blob: dict) -> list[str]:
    """Guardrails that must hold for ANY S to count. Returns list of BROKEN guardrails."""
    broken = []
    for ch in CH:
        r0 = _step0(blob, ch)
        if r0["crps"] > GUARD_CRPS_MAX[ch]:
            broken.append(f"CRPS:{ch}={r0['crps']:.3f}>{GUARD_CRPS_MAX[ch]}")
        if (
            r0.get("gate_ap") is None
            or not math.isfinite(r0["gate_ap"])
            or r0["gate_ap"] < GUARD_AP_MIN[ch]
        ):
            broken.append(f"gate_AP:{ch}={r0.get('gate_ap')}<{GUARD_AP_MIN[ch]}")
    return broken


def verdict(blob: dict) -> dict:
    """Return the full P1 verdict dict for one audit JSON."""
    k1 = _k1(blob)
    if k1:  # instability short-circuits everything
        return {
            "verdict": "KILL",
            "kill": ["K1"] + k1,
            "survive_via": None,
            "s1": {},
            "s2": {},
            "guardrails_broken": [],
            "metrics": _metric_snapshot(blob),
        }
    ratio_med = {ch: _step0(blob, ch)["ratio_med"] for ch in CH}
    k2 = all(ratio_med[ch] < K2_RATIO_MED_MIN for ch in CH)
    s1, s2 = _s1(blob), _s2(blob)
    s1_both, s2_both = all(s1.values()), all(s2.values())
    broken = _guardrails(blob)
    survive_via = None
    if not k2 and not broken:
        if s1_both:
            survive_via = "S1"
        elif s2_both:
            survive_via = "S2"
    kill = []
    if k2:
        kill.append(
            f"K2:ratio_med sb={ratio_med['lr_sb_best']:.2f} ns={ratio_med['lr_ns_best']:.2f}"
        )
    if not s1_both and not s2_both:
        kill.append("K3:no S-criterion on both channels")
    if (s1_both or s2_both) and broken:
        kill.append("F2:S-criterion met but guardrail broken: " + ", ".join(broken))
    v = "PASS" if survive_via else "KILL"
    return {
        "verdict": v,
        "survive_via": survive_via,
        "kill": kill,
        "s1": s1,
        "s2": s2,
        "guardrails_broken": broken,
        "metrics": _metric_snapshot(blob),
    }


def _metric_snapshot(blob: dict) -> dict:
    out = {}
    for ch in CH:
        r0 = _step0(blob, ch)
        out[ch] = {
            "mass_zeros_pct": round(100.0 * r0["frac_zero_mass"], 1),
            "spearman": round(r0["spearman_pos"], 3),
            "range_comp": round(r0["range_compression"], 2),
            "ratio_med": round(r0["ratio_med"], 2),
            "crps": round(r0["crps"], 3),
            "gate_ap": round(r0["gate_ap"], 3)
            if r0.get("gate_ap") is not None and math.isfinite(r0["gate_ap"])
            else None,
            "pred_max": round(r0["pred_max"], 1),
            # rollout magnitude — INFO ONLY (separate C-113 axis), never gates the T=0
            # verdict
            "rollout_pred_max": round(
                max((s.get("pred_max", 0) or 0) for s in blob[ch].values()), 1
            ),
        }
    return out


def md_row(arm: str, v: dict) -> str:
    m = v["metrics"]
    detail = f"via {v['survive_via']}" if v["verdict"] == "PASS" else "; ".join(v["kill"]) or "—"
    return (
        f"| {arm} | **{v['verdict']}** | {detail} | "
        f"sb %mz={m['lr_sb_best']['mass_zeros_pct']} sp={m['lr_sb_best']['spearman']} "
        f"rc={m['lr_sb_best']['range_comp']} CRPS={m['lr_sb_best']['crps']} | "
        f"ns %mz={m['lr_ns_best']['mass_zeros_pct']} sp={m['lr_ns_best']['spearman']} "
        f"rc={m['lr_ns_best']['range_comp']} CRPS={m['lr_ns_best']['crps']} |"
    )


def confirm(paths: list[str]) -> dict:
    """P2: a survivor is CONFIRMED only if its P1 survive-criterion holds on all seeds."""
    blobs = [json.load(open(p)) for p in paths]
    v0 = verdict(blobs[0])
    crit = v0["survive_via"]
    if crit is None:
        return {
            "confirmed": False,
            "reason": "seed-1 is not a survivor",
            "per_seed": [v0["verdict"]],
        }
    checks = []
    for b in blobs:
        s = _s1(b) if crit == "S1" else _s2(b)
        checks.append(all(s.values()) and not _guardrails(b))
    return {
        "confirmed": all(checks),
        "criterion": crit,
        "per_seed_holds": checks,
        "reason": None if all(checks) else "F4: criterion fails on ≥1 confirm seed",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audit", nargs="?", help="single audit .json (P1)")
    ap.add_argument("--confirm", help="comma-separated 3 audit .json (P2)")
    ap.add_argument("--arm", default="arm")
    args = ap.parse_args()
    if args.confirm:
        res = confirm(args.confirm.split(","))
        print(json.dumps(res, indent=2))
    else:
        v = verdict(json.load(open(args.audit)))
        print(json.dumps(v, indent=2))
        print("\nMD_ROW " + md_row(args.arm, v))


if __name__ == "__main__":
    main()
