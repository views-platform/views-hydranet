"""lesson_curve_gate.py — the pre-registered decision rule for the lesson curve.

Companion to `floor_gate.py` and shaped the same way: the *rule* lives here, tracked and
tested, while the CSV reading and markdown rendering live in the dossier's
`tools/verify_curve.py`. That split is not tidiness — a tracked test may not runtime-load the
gitignored `reports/` tree, so a rule that lives only in a dossier is a rule with no test in
CI, which is how a decision rule quietly stops meaning anything.

The question
------------
Does training past 160 lessons keep buying rollout **retention** — `R = AP(h18)/AP(h1)`,
free-running? Pre-registration: `reports/2026-08-18_lesson_curve_dossier/05_analysis_plan.md`
(LOCKED 2026-08-18).

The rule
--------
Seed-to-seed variance, not origin-block variance, is what applies when comparing two *training
runs*. So the anchor (L=160) is run at four seeds and a new run's plausible range is the
one-sided 95% **prediction bound**::

    k        = t(3, 0.95) * sqrt(1 + 1/4) = 2.631
    bound(X) = mean(X over the anchor seeds) + k * sd(X over the anchor seeds)

applied to **both** endpoints — absolute `F = AP(h18)` (primary) and `R` (co-primary) — because
a ratio can move on its denominator, and the two must agree.

Four states, because "no effect" and "could not tell" are different answers and the 2026-08-14
sweep could not tell them apart:

* **RISING** — the longest arm clears both bounds *and* the measurement floor (`3 x MDE_F`).
* **PLATEAU** — inside both bounds *and* the bound is narrower than the pre-registered effect
  `theta`.
  A null is only declarable when the interval excludes the effect.
* **UNDERPOWERED** — inside the bounds but the bound is not narrower than `theta`.
* **G1-STOP** — `k * sigma_seed(R) >= 0.30`, i.e. training-run noise swamps most of the 0.4687
  gap to
  the ceiling. No single-seed lesson point can resolve anything; that is itself the finding.
* **VOID** — a falsifier fired, so the numbers are not a result.

`theta = 0.14` is `0.30 * (R_oracle(160) - R(160)) = 0.30 * (1.010134 - 0.541482)`, both
measured. The floor-limited 40-lesson retention sets no threshold here.
"""

from __future__ import annotations

import hashlib
import json
import math

__all__ = [
    "ANCHOR_L",
    "G1_STOP",
    "H_BASE",
    "H_STAR",
    "K_PRED",
    "REF_N",
    "THETA",
    "curve_verdict",
    "rule_md5",
]

H_BASE = 1
H_STAR = 18
REF_N = 170430
ANCHOR_L = 160
THETA = 0.14  # 0.30 * (1.010134 - 0.541482), both measured; see the module docstring
K_PRED = 2.631  # t(3, 0.95) * sqrt(1 + 1/4) — one-sided 95% prediction bound for a NEW run
G1_STOP = 0.30  # k * sigma_seed(R) at/above this: nothing learnable at one seed per point

_STATES = ("RISING", "PLATEAU", "UNDERPOWERED", "G1-STOP", "VOID")


def rule_md5(
    *,
    theta: float = THETA,
    k_pred: float = K_PRED,
    g1_stop: float = G1_STOP,
    anchor_l: int = ANCHOR_L,
    h_star: int = H_STAR,
) -> str:
    """Hash of every constant the verdict depends on — the pre-registration ↔ driver handshake.

    Same device as `floor_gate.threshold_md5`: if someone relaxes a threshold after seeing a
    control, the hash moves and the licence is void.
    """
    return hashlib.md5(
        json.dumps(
            {"THETA": theta, "K": k_pred, "G1": g1_stop, "ANCHOR_L": anchor_l, "H_STAR": h_star},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _falsifiers(arms: list[dict], *, ref_n: int) -> list[str]:
    """F1-F6. Each means the numbers are not a result, so these run before the rule."""
    problems: list[str] = []

    for a in arms:
        label = a["label"]

        # F2 — arms must sit on one support or their APs are not comparable
        if int(a.get("n_cells", ref_n)) != ref_n:
            problems.append(f"F2: {label} scored {a['n_cells']} cells != {ref_n}")

        # F1 — step 1 has no feedback, so the oracle cannot differ from the control there
        o1 = a.get("oracle_h1")
        if o1 is not None and abs(float(a["ap_h1"]) - float(o1)) > 1e-6:
            problems.append(
                f"F1: {label} h{H_BASE} control {float(a['ap_h1']):.8f} != oracle "
                f"{float(o1):.8f} — step 1 has no feedback, so something other than the "
                "feedback path moved"
            )

        # the lesson curve is the eps=0 axis; an SS arm here is a different experiment
        if float(a.get("ss_epsilon_max", 0.0)) != 0.0:
            problems.append(
                f"{label}: ss_epsilon_max={a['ss_epsilon_max']} — scheduled sampling is a "
                "different intervention and must not be pooled into the lesson curve"
            )

    # F3 — two arms with one weight hash means one model was scored twice
    seen: dict[str, str] = {}
    for a in arms:
        h = a.get("weight_sha256")
        if not h:
            continue
        if h in seen:
            problems.append(
                f"F3: {seen[h]} and {a['label']} share a weight hash — the same model scored twice"
            )
        seen[h] = a["label"]

    # F6 — arms built at different repo HEADs are not comparable
    heads = {a["head"] for a in arms if a.get("head")}
    if len(heads) > 1:
        problems.append(f"F6: arms span {len(heads)} repo HEADs — not comparable")

    # F4 — a control that cannot show an effect is not evidence (C-299)
    try:
        from scripts.floor_gate import floor_gate
    except Exception as exc:  # noqa: BLE001
        problems.append(f"F4: the floor gate could not be imported: {exc!r}")
    else:
        for a in arms:
            try:
                g = floor_gate(
                    ap_control=float(a["ap_h18"]),
                    n_cells=int(a["n_cells"]),
                    n_event=int(a["n_event"]),
                    horizon=H_STAR,
                    target="sb",
                )
            except Exception as exc:  # noqa: BLE001
                problems.append(f"F4: floor gate raised on {a['label']}: {exc!r}")
                continue
            a["fg_a"] = g["clauses"]["FG-A"]["verdict"]
            if a["fg_a"] != "PASS":
                problems.append(
                    f"F4: {a['label']} fails FG-A — the control cannot show an effect at h{H_STAR}"
                )
    return problems


def _sd(xs: list[float]) -> float:
    n = len(xs)
    mean = sum(xs) / n
    return (sum((x - mean) ** 2 for x in xs) / (n - 1)) ** 0.5


def curve_verdict(
    arms: list[dict],
    *,
    theta: float = THETA,
    k_pred: float = K_PRED,
    g1_stop: float = G1_STOP,
    anchor_l: int = ANCHOR_L,
    ref_n: int = REF_N,
) -> dict:
    """Evaluate the pre-registered rule over parsed arm records.

    Each record needs ``label, total_lessons, torch_seed, ap_h1, ap_h18, n_cells, n_event`` and may
    carry ``oracle_h1, oracle_h18, mde_f, head, weight_sha256, ss_epsilon_max``.

    Returns ``{state, detail, problems, sigma_seed_r, sigma_seed_f, bound_r, bound_f, mean_r,
    mean_f, mde_f, anchor, longest, decomposition}``. Never raises on data — a bad input becomes
    a VOID with the reason attached, because a crash would replace a partial result with none.
    """
    out: dict = {
        "state": "VOID",
        "detail": "",
        "problems": [],
        "sigma_seed_r": None,
        "sigma_seed_f": None,
        "bound_r": None,
        "bound_f": None,
        "mean_r": None,
        "mean_f": None,
        "mde_f": None,
        "anchor": [],
        "longest": None,
        "decomposition": [],
        "rule_md5": rule_md5(theta=theta, k_pred=k_pred, g1_stop=g1_stop, anchor_l=anchor_l),
    }
    if not arms:
        out["detail"] = "no arm has both a config record and a score"
        return out

    for a in arms:
        c, f = float(a["ap_h1"]), float(a["ap_h18"])
        a["C"], a["F"] = c, f
        a["R"] = f / c if c > 0 else float("nan")
        a["O"] = float(a["oracle_h18"]) if a.get("oracle_h18") is not None else None

    problems = _falsifiers(arms, ref_n=ref_n)
    out["problems"] = problems

    anchor = sorted(
        (a for a in arms if int(a["total_lessons"]) == anchor_l), key=lambda a: a["torch_seed"]
    )
    longer = sorted(
        (a for a in arms if int(a["total_lessons"]) > anchor_l), key=lambda a: a["total_lessons"]
    )
    out["anchor"] = [a["label"] for a in anchor]

    # the decomposition is reportable even when the verdict is not
    base = anchor[0] if anchor else None
    if base and base["C"] > 0 and base["F"] > 0:
        for m in longer:
            if m["C"] > 0 and m["F"] > 0:
                out["decomposition"].append(
                    {
                        "total_lessons": int(m["total_lessons"]),
                        "dlog_F": math.log(m["F"] / base["F"]),
                        "dlog_C": math.log(m["C"] / base["C"]),
                        "dlog_R": math.log(m["R"] / base["R"]),
                        "oracle_h18": m["O"],
                    }
                )

    if problems:
        out["detail"] = "harness invariants failed — the numbers are not a result"
        return out

    if len(anchor) < 3:
        out["state"] = "UNDERPOWERED"
        out["detail"] = (
            f"{len(anchor)} arm(s) at L={anchor_l}; sigma_seed needs at least 3 and the "
            "pre-registered bound assumes 4"
        )
        return out

    rs = [a["R"] for a in anchor]
    fs = [a["F"] for a in anchor]
    mean_r, mean_f = sum(rs) / len(rs), sum(fs) / len(fs)
    sigma_r, sigma_f = _sd(rs), _sd(fs)
    mdes = [float(a["mde_f"]) for a in anchor if a.get("mde_f")]
    out.update(
        sigma_seed_r=sigma_r,
        sigma_seed_f=sigma_f,
        mean_r=mean_r,
        mean_f=mean_f,
        bound_r=mean_r + k_pred * sigma_r,
        bound_f=mean_f + k_pred * sigma_f,
        mde_f=sum(mdes) / len(mdes) if mdes else None,
    )

    if k_pred * sigma_r >= g1_stop:
        out["state"] = "G1-STOP"
        out["detail"] = (
            f"k x sigma_seed(R) = {k_pred * sigma_r:.4f} >= {g1_stop}. Training-run variance "
            "on this vehicle swamps most of the 0.4687 gap to the ceiling, so no single-seed "
            "lesson point can resolve anything. This IS the result — and it re-scopes every "
            "single-seed claim made on this vehicle."
        )
        return out

    if not longer:
        out["state"] = "UNDERPOWERED"
        out["detail"] = f"sigma_seed measured but no arm above L={anchor_l} has been scored yet"
        return out

    top = longer[-1]
    out["longest"] = top["label"]
    r_ok = top["R"] > out["bound_r"]
    f_ok = top["F"] > out["bound_f"]
    mde_ok = out["mde_f"] is not None and (top["F"] - base["F"]) > 3 * out["mde_f"]

    if r_ok and f_ok and mde_ok:
        out["state"] = "RISING"
        out["detail"] = (
            f"L={top['total_lessons']} clears the prediction bound on both endpoints and the "
            f"measurement floor. Training continues to buy retention past {anchor_l}."
        )
    elif not (r_ok or f_ok) and k_pred * sigma_r < theta:
        out["state"] = "PLATEAU"
        out["detail"] = (
            f"L={top['total_lessons']} sits inside the prediction bound on both endpoints, and "
            f"the bound ({k_pred * sigma_r:.4f}) is narrower than theta ({theta}) — so this is a "
            "null, not a shrug. Training is done as a lever at this configuration."
        )
    else:
        out["state"] = "UNDERPOWERED"
        out["detail"] = (
            f"the prediction bound ({k_pred * sigma_r:.4f}) is not narrower than theta "
            f"({theta}), or the endpoints disagree. 'No effect' and 'could not tell' are not "
            "distinguishable here."
        )
    return out
