"""ss_sweep_gate.py — the pre-registered decision rule for the scheduled-sampling sweep.

Companion to `lesson_curve_gate.py` and `floor_gate.py`, and shaped the same way: the *rule* lives
here, tracked and unit-tested, while reading CSVs and rendering markdown lives in the dossier's
`tools/verify_sweep.py`. A tracked test may not runtime-load the gitignored `reports/` tree, so a
rule that lived only in a dossier would be a rule with no test in CI — which is how a decision rule
fed by ~30 GPU-hours quietly stops meaning anything.

The question
------------
Does correctly-configured scheduled sampling (`ss_feedback='sample'`) change free-running rollout
retention, on a vehicle that has measurable retention?
Pre-registration: `reports/2026-08-17_ss_retention_dossier/05_analysis_plan.md` (LOCKED 2026-08-17,
AMENDMENT 1 2026-08-21 re-scoping it from L=160 to L=300).

The rule
--------
* **Primary** — `AP_sb(h=18)`, free-running. Direction is pre-registered: **SS lowers it**.
  One-sided.
* **Co-primary** — retention `AP(h18)/AP(h1)`. Must **agree in sign** with the primary, because a
  ratio can move on its denominator.
* **The guard** — `|mean ΔAP(h1)| <= 3 * MDE_AP(h1)`. If SS moved the *anchor*, then "retention" is
  the wrong frame for whatever happened and the result is a **traded failure**, not a retention
  result. This clause was in the pre-registration from the start and was missing from the
  implementation until 2026-08-21; a run where SS wrecked one-step skill would have been
  reported as a retention effect.
* **The test** — exact one-sided permutation on the **seed-level** values. With 4 per side the
  smallest reachable p is 1/C(8,4) = 0.014; with 3 per side it is 1/C(6,3) = 0.05, i.e. exactly at
  alpha, so 3 is the minimum that can reach significance at all.

Four states, because "no effect" and "could not tell" are different answers:

* **EFFECT** — p <= alpha, the drop clears 3 x MDE, and both endpoints agree in sign.
* **NULL** — p > alpha *and* the interval excludes a theta-sized effect. Only then is a null real.
* **UNDERPOWERED** — p > alpha but the interval still admits theta.
* **VOID** — a falsifier fired, or the post-hoc floor gate on the sweep's own controls failed.

**Censoring clause:** a treated arm below `2 x prevalence(h*)` is at the floor, so the effect
*magnitude* is censored — report it as ">= X", never as a point estimate.
"""

from __future__ import annotations

import hashlib
import json
from itertools import combinations

__all__ = [
    "ALPHA",
    "GUARD_K",
    "H_BASE",
    "H_STAR",
    "MDE_K",
    "MIN_PER_SIDE",
    "REF_N",
    "THETA",
    "rule_md5",
    "sweep_verdict",
]

H_BASE = 1
H_STAR = 18
REF_N = 170430
THETA = 0.30  # the effect size a NULL must exclude, as a fraction of the control mean
ALPHA = 0.05
MDE_K = 3.0  # an EFFECT's drop must clear this many MDEs
GUARD_K = 3.0  # the anchor guard: |mean dAP(h1)| must stay within this many MDE_AP(h1)
MIN_PER_SIDE = 3  # below this the exact test cannot reach alpha at all


def rule_md5(
    *,
    theta: float = THETA,
    alpha: float = ALPHA,
    mde_k: float = MDE_K,
    guard_k: float = GUARD_K,
    h_star: int = H_STAR,
) -> str:
    """Hash of every constant the verdict depends on — the pre-registration <-> driver handshake.

    Same device as `floor_gate.threshold_md5`: relax a threshold after seeing the arms and the hash
    moves, so the licence is visibly void rather than silently changed.
    """
    return hashlib.md5(
        json.dumps(
            {"THETA": theta, "ALPHA": alpha, "MDE_K": mde_k, "GUARD_K": guard_k, "H_STAR": h_star},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def perm_p_one_sided(treated: list[float], control: list[float]) -> float:
    """Exact one-sided permutation p for 'treated is LOWER than control', on seed-level values.

    Enumerates every split of the pooled values, so there is no sampling error in the p itself.
    """
    k, pool = len(treated), treated + control
    n = len(pool)
    obs = sum(treated) / k - sum(control) / (n - k)
    hits = tot = 0
    for idx in combinations(range(n), k):
        t = [pool[i] for i in idx]
        rest = [pool[i] for i in range(n) if i not in idx]
        if sum(t) / k - sum(rest) / (n - k) <= obs:
            hits += 1
        tot += 1
    return hits / tot


def _falsifiers(arms: list[dict], *, ref_n: int) -> list[str]:
    """F3-F6. F1/F2 are asserted pre-GPU by make_ss_arm.py and cannot be re-checked from scores."""
    problems: list[str] = []

    for a in arms:
        if int(a.get("n_cells", ref_n)) != ref_n:
            problems.append(f"F3: {a['label']} scored {a['n_cells']} cells != {ref_n}")
        n_o = a.get("n_origins")
        if n_o is not None and int(n_o) != 13:
            problems.append(f"F3: {a['label']} has {n_o} origins != 13")

    # F5 — two arms with one weight hash means the same model was evaluated twice
    seen: dict[str, str] = {}
    for a in arms:
        h = a.get("weight_sha256")
        if not h:
            continue
        if h in seen:
            problems.append(f"F5: {seen[h]} and {a['label']} share a weight hash")
        seen[h] = a["label"]

    # F4 — an identical AP at any horizon means one cube was scored twice
    for (la, aa), (lb, ab) in combinations([(a["label"], a) for a in arms], 2):
        for h in (H_BASE, H_STAR):
            va, vb = aa.get(f"ap_h{h}"), ab.get(f"ap_h{h}")
            if va is not None and vb is not None and abs(float(va) - float(vb)) < 1e-12:
                problems.append(f"F4: {la} and {lb} share an identical AP at h{h}")

    # F6 — arms built from different result-producing code are not comparable. Prefer a code
    # fingerprint over a commit id: docs-only commits move the id and not the code (lesson-curve
    # AMENDMENT 1). Unknown provenance is flagged, never silently exempted.
    key = "code_fingerprint" if any(a.get("code_fingerprint") for a in arms) else "head"
    missing = [a["label"] for a in arms if not a.get(key)]
    if missing:
        problems.append(
            f"F6: {len(missing)} arm(s) have no {key} — provenance unknown, comparability cannot "
            f"be asserted: {', '.join(sorted(missing))}"
        )
    versions = {a[key] for a in arms if a.get(key)}
    if len(versions) > 1:
        problems.append(f"F6: arms span {len(versions)} result-producing code versions")

    # every arm must be the same length, or this is a lesson-curve contrast wearing an SS hat
    lengths = {int(a["total_lessons"]) for a in arms}
    if len(lengths) > 1:
        problems.append(
            f"arms span {len(lengths)} lesson counts {sorted(lengths)} — the SS contrast requires "
            "one training length; a mixed set confounds SS with training length"
        )
    return problems


def _floor_gate_controls(controls: list[dict]) -> list[str]:
    """Post-hoc floor gate on the sweep's OWN controls. Failure voids the sweep (§5)."""
    problems: list[str] = []
    try:
        from scripts.floor_gate import floor_gate
    except Exception as exc:  # noqa: BLE001
        return [f"the post-hoc floor gate could not be imported: {exc!r}"]
    for a in controls:
        try:
            g = floor_gate(
                ap_control=float(a["ap_h18"]),
                n_cells=int(a["n_cells"]),
                n_event=int(a["n_event"]),
                horizon=H_STAR,
                target="sb",
            )
        except Exception as exc:  # noqa: BLE001
            problems.append(f"post-hoc floor gate raised on {a['label']}: {exc!r}")
            continue
        a["fg_a"] = g["clauses"]["FG-A"]["verdict"]
        if a["fg_a"] != "PASS":
            problems.append(f"post-hoc floor gate: control {a['label']} fails FG-A")
    return problems


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def sweep_verdict(
    arms: list[dict],
    *,
    theta: float = THETA,
    alpha: float = ALPHA,
    mde_k: float = MDE_K,
    guard_k: float = GUARD_K,
    min_per_side: int = MIN_PER_SIDE,
    ref_n: int = REF_N,
) -> dict:
    """Evaluate the pre-registered rule over parsed arm records.

    Each record needs ``label, total_lessons, torch_seed, ss_epsilon_max, ap_h1, ap_h18, n_cells,
    n_event`` and may carry ``mde_h1, mde_h18, n_origins, head, code_fingerprint, weight_sha256``.

    Never raises on data — a bad input becomes a VOID with the reason attached, because a crash
    would replace a partial result with none.
    """
    out: dict = {
        "state": "VOID",
        "detail": "",
        "problems": [],
        "n_control": 0,
        "n_treated": 0,
        "mean_control_h18": None,
        "mean_treated_h18": None,
        "diff_h18": None,
        "diff_retention": None,
        "endpoints_agree": None,
        "p_value": None,
        "mde_h18": None,
        "mde_h1": None,
        "guard_delta_h1": None,
        "guard_ok": None,
        "censored": [],
        "rule_md5": rule_md5(
            theta=theta, alpha=alpha, mde_k=mde_k, guard_k=guard_k, h_star=H_STAR
        ),
    }
    if not arms:
        out["detail"] = "no arm has both a config record and a score"
        return out

    for a in arms:
        c, f = float(a["ap_h1"]), float(a["ap_h18"])
        a["R"] = f / c if c > 0 else float("nan")

    problems = _falsifiers(arms, ref_n=ref_n)

    control = sorted(
        (a for a in arms if float(a["ss_epsilon_max"]) == 0.0), key=lambda a: a["torch_seed"]
    )
    treated = sorted(
        (a for a in arms if float(a["ss_epsilon_max"]) > 0.0), key=lambda a: a["torch_seed"]
    )
    out["n_control"], out["n_treated"] = len(control), len(treated)

    doses = {float(a["ss_epsilon_max"]) for a in treated}
    if len(doses) > 1:
        problems.append(
            f"treated arms span {len(doses)} doses {sorted(doses)} — the pre-registered "
            "contrast is a single dose; pooling doses would test a different hypothesis"
        )

    problems += _floor_gate_controls(control)
    out["problems"] = problems
    if problems:
        out["detail"] = "harness invariants failed — the numbers are not a result"
        return out

    if len(control) < min_per_side or len(treated) < min_per_side:
        out["state"] = "UNDERPOWERED"
        out["detail"] = (
            f"{len(control)} control and {len(treated)} treated arm(s); the exact test cannot "
            f"reach alpha={alpha} with fewer than {min_per_side} per side"
        )
        return out

    c18 = [float(a["ap_h18"]) for a in control]
    t18 = [float(a["ap_h18"]) for a in treated]
    cR = [a["R"] for a in control]
    tR = [a["R"] for a in treated]
    c1 = [float(a["ap_h1"]) for a in control]
    t1 = [float(a["ap_h1"]) for a in treated]

    mde18 = [float(a["mde_h18"]) for a in arms if a.get("mde_h18") is not None]
    mde1 = [float(a["mde_h1"]) for a in arms if a.get("mde_h1") is not None]
    out.update(
        mean_control_h18=_mean(c18),
        mean_treated_h18=_mean(t18),
        diff_h18=_mean(t18) - _mean(c18),
        diff_retention=_mean(tR) - _mean(cR),
        mde_h18=_mean(mde18) if mde18 else None,
        mde_h1=_mean(mde1) if mde1 else None,
        p_value=perm_p_one_sided(t18, c18),
    )
    # both endpoints must move the same way; a ratio can move on its denominator alone
    out["endpoints_agree"] = (out["diff_h18"] < 0) == (out["diff_retention"] < 0)

    # ---- the guard: did SS move the ANCHOR rather than the retention? --------------------------
    out["guard_delta_h1"] = _mean(t1) - _mean(c1)
    if out["mde_h1"] is None:
        out["guard_ok"] = None
    else:
        out["guard_ok"] = abs(out["guard_delta_h1"]) <= guard_k * out["mde_h1"]

    # ---- censoring: a treated arm at the floor cannot carry a point estimate -------------------
    for a in treated:
        prev = int(a["n_event"]) / int(a["n_cells"])
        if float(a["ap_h18"]) < 2.0 * prev:
            out["censored"].append(a["label"])

    if out["guard_ok"] is None:
        out["state"] = "UNDERPOWERED"
        out["detail"] = (
            "no arm carries an MDE at h1, so the anchor guard cannot be evaluated — and without "
            "it a move in AP(h18) cannot be attributed to retention rather than one-step skill"
        )
        return out
    if not out["guard_ok"]:
        out["state"] = "VOID"
        out["detail"] = (
            f"GUARD VIOLATED: mean dAP(h1) = {out['guard_delta_h1']:+.4f}, outside "
            f"{guard_k} x MDE_AP(h1) = {guard_k * out['mde_h1']:.4f}. Scheduled sampling moved "
            "the ANCHOR, so 'retention' is the wrong frame for what happened — report this as a "
            "traded failure, not as a retention result (pre-registration §4)."
        )
        return out

    p = out["p_value"]
    diff = out["diff_h18"]
    theta_abs = theta * out["mean_control_h18"]
    mde_ok = out["mde_h18"] is not None and abs(diff) >= mde_k * out["mde_h18"]

    if p <= alpha and mde_ok and out["endpoints_agree"]:
        out["state"] = "EFFECT"
        cens = (
            f" Magnitude CENSORED at the floor for {', '.join(out['censored'])} — report as '>=', "
            "never as a point estimate."
            if out["censored"]
            else ""
        )
        out["detail"] = (
            f"p={p:.4f} <= {alpha}, |mean dAP(h18)|={abs(diff):.4f} >= "
            f"{mde_k} x MDE={mde_k * out['mde_h18']:.4f}, and both endpoints agree in sign. "
            f"Scheduled sampling {'LOWERS' if diff < 0 else 'RAISES'} AP@h{H_STAR}.{cens}"
        )
    elif (
        p > alpha and out["mde_h18"] is not None and abs(diff) + mde_k * out["mde_h18"] < theta_abs
    ):
        out["state"] = "NULL"
        out["detail"] = (
            f"p={p:.4f} > {alpha} and the interval excludes a {theta:.0%} effect "
            f"({theta_abs:.4f} AP). This is a null, not a shrug: scheduled sampling does not "
            f"change AP@h{H_STAR} by as much as {theta:.0%} here, at this training length."
        )
    else:
        out["state"] = "UNDERPOWERED"
        if not out["endpoints_agree"]:
            why = (
                f"the endpoints disagree in sign (dAP(h18) {diff:+.4f}, dretention "
                f"{out['diff_retention']:+.4f})"
            )
        elif out["mde_h18"] is None:
            why = "no arm carries an MDE at h18, so the measurement floor is unknown"
        else:
            why = (
                f"p={p:.4f} and the interval does not exclude a {theta:.0%} effect "
                f"({theta_abs:.4f})"
            )
        out["detail"] = f"{why}. 'No effect' and 'could not tell' are not distinguishable here."
    return out
