"""sigma_hat_aggregate.py — RP-2: sigma-hat for B-1a and B-1b from three true seed replicates.

⛔ WHAT THIS IS AND IS NOT. B-1b's own committed output ends with: *"M-3 (sigma-hat) is absent and
blocked on views-models#625. Until it exists these are BASELINES and B-1b carries NO decision
rule."* This script supplies that sigma-hat. It is NOT the gate-slope sigma-hat of the 2026-10-21
readout -- that is a DIFFERENT STATISTIC and the two must never be quoted against each other
(GLOSSARY §5e, instance 4; the collision that produced W-83). See M74 and C-390.

THE ESTIMATOR IS LOCKED IN ADVANCE, not chosen after seeing the spread:
`2026-10-18_weight_decay_arms_prereg.md:1051` -- *"sigma-hat is the pooled sample sd"*, ddof=1,
n=3, **2 df**. ⚠️ Printed beside every value, because 2 df is thin and a bare sigma-hat invites
being read as sigma:
  * 95% CI on sigma-hat at 2 df: **[0.52 sigma-hat, 6.28 sigma-hat]** -- a **12.1-fold** span.
  * the honest bar against ONE unreplicated arm: t(0.975, 2) * sqrt(1 + 1/3) = **4.97 sigma-hat**
    (~3.0 at n=5, ~2.5 at n=10, so the cost of not replicating further is visible).
⇒ Every bar is derived from the UPPER limit, never from the point estimate.

D-4: the summary statistic is SWEPT (sd ddof=1 [LOCKED] / sd ddof=0 / range /
max pairwise |d|/sqrt2) and the whole PROFILE is reported -- B-1a over d x {E-W, isotropic}
x {in, out}, B-1b over n x {body, gated} x {in, out} -- because a sigma-hat flat in d is a
different instrument from one that is not. Reported absolutely AND as a CV.
{body, gated} x {in, out} -- because a sigma-hat flat in d is a different instrument from one that
is not. Reported absolutely AND as a CV.

C-351 slots: population = 64,818 land cells, in-sample months 421-456 and out-of-sample 457-469 ·
transform = as each source instrument computes it (B-1a variogram-ratio, B-1b pooled FSS) ·
estimator = pooled sample sd ddof=1 over n=3 seeds · CV = sigma-hat / |mean| .

⛔⛔ THE TRANSFER LIMIT, which is worse than it looks and is printed by this script so it cannot be
dropped when the number is quoted. No two rows share BOTH keys:

  | published B-1a 0.391 / B-1b 0.6267 | nb            | dropout 0.15  | checkpoint DELETED |
  | the live nb control                | nb            | dropout 0.125 | live               |
  | the three replicates               | truncated_nb  | dropout 0.15  | live               |

B-1a on truncated_nb returned 1.462 against nb's 0.391 -- a 3.7x level shift FROM THE FAMILY ALONE
-- and vehicle.py:43,60 says in capitals, twice, that numbers under one family are never quoted
against the other. ⇒ **sigma-hat discharges the noise-floor requirement for truncated_nb @ 0.15,
and only PROVISIONALLY, as a transported CV, for nb.**

================================================================================================
GATES, DECLARED BEFORE THE RUN
================================================================================================
G-4  ⭐ THE PARSER GATE, and it runs FIRST. The parser must recover **0.391** (B-1a, in sample,
     E-W, d1) and **0.6267** (B-1b, in sample, body alone, n1) from the CONTROL's *committed*
     `.out` files before it is allowed to parse a single replicate. ⛔ A parser that silently
     mismatches a column would produce a plausible sigma-hat from the wrong numbers -- the C-383
     shape, which has already cost this project a day.

G-5  THE ALLOWLIST. Each replicate `.out` must declare a vehicle sha in
     {c7745155, 80696bb5, 23fb3221}, all three present, and **amber (2a5efc2a) ABSENT** --
     different pod session, git 73c3f42, and views-models forbids pooling it. ⛔ VOID otherwise.

G-6  ⛔ NEVER POOL THE RERUN CONTRAST. silver has a second checkpoint at the same seed; mixing it
     in would make sigma-hat a mixture of two seed-contrasts and one rerun-contrast, not
     interpretable as either.

G-7  NON-VACUITY. At least one sigma-hat in the profile must be > 1e-9. If every cell is exactly
     zero the parse has almost certainly read one file three times, and G-5 would not catch it if
     the shas were read from the wrong lines.

G-8  The three replicates must report three DIFFERENT values for B-1a in/E-W/d1. Identical values
     across three seeds would contradict M74's finding that 0 of 175 float tensors agree.
"""

from __future__ import annotations

import pathlib
import re
import sys

import numpy as np

D = pathlib.Path("reports/2026-10-28_sigma_hat")
UNDERFIRE = pathlib.Path("reports/2026-10-12_underfire_diagnosis")
GAP = pathlib.Path("reports/2026-10-26_body_optimisation_gap")
ALLOW = {"c7745155", "80696bb5", "23fb3221"}
REFUSED = "2a5efc2a"
TAGS = ["_s42", "_s43", "_s44"]

B1A_D = [1, 2, 4, 8, 16, 32]
B1B_N = [1, 3, 5, 9, 17, 33, 65]

fails: list[str] = []


def gate(ok: bool, name: str, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


def parse_b1a(text: str) -> dict[tuple[str, str, int], float]:
    """B-1a: 'in sample      E-W        d1=0.391  d2=0.387 …' -> {(window, axis, d): value}."""
    out: dict[tuple[str, str, int], float] = {}
    for line in text.splitlines():
        m = re.match(r"\s+(in sample|out of sample)\s+(E-W|isotropic)\s+(d\d+=.*)$", line)
        if not m:
            continue
        win, axis = ("in" if m.group(1) == "in sample" else "out"), m.group(2)
        for dd, val in re.findall(r"d(\d+)=(-?[\d.]+)", m.group(3)):
            out[(win, axis, int(dd))] = float(val)
    return out


def parse_b1b(text: str) -> dict[tuple[str, str, int], float]:
    """B-1b: block headers set window and forecast; 'M-1  pooled FSS   n1=…' carries the values."""
    out: dict[tuple[str, str, int], float] = {}
    win = fc = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("in sample  months"):
            win = "in"
        elif s.startswith("out of sample  months"):
            win = "out"
        elif s == "body alone (mu)":
            fc = "body"
        elif s == "gated forecast (gate x mu)":
            fc = "gated"
        elif s.startswith("M-1  pooled FSS") and win and fc:
            for nn, val in re.findall(r"n(\d+)=(-?[\d.]+)", s):
                out[(win, fc, int(nn))] = float(val)
    return out


print("=" * 96)
print("RP-2 — sigma-hat for B-1a and B-1b, from three true seed replicates")
print("=" * 96)

# ------------------------------------------------------------------ G-4, the parser gate, FIRST
print("\nG-4  THE PARSER GATE — against the CONTROL's committed .out files")
ctl_a = parse_b1a((UNDERFIRE / "body_scoreboard.out").read_text())
ctl_b = parse_b1b((GAP / "b1b_placement_baseline.out").read_text())
gate(
    ctl_a.get(("in", "E-W", 1)) == 0.391,
    "G-4a",
    f"B-1a in/E-W/d1 parsed as {ctl_a.get(('in', 'E-W', 1))}, want 0.391 "
    f"({len(ctl_a)} cells parsed)",
)
gate(
    ctl_b.get(("in", "body", 1)) == 0.6267,
    "G-4b",
    f"B-1b in/body/n1 parsed as {ctl_b.get(('in', 'body', 1))}, want 0.6267 "
    f"({len(ctl_b)} cells parsed)",
)
gate(
    len(ctl_a) == 24 and len(ctl_b) == 28,
    "G-4c",
    f"full grids recovered: B-1a {len(ctl_a)}/24 cells, B-1b {len(ctl_b)}/28 cells",
)
if fails:
    print(
        "\n⛔ VOID — the parser cannot reproduce the control. It is NOT allowed to parse a "
        "replicate.\n   (A parser that mismatches a column makes a plausible sigma-hat from the "
        "wrong numbers: C-383.)"
    )
    sys.exit(1)

# ------------------------------------------------------------------ load the replicates
print("\nloading the three replicates")
A: dict[str, dict] = {}
B: dict[str, dict] = {}
shas: dict[str, str] = {}
for t in TAGS:
    fa, fb = D / f"body_scoreboard{t}.out", D / f"b1b_placement_baseline{t}.out"
    for f in (fa, fb):
        if not f.is_file():
            print(f"  ⛔ missing {f} — run the chain first")
            sys.exit(1)
    ta, tb = fa.read_text(), fb.read_text()
    A[t], B[t] = parse_b1a(ta), parse_b1b(tb)
    m = re.search(r"sha256 ([0-9a-f]{8})", ta) or re.search(r"vehicle sha ([0-9a-f]{8})", ta)
    shas[t] = m.group(1) if m else "UNKNOWN"
    print(f"  {t}  sha {shas[t]}  B-1a {len(A[t])} cells  B-1b {len(B[t])} cells")

gate(
    set(shas.values()) == ALLOW and REFUSED not in shas.values(),
    "G-5",
    f"allowlist: {sorted(shas.values())} == {sorted(ALLOW)}, amber({REFUSED}) absent",
)
gate(len(set(shas.values())) == 3, "G-6", "exactly three DISTINCT shas — no rerun contrast pooled")


# ------------------------------------------------------------------ sigma-hat
def summarise(vals: list[float]) -> dict[str, float]:
    v = np.asarray(vals, dtype=np.float64)
    pair = max(abs(v[i] - v[j]) for i in range(len(v)) for j in range(i + 1, len(v)))
    return {
        "mean": float(v.mean()),
        "sd_ddof1": float(v.std(ddof=1)),  # ⭐ LOCKED
        "sd_ddof0": float(v.std(ddof=0)),
        "range": float(v.max() - v.min()),
        "maxpair_over_sqrt2": float(pair / np.sqrt(2.0)),
    }


def profile(store: dict[str, dict], keys: list[tuple]) -> dict[tuple, dict]:
    ok = [k for k in keys if all(k in store[t] for t in TAGS)]
    return {k: summarise([store[t][k] for t in TAGS]) for k in ok}


keys_a = [(w, ax, d) for w in ("in", "out") for ax in ("E-W", "isotropic") for d in B1A_D]
keys_b = [(w, fc, n) for w in ("in", "out") for fc in ("body", "gated") for n in B1B_N]
prof_a, prof_b = profile(A, keys_a), profile(B, keys_b)

allsd = [c["sd_ddof1"] for c in list(prof_a.values()) + list(prof_b.values())]
gate(max(allsd) > 1e-9, "G-7", f"NON-VACUITY: max sigma-hat over the profile {max(allsd):.6f}")
v1 = [A[t][("in", "E-W", 1)] for t in TAGS]
gate(
    len(set(v1)) == 3,
    "G-8",
    f"the three seeds give three DIFFERENT B-1a in/E-W/d1 values: {v1}",
)

CI_LO, CI_HI, BAR = 0.52, 6.28, 4.97


def table(name: str, prof: dict, axis_lab: str, scale_lab: str) -> None:
    print("\n" + "=" * 96)
    print(f"  {name} — sigma-hat PROFILE (n=3, ddof=1, 2 df)")
    print("=" * 96)
    print(
        f"  {'window':<5} {axis_lab:<10} {scale_lab:>4} {'mean':>9} {'sd(1)':>9} {'CV':>8} "
        f"{'sd(0)':>9} {'range':>9} {'pair/√2':>9} {'4.97σ̂ bar':>11}"
    )
    for k, c in prof.items():
        cv = c["sd_ddof1"] / abs(c["mean"]) if c["mean"] else float("nan")
        print(
            f"  {k[0]:<5} {k[1]:<10} {k[2]:>4} {c['mean']:>9.4f} {c['sd_ddof1']:>9.4f} "
            f"{cv:>8.3f} {c['sd_ddof0']:>9.4f} {c['range']:>9.4f} "
            f"{c['maxpair_over_sqrt2']:>9.4f} {BAR * c['sd_ddof1']:>11.4f}"
        )
    sds = [c["sd_ddof1"] for c in prof.values()]
    cvs = [c["sd_ddof1"] / abs(c["mean"]) for c in prof.values() if c["mean"]]
    print(
        f"\n  across the profile: sigma-hat {min(sds):.4f}…{max(sds):.4f} "
        f"({max(sds) / max(min(sds), 1e-12):.1f}x), CV {min(cvs):.3f}…{max(cvs):.3f}"
    )


table("B-1a  (spatial structure, variogram ratio)", prof_a, "axis", "d")
table("B-1b  (placement, pooled FSS)", prof_b, "forecast", "n")

print("\n" + "=" * 96)
print("  HOW TO READ THESE — printed, not left to the reader")
print("=" * 96)
print(f"""
  ⚠️ 2 df. The 95% CI on each sigma-hat is [{CI_LO}σ̂, {CI_HI}σ̂] — a 12.1-fold span. ⛔ Derive every
     bound from the UPPER limit, never from the point estimate.
  ⚠️ The honest bar against ONE unreplicated arm is {BAR}σ̂ (printed per row above). It falls to
     ~3.0σ̂ at n=5 and ~2.5σ̂ at n=10, so the cost of stopping at three seeds is visible.
  ⛔ Is sigma-hat FLAT in the scale parameter? Read the per-row columns, not the summary: a
     sigma-hat that grows with d or n is a different instrument from one that does not, and only
     the profile can say which this is.
  ⛔⛔ TRANSFER. These are `truncated_nb` @ dropout 0.15. The published B-1a 0.391 / B-1b 0.6267
     are `nb` @ 0.15 on a DELETED checkpoint; the live `nb` control is @ 0.125. B-1a on trunc_nb
     returned 1.462 against nb's 0.391 — a 3.7x level shift from the family alone. ⇒ sigma-hat
     discharges the noise floor for **truncated_nb @ 0.15**, and only PROVISIONALLY, as a
     transported CV, for `nb`. Amend the prereg's rule 7 to say which half is discharged.
  ⛔ NOT the gate-slope sigma-hat of 2026-10-21 (M74 / C-390). Different statistic; never quoted
     against this one.""")

print("\n" + "=" * 96)
print(f"  {'ALL GATES PASS' if not fails else '⛔ FAILED GATES: ' + ', '.join(fails)}")
print("=" * 96)
sys.exit(1 if fails else 0)
