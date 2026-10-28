"""M6 (#483) — re-apply the guard auditor's OWN surviving mutations and confirm each is now caught.

⛔ WHY THIS FILE EXISTS. Two independent clean-context guard audits (falsify guard mode) ran
against `tests/test_placement_rho.py`. The first applied 27 mutations to `scripts/placement_rho.py`
and 13 SURVIVED. The guards were strengthened, the second audit closed 12 of those 13 — the
thirteenth is a backstop the source itself admits is unreachable — and found SIX NEW live
survivors, two of which made the instrument return ~0.997 instead of 0.0 on the exact
constant-magnitude defect it was built to detect, with the whole suite green.

⭐ The point of this script is that the SECOND round of fixes must not be taken on trust either.
It re-applies each named survivor and reports whether the strengthened suite now fails.

⚠️ INDEPENDENCE, stated precisely because it is the thing that makes a guard audit worth anything.
The MUTATIONS are the independent auditor's design, not mine — I did not imagine these failures
and could not have. This script only EXECUTES them, so the independence that matters (who modelled
the failure) is preserved; what it does not have is an independent choice of which mutations to
try, which is why it is a verification of a prior audit and not a substitute for one.

⛔ It writes to `scripts/placement_rho.py` and restores it byte-for-byte, asserting the
restoration before it reports. Run only on a clean tree.

Run: python reports/2026-10-27_metric_redteam/verify_guard_mutations.py
"""
import pathlib
import subprocess

SRC = pathlib.Path("scripts/placement_rho.py")
PY = "/home/simon/anaconda3/envs/views-hydranet-env/bin/python"

#: the monthly per-month refusal, verbatim, as the anchor for X3 and X9
MONTHLY_REFUSAL = (
    "        if sel.sum() < 2:\n"
    "            raise PlacementRhoError(\n"
    '                f"month {m}: population {population!r} selected {int(sel.sum())} cells"\n'
    "            )"
)

MUTS = [
    ("X1  monthly ignores the declared population",
     "        sel = inm & _select(pred, truth, population)", "        sel = inm"),
    ("X2  the default population becomes the blind one",
     '    population: str = "events",', '    population: str = "all",'),
    ("X3  delete the monthly per-month refusal", MONTHLY_REFUSAL + "\n", ""),
    ("X9  the monthly refusal becomes a silent skip", MONTHLY_REFUSAL,
     "        if sel.sum() < 2:\n            continue"),
    ("X14 sweep invents the monthly cells when no months are given",
     """            if pool == "monthly" and month is None:
                continue""",
     """            if pool == "monthly" and month is None:
                out[(pop, pool)] = 0.0
                continue"""),
    ("X16 silent clip of negative predictions",
     "    pred = np.asarray(pred, dtype=np.float64)",
     "    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, None)"),
    ("X27 silent rounding of the truth",
     "    truth = np.asarray(truth, dtype=np.float64)",
     "    truth = np.round(np.asarray(truth, dtype=np.float64))"),
    ("X8  shape check weakened to a size check",
     "    if month.shape != truth.shape:", "    if month.size != truth.size:"),
    ("X15 the float() return coercion dropped",
     "    return float(r)", "    return r"),
    ("X7  the dtype coercion dropped",
     "    pred = np.asarray(pred, dtype=np.float64)", "    pred = np.asarray(pred)"),
]

orig = SRC.read_text()
print(f"{'mutation':<52s} {'applied':>8s}  verdict")
results = []
for name, old, new in MUTS:
    n = orig.count(old)
    if n != 1:
        print(f"{name:<52s} {'ANCHOR':>8s}  ⛔ anchor matched {n} times -- NOT RUN")
        results.append((name, "NOT RUN"))
        continue
    SRC.write_text(orig.replace(old, new, 1))
    r = subprocess.run([PY, "-m", "pytest", "tests/test_placement_rho.py", "-q", "--no-header",
                        "-x", "-p", "no:cacheprovider"], capture_output=True, text=True)
    SRC.write_text(orig)
    caught = r.returncode != 0
    tail = [ln for ln in r.stdout.strip().splitlines() if "passed" in ln or "failed" in ln]
    print(f"{name:<52s} {'yes':>8s}  {'CAUGHT' if caught else '⛔ SURVIVED'}"
          f"   {tail[-1] if tail else ''}")
    results.append((name, "CAUGHT" if caught else "SURVIVED"))

assert SRC.read_text() == orig, "⛔ the source was not restored"
print()
print("reverted: scripts/placement_rho.py is byte-identical to its pre-run state")
surv = [n for n, v in results if v != "CAUGHT"]
print(f"{len(results) - len(surv)} of {len(results)} CAUGHT"
      + (f";  still surviving: {surv}" if surv else ";  none survive"))
