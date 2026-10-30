#!/usr/bin/env python
"""determinism_matrix.py — RP-2 step 0: is the pipeline deterministic, and is the version inert?

⛔ WHY THIS RUNS BEFORE ANY COMPUTE. The research programme says RP-2's noise floor is "unrun".
It is NOT. `/home/simon/Documents/kestrel_sigma_backup_20261021/READOUT.md` (2026-10-21) already
reports sigma-hat for the gate-calibration slope `b`. ⇒ W-83's shape: a quantity declared missing
while the artifact sat on this machine. This script establishes what that readout actually settled
before RP-2 spends 37 minutes measuring anything.

⛔⛔ AND TWO COMMITTED DOCUMENTS IN THAT SAME DIRECTORY CONTRADICT EACH OTHER. This is the living
report's L8 class ("two committed documents can contradict each other, and no gate in either one
looks"), so it gets a gate rather than a preference:

  * ARTIFACT_METADATA.md, on `silver_0_1_4` (staged LATER, "pod five"):
      "Its 190 weight tensors are bit-identical to `silver`'s. Same config, same seed,
       same git sha; views-hydranet 0.1.4 instead of 0.1.3."
    ⇒ the 0.1.3 -> 0.1.4 boundary is NUMERICALLY INERT for this config.

  * READOUT.md, §"The confound", on the same boundary:
      "By elimination the excess sits on the 0.1.3/0.1.4 boundary."
    ⇒ that boundary CARRIES the excess that makes the as-launched sigma-hat 4-200x the
      same-version pair.

⛔ Both cannot be true. If the boundary is inert there is nothing to eliminate onto it, the
as-launched spread is a SEED effect, and the honest sigma-hat is the as-launched 0.0351..0.0932
on df=2 -- NOT the same-version pair of 0.0002..0.0196, which would then be an underestimate
got by dropping an outlying seed. That changes every decision rule RP-2 hands downstream.

⇒ This script does not argue. It loads every checkpoint and compares tensors.

C-351 slots: population = the 6 staged kestrel checkpoints at 300 lessons · transform = raw
float32 weight tensors, no normalisation · estimator = exact bitwise equality
(`torch.equal`), with max |delta| printed only for pairs that are NOT equal.

================================================================================================
GATES, DECLARED BEFORE THE RUN
================================================================================================
G-1  VOID GATE. Every `.pt` and every `.pt.config.json` matches `MANIFEST.sha256`. ⛔ If any file
     fails, the artifacts are not what the two documents describe and EVERYTHING below is void --
     including the contradiction itself, which would then be about files we do not have.

G-2  The READOUT's own M4 claim, re-measured: `silver` vs `amber` is 190 of 190 bit-identical.
     ⚠️ Note what this asserts: bit-identity across DIFFERENT git shas (2e73de3 vs 73c3f42) AND
     different library versions (0.1.3 vs 0.1.2). That is a strong claim and it is the basis of
     "stochastic run-to-run variation is zero".

G-3  ⭐ THE DECISIVE ONE. ARTIFACT_METADATA's claim: `silver` vs `silver_0_1_4` is 190 of 190
     bit-identical, i.e. the 0.1.3 -> 0.1.4 boundary is numerically inert.

G-4  NON-VACUITY. At least one pair in the matrix must DIFFER. If every pair of all six
     checkpoints is bit-identical, the comparison is not measuring what it claims -- it is
     reporting a loader that returns the same object -- and G-2/G-3 passing would mean nothing.
     ⇒ `bronze` (seed 43) vs `cobalt` (seed 44) MUST differ.

G-5  The tensor count is COUNTED, not assumed. Both documents say 190; a count over a state dict
     is a computation, not an observation (W-92's lesson).

================================================================================================
MY RISKY PREDICTION, recorded before execution (PREDICTION_LEDGER)
================================================================================================
P1  G-2 and G-3 BOTH PASS, and G-4 passes. ⇒ the version boundary is inert, the READOUT's
    "by elimination" inference is REFUTED, and the as-launched sigma-hat IS a seed variance.
    Confidence: moderate. Both claims are asserted by documents written from a manifest, and
    ARTIFACT_METADATA explicitly calls `silver_0_1_4` "the single-variable test that killed the
    library-version hypothesis" -- so the later document already knows it contradicts the earlier.

P2  ⛔ The counter-case I must not hide: if G-3 FAILS, ARTIFACT_METADATA is wrong, the READOUT's
    version inference survives, and the same-version pair is the right sigma-hat after all.
"""

from __future__ import annotations

import hashlib
import itertools
import pathlib
import sys

import torch

BACKUP = pathlib.Path("/home/simon/Documents/kestrel_sigma_backup_20261021")
ARMS = {
    "amber": "amber/calibration_model_20261013_010739.pt",
    "silver": "silver/calibration_model_20261019_152513.pt",
    "silver_0_1_4": "silver_0_1_4/calibration_model_20261021_092504.pt",
    "bronze": "bronze/calibration_model_20261021_010229.pt",
    "cobalt": "cobalt/calibration_model_20261021_005527.pt",
    "indigo": "indigo/calibration_model_20261020_113202.pt",
}
#: seed / library version / weight_decay, from ARTIFACT_METADATA.md's own table
KEYS = {
    "amber": (42, "0.1.2", 0.1),
    "silver": (42, "0.1.3", 0.1),
    "silver_0_1_4": (42, "0.1.4", 0.1),
    "bronze": (43, "0.1.4", 0.1),
    "cobalt": (44, "0.1.4", 0.1),
    "indigo": (42, "0.1.4", 0.0),
}

fails: list[str] = []


def gate(ok: bool, name: str, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        fails.append(name)


print("=" * 96)
print("RP-2 step 0 — the determinism matrix, and a contradiction between two documents")
print("=" * 96)

# ---------------------------------------------------------------- G-1, the void gate
print("\nG-1  VOID GATE — every staged file against MANIFEST.sha256")
manifest = (BACKUP / "MANIFEST.sha256").read_text().strip().split("\n")
bad: list[str] = []
checked = 0
for line in manifest:
    want, rel = line.split(None, 1)
    f = BACKUP / rel.strip()
    if not f.is_file():
        bad.append(f"{rel.strip()} MISSING")
        continue
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    checked += 1
    if h != want:
        bad.append(f"{rel.strip()} sha mismatch")
detail = f"{checked} files verified, {len(bad)} bad" + (f" -> {bad[:3]}" if bad else "")
gate(not bad, "G-1", detail)
if fails:
    print("\n⛔ VOID — the artifacts are not what the documents describe. Nothing below runs.")
    sys.exit(1)

# ---------------------------------------------------------------- load
print("\nloading six state dicts (CPU)")
sd = {a: torch.load(BACKUP / p, map_location="cpu", weights_only=False) for a, p in ARMS.items()}
counts = {a: len(d) for a, d in sd.items()}
print("  tensor counts:", counts)
gate(
    len(set(counts.values())) == 1,
    "G-5",
    f"all six state dicts hold the SAME number of entries: {sorted(set(counts.values()))} "
    f"(both documents say 190 — COUNTED, not assumed)",
)
n_tensors = next(iter(counts.values()))

# ---------------------------------------------------------------- the matrix
print("\n" + "=" * 96)
print("  THE PAIRWISE BIT-IDENTITY MATRIX — bit-identical tensors out of", n_tensors)
print("=" * 96)
names = list(ARMS)
ident: dict[tuple[str, str], int] = {}
maxdelta: dict[tuple[str, str], float] = {}
for a, b in itertools.combinations(names, 2):
    same = 0
    md = 0.0
    for k in sd[a]:
        ta, tb = sd[a][k], sd[b][k]
        if torch.equal(ta, tb):
            same += 1
        elif ta.is_floating_point():
            md = max(md, float((ta - tb).abs().max()))
    ident[(a, b)] = same
    maxdelta[(a, b)] = md

hdr = "  " + " " * 14 + "".join(f"{n[:12]:>14}" for n in names)
print(hdr)
for a in names:
    row = f"  {a:<14}"
    for b in names:
        if a == b:
            row += f"{'—':>14}"
        else:
            key = (a, b) if (a, b) in ident else (b, a)
            row += f"{ident[key]:>10}/{n_tensors:<3}"
    print(row)

print("\n  max |delta| for pairs that are NOT bit-identical:")
for (a, b), md in sorted(maxdelta.items(), key=lambda kv: -kv[1]):
    if ident[(a, b)] != n_tensors:
        sa, sb = KEYS[a], KEYS[b]
        diff = [
            lab
            for lab, x, y in (("seed", sa[0], sb[0]), ("ver", sa[1], sb[1]), ("wd", sa[2], sb[2]))
            if x != y
        ]
        print(
            f"    {a:<14} vs {b:<14}  {ident[(a, b)]:>4}/{n_tensors} identical  "
            f"max|Δ| {md:.6e}   differs in: {','.join(diff) or 'NOTHING RECORDED'}"
        )

# ---------------------------------------------------------------- the declared gates
print("\n" + "=" * 96)
print("  GATES")
print("=" * 96)


def pair(a: str, b: str) -> int:
    return ident[(a, b)] if (a, b) in ident else ident[(b, a)]


gate(
    pair("silver", "amber") == n_tensors,
    "G-2",
    f"READOUT's M4: silver vs amber {pair('silver', 'amber')}/{n_tensors} bit-identical "
    f"(across git 2e73de3 vs 73c3f42, lib 0.1.3 vs 0.1.2)",
)
gate(
    pair("silver", "silver_0_1_4") == n_tensors,
    "G-3",
    f"⭐ ARTIFACT_METADATA: silver vs silver_0_1_4 {pair('silver', 'silver_0_1_4')}/{n_tensors} "
    f"bit-identical ⇒ the 0.1.3→0.1.4 boundary is numerically INERT",
)
gate(
    pair("bronze", "cobalt") != n_tensors,
    "G-4",
    f"NON-VACUITY: bronze(42+1) vs cobalt(42+2) differ — {pair('bronze', 'cobalt')}/{n_tensors} "
    f"identical, max|Δ| {maxdelta[('bronze', 'cobalt')]:.6e}",
)

# ---------------------------------------------------------------- the adjudication
print("\n" + "=" * 96)
print("  THE ADJUDICATION — which of the two documents is right about the version boundary")
print("=" * 96)
inert = pair("silver", "silver_0_1_4") == n_tensors
v_meta = "UPHELD" if inert else "REFUTED"
v_read = "REFUTED" if inert else "UPHELD"
print()
print(f"  ARTIFACT_METADATA  'silver_0_1_4 == silver; 0.1.4 vs 0.1.3'      -> {v_meta}")
print(f"  READOUT            'the excess sits on the 0.1.3/0.1.4 bound'   -> {v_read}")
print()
if inert:
    print("""  ⇒ The version boundary moves NOTHING here, so it cannot carry the excess the
    READOUT eliminated onto it. With stochastic variation measured at zero and hardware excluded,
    the only variable left between silver(42) and bronze(43)/cobalt(44) is THE SEED.

  ⛔ CONSEQUENCE FOR RP-2, and it is the opposite of the READOUT's reading:
    * the as-launched sigma-hat_b (0.0351 n>=1000 / 0.0932 all / 0.0434 sqrt-n, df=2) IS a seed
      variance and is the honest number;
    * the same-version pair (0.0002..0.0196, df=1) is NOT a cleaner estimate -- it is the
      same estimate with the outlying seed dropped, a 3x..200x UNDERSTATEMENT of the floor;
    * so every decision rule built on the same-version pair is built on the smaller of two
      numbers for a reason that does not hold.

  ⚠️ What this does NOT establish: that seed 42 is an outlier rather than 43/44 being the
    pair that happens to agree. n=3, df=2, 95% CI [0.52 sigma-hat, 6.28 sigma-hat] -- a
    12.1-fold span. The READOUT's own pre-registered prediction is also resolved here: it said
    seed-42-on-0.1.4 would show bronze/cobalt's positive-throughout pattern if version mattered.
    silver_0_1_4 IS seed-42-on-0.1.4 and is bit-identical to silver, which CROSSES ZERO.
    ⇒ the version hypothesis is dead on its own pre-registered terms.""")
else:
    print("""  ⇒ ARTIFACT_METADATA.md is WRONG and must be corrected. The READOUT's version
    inference survives, and the same-version pair remains the right sigma-hat.
    ⇒ P1 is REFUTED and P2 holds.""")

print("\n" + "=" * 96)
print(f"  {'ALL GATES PASS' if not fails else '⛔ FAILED GATES: ' + ', '.join(fails)}")
print("=" * 96)
