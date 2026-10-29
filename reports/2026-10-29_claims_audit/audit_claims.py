"""Triage every numeric claim in the documents this week touched.

WHY THIS EXISTS
---------------
Two failures this week made the written record untrustworthy rather than merely wrong:

  W-107  the frozen `nb` cube is TEACHER-FORCED, so "beats what ships by 22.5%" was a
         nowcast statement published as a forecast one. The arithmetic was never wrong;
         a label was.
  W-108  an AUC pair published as evidence for the inference mode moved TWO variables
         at once (emission and mode), so a one-variable attribution rested on a
         two-variable contrast.

The cost is not the numbers. It is that "what do I actually know?" stopped being
answerable. This script makes it answerable MECHANICALLY, so the answer does not depend
on my memory or my good faith.

WHAT IT DOES
------------
Walks every claim-bearing `.md` that `git log --since=2026-10-26` says we touched,
extracts every measurement number, and flags five failure classes drawn from what
actually went wrong rather than from imagination:

  NO-ARTIFACT    a measurement number that appears in no committed `.out` at all
  DECLARED-ONLY  a number that appears in `.out` files ONLY as a declared expectation
                 or a quoted ledger value, and was never PRODUCED by any run
  CROSS-SOURCE   two numbers compared on one line whose producing `.out` files are
                 disjoint -- the W-107 / W-108 shape
  MODE-UNNAMED   a comparison that names no inference mode (the C-351 slot whose
                 absence is C-399)
  ABSENCE        an absence asserted about the repo -- the W-99 shape, where a scoped
                 absence ("not under views-models") was written up as a general one
                 ("deleted") and the artifact was on disk all along

THE PRODUCED-VS-DECLARED DISTINCTION IS THE WHOLE POINT
-------------------------------------------------------
`0.039327` -- M67's published headline -- appears in FOUR committed `.out` files, so a
naive "does this number exist in an artifact?" check passes it. Read those four lines
and every one of them is a failed gate's expectation, an explicitly "NOT gated" record,
or a table row quoting M67. The number has never been produced by any implementation
(W-98). So:

  * on a gate line, only numbers AFTER the word `got` count as produced;
  * a line carrying an expectation marker and no `got` produces nothing;
  * a line citing a ledger ID (`M67`, `M85`, ...) is quoting, not producing.

C-351 SLOTS this script reports per claim: population / transform / estimator /
inference mode. It can only check the fourth mechanically; the other three are reported
as present-or-absent keywords for the human read that follows.

GATE G-0 -- RUN BEFORE ANY OUTPUT IS READ
-----------------------------------------
The pattern set is the thing under test, so it is tested against four claims already
KNOWN to be bad. If it misses one, the patterns are wrong and the fix belongs in this
file, not in the triage output (W-98: a copy gate's anchor is the implementation, never
a ledger row). On failure the script prints the gate and exits WITHOUT the triage, so an
untrusted pattern set cannot produce a reassuring table.

  G-0a  M85's AUC pair (0.99777 against 0.72340) is flagged
  G-0b  the "22.5% against what ships" phrase is flagged
  G-0c  the "deleted checkpoint" sentence is flagged ABSENCE
  G-0d  0.039327 is flagged DECLARED-ONLY  <-- the sharp one

AMENDMENT 1 — 2026-10-29, after the first run. THE PATTERNS WERE TOO LOOSE.
--------------------------------------------------------------------------
The first run flagged 940 of 1,679 lines (56.0%) against a pre-declared stop threshold
of 30%, so its output was not read. Three causes, each fixed structurally rather than by
quietly widening a tolerance (W-96):

  1. ⭐ THE REAL ONE: it flagged every LINE CARRYING A NUMBER, not every CLAIM. A table
     row of raw measurements inside a FINDING is DATA, and the document's own header
     supplies its mode and population. A claim is a line that ASSERTS -- bold text, a
     verdict marker, or a cross-document comparison. `is_claim` now makes that cut, and
     it is the tightening that matters.
  2. Rounding produced false NO-ARTIFACT flags: a document writing 0.0392 for a measured
     0.039186 is honest restatement, but the strings differ. Matching is now NUMERIC at
     the document's own stated precision -- both sides rounded to the same number of
     decimal places -- so a restatement matches and a different number does not.
  3. 110 `.out` files are on disk and only 85 are tracked, so numbers from untracked runs
     looked orphaned. Those are now their own, lesser class UNTRACKED-ARTIFACT: a real
     problem (the evidence is not committed) but NOT the W-98 problem (the number was
     never produced at all).

MODE-UNNAMED now reads a context window -- the nearest preceding heading plus the five
lines above -- because a FINDING that names its inference mode in its header does not
have to repeat it on every line.

AMENDMENT 2 — 2026-10-29. TWO SCOPE ERRORS OF MINE, found by SAMPLING the flags.
--------------------------------------------------------------------------------
Amendment 1 got the rate to 39.5%, still over the 30% stop threshold, so I read a sample
of the flags instead of tightening by guess. Both remaining causes were scope errors:

  1. ⭐ `→` IS NOT A COMPARISON IN THIS CORPUS, IT IS A TRANSITION. "MDE 0.0541 → 0.0086
     from pairing alone" and "the model left to feed on itself goes 0.474 → 0.257 →
     0.137" are before/after statements about ONE setup, and an inference-mode label is
     irrelevant to them. The claims where the mode is load-bearing name a RIVAL: a
     baseline, climatology, persistence, what ships, another method. MODE-UNNAMED now
     requires a rival reference, and arrows are dropped from it entirely. They remain
     available to CROSS-SOURCE, where the numbers' provenance does the work rather than
     the punctuation.
  2. ⛔ IT AUDITED WHOLE FILES RATHER THAN WHAT WE WROTE. `2026-09-30_evaluation_learning_
     report.md` is 2,600+ lines and contributed 56 flags because ONE line in it was
     touched this week. The scope is now the lines this window ADDED OR CHANGED, taken
     from `git log -p`, which is both what Simon asked for ("everything touched by us the
     last couple of days") and far tighter.

     ⚠️ ITS LIMIT, STATED PLAINLY: a pre-existing claim in a file we touched but did not
     change is NOT audited here. That is a deliberate boundary, not an oversight -- the
     alternative is auditing the repository's whole history. The one case that matters,
     pre-existing claims INVALIDATED by this week's retraction, was checked separately:
     every document quoting the withdrawn cross-regime numbers carries a banner, and
     ADR-073 §0 carries a correction block.

⛔ The four G-0 gates are UNCHANGED through both amendments and must still pass.
Tightening a pattern set is only legitimate if it still catches everything known to be
bad -- and G-0b's "22.5% against what ships" is itself a rival comparison, so Amendment
2's narrower MODE check is tested by the gate rather than assumed safe by me.

LIMITS, stated before the run
-----------------------------
This is a TRIAGE, not a verdict. It finds candidates by pattern; a human read assigns
STANDS / RESCOPE / DISCLAIM / RETRACT. It cannot detect a wrong number that is
internally consistent, and it cannot detect a correct number described in prose that
happens to carry no keyword. A flag is a reason to read, never a reason to banner --
over-disclaiming is itself damage, because it teaches distrust of sound work.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SINCE = "2026-10-26"

# A measurement number: a decimal carrying at least three decimal places, or any
# percentage. Deliberately excludes bare integers -- years, issue numbers, line numbers
# and population counts are not the failure class here, and including them buries the
# signal.
#   AMENDMENT 3 — a defect in this instrument, found by reading its own flags. The
#   exponent used to be unmatched, so `9.983e-01` was read as the number `9.983` and
#   `7.6e-17` as `7.6`, producing 8 phantom NO-ARTIFACT / DECLARED-ONLY flags. The
#   exponent is now consumed and `float()` normalises it, so the mantissa is never
#   compared as if it were the value. ⚠️ A false flag is not a harmless over-catch here:
#   the register's whole purpose is to be trustworthy line by line, and a phantom row
#   teaches distrust of a sound document exactly as a missing row hides a bad one.
EXP = r"(?:[eE][+-]?\d+)?"
NUM = re.compile(rf"\d+\.\d{{3,}}{EXP}|\d+\.\d+{EXP}\s?%")
# Only ABSOLUTE decimals carry checkable provenance. Amendment 1: a PERCENTAGE is
# normally derived from two measurements, so it legitimately appears in no `.out` and
# flagging it NO-ARTIFACT is a false positive. Percentages still take part in the
# CROSS-SOURCE and MODE-UNNAMED checks, where they are informative.
ABS_NUM = re.compile(rf"\d+\.\d{{3,}}{EXP}")
PRECISIONS = range(3, 7)

# A line that quotes a value from elsewhere rather than producing one.
DECLARED = re.compile(
    r"\bFAIL\b|\bexpect|\breproduce|\bpublished\b|\bledger\b|\bheadline\b|\bimplied\b"
    r"|\bNOT gated\b|\bnever gated\b|\bpredict|\bdeclared\b|\bM\d{2,}\b|\bcommitted\b",
    re.I,
)
GOT = re.compile(r"\bgot\b", re.I)

# The model is fed its own output, or it is fed the truth. Naming which is the slot
# whose absence is C-399. NOTE: in-sample / out-of-sample is a PARTITION, not a mode,
# and is deliberately absent from this list.
MODE = re.compile(
    r"teacher-forc|autoregress|nowcast|free-running|rollout|delivered|T=0|step[- ]zero"
    r"|step 0|h=1|one-step|AR-|TF-",
    re.I,
)
COMPARE = re.compile(
    r"\bvs\b|\bversus\b|\bagainst\b|\bbeats?\b|better than|worse than|->|→|\boutperform",
    re.I,
)
# Amendment 2: the claims where the inference mode is LOAD-BEARING are the ones naming a
# RIVAL. An arrow is a before/after transition within one setup and needs no mode label;
# "against what ships" does.
RIVAL = re.compile(
    r"\bvs\b|\bversus\b|\bagainst\b|\bbeats?\b|better than|worse than|\boutperform"
    r"|\bbaseline\b|climatolog|persistence|what ships|\bno model\b|\bdelivered\b"
    r"|conflictolog|white_ranger",
    re.I,
)
# A line that ASSERTS rather than TABULATES. Amendment 1: this is the cut that matters.
# A raw measurement row inside a FINDING is data, and the document's header supplies its
# mode and population; a bolded or marked sentence is a claim that must stand alone.
VERDICT = re.compile(r"\*\*|⭐|⛔|⚠️|✅|⇒")
ABSENCE = re.compile(
    r"\bdeleted\b|\bdoes not exist\b|\bis absent\b|cannot be regenerated|\bnowhere\b"
    r"|\bno longer\b|\bnever (?:written|existed|ran)\b|\bmissing\b|\bgone\b", re.I
)

POPULATION = re.compile(r"cell-month|842,634|land cell|month_id|months? \d{3}", re.I)
TRANSFORM = re.compile(r"log1p|log\(|count space|counts|no transform|asinh", re.I)
ESTIMATOR = re.compile(r"plug-in|fair|1/m|m\(m-1\)|pooled|median|mean\b", re.I)


def sh(*args: str) -> str:
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=True).stdout


def touched_docs() -> list[Path]:
    """The claim-bearing documents this week touched, per git rather than per memory."""
    out = sh("git", "log", f"--since={SINCE}", "--name-only", "--pretty=format:", "--all")
    names = sorted({n for n in out.splitlines() if n.endswith(".md")})
    return [REPO / n for n in names if (REPO / n).is_file()]


def written_lines() -> set[str]:
    """The text of every line this window ADDED or CHANGED (Amendment 2).

    Keyed by text rather than line number so that later edits shifting a file do not
    lose the mapping. ⚠️ A pre-existing claim in a file we touched but did not change is
    deliberately out of scope -- see Amendment 2 in the module docstring for why, and
    for what was checked separately instead."""
    out = sh("git", "log", f"--since={SINCE}", "-U0", "-p", "--pretty=format:", "--all",
             "--", "*.md")
    return {
        s for ln in out.splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
        if (s := ln[1:].strip())
    }


def _places(n: str) -> int:
    """How many decimal places the document actually stated, clamped to the index range.
    An exponent form is already normalised by `float()`, so it is compared at full
    precision rather than by counting the characters after its dot (Amendment 3)."""
    if "e" in n.lower():
        return max(PRECISIONS)
    k = len(n.split(".")[1])
    return min(max(k, min(PRECISIONS)), max(PRECISIONS))


def _index(names: list[str]) -> tuple[dict, dict]:
    """Index `.out` numbers by value at each precision, split PRODUCED vs DECLARED.

    Matching is numeric at the document's own stated precision -- both sides rounded to
    the same number of decimal places -- so a document restating a measured 0.039186 as
    0.0392 matches, while a genuinely different number does not (Amendment 1)."""
    produced: dict = {k: defaultdict(set) for k in PRECISIONS}
    declared: dict = {k: defaultdict(set) for k in PRECISIONS}

    def add(target: dict, nums: list[str], name: str) -> None:
        for n in nums:
            v = float(n)
            for k in PRECISIONS:
                target[k][round(v, k)].add(name)

    for name in names:
        try:
            text = (REPO / name).read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            nums = ABS_NUM.findall(line)
            if not nums:
                continue
            if not DECLARED.search(line):
                add(produced, nums, name)
            elif GOT.search(line):
                # A gate line: what the run actually produced follows `got`.
                head, _, tail = line.partition("got")
                add(declared, ABS_NUM.findall(head), name)
                add(produced, ABS_NUM.findall(tail), name)
            else:
                add(declared, nums, name)
    return produced, declared


def out_corpus():
    """Tracked outputs are the evidence. Untracked ones are a lesser, different problem:
    the number was produced but its evidence is not committed (Amendment 1)."""
    on_disk = sorted(p.relative_to(REPO).as_posix() for p in (REPO / "reports").rglob("*.out"))
    tracked = [n for n in sh("git", "ls-files", "reports").splitlines() if n.endswith(".out")]
    untracked = [n for n in on_disk if n not in set(tracked)]
    produced, declared = _index(tracked)
    untracked_produced, _ = _index(untracked)
    return produced, declared, untracked_produced, len(tracked), len(untracked)


def is_claim(line: str) -> bool:
    """A line that ASSERTS, not one that tabulates (Amendment 1, the main tightening)."""
    if VERDICT.search(line):
        return True
    return bool(COMPARE.search(line) and len(NUM.findall(line)) >= 2)


def claim_lines(path: Path) -> list[tuple[int, str, str]]:
    """Every line carrying a number, with the context a MODE check may legitimately use:
    the nearest preceding heading plus the five lines above it."""
    lines = path.read_text(errors="replace").splitlines()
    rows = []
    heading = ""
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            heading = line
        if NUM.search(line):
            ctx = heading + " " + " ".join(lines[max(0, i - 5):i])
            rows.append((i + 1, line.strip(), ctx))
    return rows


def classify(line: str, ctx: str, produced, declared, untracked) -> tuple[set[str], list[str]]:
    """Return the flags this claim earns and the absolute numbers it carries."""
    nums = ABS_NUM.findall(line)
    flags: set[str] = set()
    srcs = []

    for n in nums:
        k = _places(n)
        v = round(float(n), k)
        if v in produced[k]:
            srcs.append(produced[k][v])
        elif v in declared[k]:
            flags.add("DECLARED-ONLY")
        elif v in untracked[k]:
            flags.add("UNTRACKED-ARTIFACT")
        else:
            flags.add("NO-ARTIFACT")

    if RIVAL.search(line) and not MODE.search(line) and not MODE.search(ctx):
        flags.add("MODE-UNNAMED")

    if COMPARE.search(line):
        if len(srcs) >= 2 and all(
            not (a & b) for i, a in enumerate(srcs) for b in srcs[i + 1:]
        ):
            flags.add("CROSS-SOURCE")

    if ABSENCE.search(line):
        flags.add("ABSENCE")

    return flags, nums


def slots(line: str) -> str:
    return "".join(
        c if rx.search(line) else "-"
        for c, rx in (("P", POPULATION), ("T", TRANSFORM), ("E", ESTIMATOR), ("M", MODE))
    )


def main() -> int:
    docs = touched_docs()
    produced, declared, untracked, n_tracked, n_untracked = out_corpus()
    print("=" * 110)
    print("CLAIMS AUDIT — every numeric claim in the documents touched since " + SINCE)
    print("=" * 110)
    print(f"  documents          {len(docs)}")
    print(f"  .out evidence      {n_tracked} tracked, {n_untracked} on disk but untracked")
    print(f"  produced @6dp      {len(produced[6]):,} distinct values")
    print(f"  declared-only @6dp {len([v for v in declared[6] if v not in produced[6]]):,}")
    print("\n  AMENDMENT 1 applied: a CLAIM asserts (bold / verdict marker / a comparison")
    print("  carrying two numbers); a bare measurement row is DATA and is not audited.")

    written = written_lines()
    print(f"  scope              {len(written):,} lines ADDED or CHANGED this window")

    rows, total = [], 0
    for d in docs:
        name = d.relative_to(REPO).as_posix()
        for ln, line, ctx in claim_lines(d):
            if line not in written or not is_claim(line):
                continue
            total += 1
            f, nums = classify(line, ctx, produced, declared, untracked)
            if f:
                rows.append((name, ln, sorted(f), slots(line), line, nums))

    # ---- GATE G-0 -------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("GATE G-0 — the pattern set is tested against four claims ALREADY KNOWN to be bad")
    print("=" * 110)

    def hit(pred) -> bool:
        return any(pred(r) for r in rows)

    gates = {
        "G-0a  M85's AUC pair 0.99777 vs 0.72340 is flagged": hit(
            lambda r: "0.99777" in r[4] and "0.72340" in r[4]
        ),
        "G-0b  the '22.5% against what ships' phrase is flagged": hit(
            lambda r: "22.5" in r[4] and re.search(r"ships|delivered", r[4], re.I)
        ),
        "G-0c  a 'deleted checkpoint' sentence is flagged ABSENCE": hit(
            lambda r: "ABSENCE" in r[2] and re.search(r"checkpoint|cube|vehicle", r[4], re.I)
        ),
        "G-0d  0.039327 is flagged DECLARED-ONLY (never produced by any run)": hit(
            lambda r: "0.039327" in r[5] and "DECLARED-ONLY" in r[2]
        ),
    }
    for name, ok in gates.items():
        print(f"  {'PASS' if ok else '⛔ FAIL'}  {name}")

    if not all(gates.values()):
        print("\n⛔⛔ G-0 FAILED. The pattern set missed a claim known to be bad, so its")
        print("    silence on everything else means nothing. Fix this script, not the output.")
        print("    ⛔ The triage is deliberately NOT printed.")
        return 1

    # ---- the triage -----------------------------------------------------------------
    rate = len(rows) / total if total else 0.0
    print("\n" + "=" * 110)
    print("TRIAGE")
    print("=" * 110)
    print(f"  asserting claims carrying a measurement     {total:,}")
    print(f"  flagged for human read                      {len(rows):,}  ({rate:.1%})")
    if rate > 0.30:
        print("  ⚠️  flag rate above 30% — the patterns are too loose. TIGHTEN BEFORE READING,")
        print("      because over-disclaiming teaches distrust of sound work.")

    by_flag: dict[str, int] = defaultdict(int)
    for r in rows:
        for f in r[2]:
            by_flag[f] += 1
    print("\n  by class:")
    for f, n in sorted(by_flag.items(), key=lambda kv: -kv[1]):
        print(f"    {n:5,}  {f}")

    by_doc: dict[str, int] = defaultdict(int)
    for r in rows:
        by_doc[r[0]] += 1
    print(f"\n  by document ({len(by_doc)} of {len(docs)} carry at least one flag):")
    for d, n in sorted(by_doc.items(), key=lambda kv: -kv[1]):
        print(f"    {n:5,}  {d}")

    clean = sorted({d.relative_to(REPO).as_posix() for d in docs} - set(by_doc))
    print(f"\n  ⭐ {len(clean)} documents whose NEW claims carry no flag — expected STANDS:")
    for d in clean:
        print(f"           {d}")

    print("\n" + "=" * 110)
    print("THE FLAGGED CLAIMS  (slots: P=population T=transform E=estimator M=inference mode)")
    print("=" * 110)
    for doc, ln, f, sl, line, _ in sorted(rows):
        print(f"\n  {doc}:{ln}  [{sl}]  {','.join(f)}")
        print(f"    {line[:300]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
