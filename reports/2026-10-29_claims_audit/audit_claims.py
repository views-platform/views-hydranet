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
NUM = re.compile(r"\d+\.\d{3,}|\d+\.\d+\s?%")

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
    r"\bvs\b|\bversus\b|\bagainst\b|\bbeats?\b|better than|worse than|->|→|\bratio\b"
    r"|\bimprove|\boutperform", re.I
)
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


def out_corpus() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Map each measurement number to the `.out` files that PRODUCED it, and to those
    that merely DECLARED it. The distinction is the whole point -- see the docstring."""
    produced: dict[str, set[str]] = defaultdict(set)
    declared: dict[str, set[str]] = defaultdict(set)
    tracked = [n for n in sh("git", "ls-files", "reports").splitlines() if n.endswith(".out")]
    for name in tracked:
        try:
            text = (REPO / name).read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            nums = NUM.findall(line)
            if not nums:
                continue
            if DECLARED.search(line):
                if GOT.search(line):
                    # A gate line: what the run actually produced follows `got`.
                    head, _, tail = line.partition("got")
                    for n in NUM.findall(head):
                        declared[n.strip()].add(name)
                    for n in NUM.findall(tail):
                        produced[n.strip()].add(name)
                else:
                    for n in nums:
                        declared[n.strip()].add(name)
            else:
                for n in nums:
                    produced[n.strip()].add(name)
    return produced, declared


def claim_lines(path: Path) -> list[tuple[int, str]]:
    rows = []
    for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if NUM.search(line):
            rows.append((i, line.strip()))
    return rows


def classify(line: str, produced, declared) -> tuple[set[str], list[str]]:
    """Return the flags this line earns and the measurement numbers it carries."""
    nums = [n.strip() for n in NUM.findall(line)]
    flags: set[str] = set()

    for n in nums:
        if n in produced:
            continue
        flags.add("DECLARED-ONLY" if n in declared else "NO-ARTIFACT")

    if COMPARE.search(line):
        if not MODE.search(line):
            flags.add("MODE-UNNAMED")
        srcs = [produced[n] for n in nums if n in produced]
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
    produced, declared = out_corpus()
    print("=" * 110)
    print("CLAIMS AUDIT — every numeric claim in the documents touched since " + SINCE)
    print("=" * 110)
    print(f"  documents        {len(docs)}")
    print(f"  produced numbers {len(produced):,} distinct, from committed .out files")
    print(f"  declared-only    {len([n for n in declared if n not in produced]):,} distinct")

    rows = []
    for d in docs:
        for ln, line in claim_lines(d):
            f, nums = classify(line, produced, declared)
            if f:
                name = d.relative_to(REPO).as_posix()
                rows.append((name, ln, sorted(f), slots(line), line, nums))

    total = sum(len(claim_lines(d)) for d in docs)

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
    print(f"  claim lines carrying a measurement number   {total:,}")
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
    print(f"\n  ⭐ {len(clean)} documents carry NO flag — expected verdict STANDS, no edit:")
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
