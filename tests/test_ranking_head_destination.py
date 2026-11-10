"""⭐⭐ THE DOSSIER'S GUARD — five mechanisms, one per named failure mode.

Simon named five failure modes on 2026-11-10, four of them measured facts about this assistant:
goes in circles · kills the idea before it is tried · slides to the gate instead of the body ·
loses focus · does not remember where we are going.

⛔⛔ WHY THIS IS A TEST AND NOT A PARAGRAPH. Every one of these five was **already written down**
in ``reports/2026-11-09_HANDOVER_ranking_body.md`` §3 and §7, committed in this repo before the
session that then committed all five again. Prose has been tried and has failed once, on the
record. A red test is the only mechanism in this repo that has reliably stopped the drift -- which
is why ~40 ``test_falsification_*`` files exist and why ``ADR-005`` calls for **Loud Failure**.

⇒ Each check below is mechanical, is checkable by Simon in one command, and **does not depend on
the assistant's intentions.** Each has a *discriminating half* that proves the check can fail.

⚠️ What this file does NOT hold: whether a measurement is actually *diagnostic*. No test holds that
judgment. It stays Simon's, exercised through the handover's three outcomes (B wins / A holds /
not resolvable at this ``n``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_DOSSIER = _REPO / "reports/2026-11-10_ranking_head_dossier"

#: ⭐⭐ THE FINISH LINE. The one number this programme exists to produce. Deleting or quietly
#: editing this string is the drift the file exists to catch, so it is asserted literally.
_FINISH_LINE_NUMBER = "+0.3127"
_FINISH_LINE_TERMS = ("ranking head", "rho_event", "trained")

#: ⛔ Mechanism 1. The next-actions budget Simon approved. Raising it is a REQUEST, not an edit:
#: change this number and the test says so, which is the point.
_APPROVED_NEXT_ACTIONS = 5

#: ⛔ Mechanism 2. Phrases that conclude against B. They are permitted ONLY on a line that also
#: carries a withdrawal marker -- recording a retracted past claim is history, not a conclusion.
_KILL_PHRASES = (
    # ⚠️ "not build" rather than "do not build": the narrower form missed "should not build",
    # caught by this file's own discriminating half on 2026-11-10. Match the verb, not the modal.
    "not build",
    "n't build",  # don't / shouldn't / can't
    "not worth building",
    "abandon",
    "b is dead",
    "b loses",
    "b does not work",
)
_WITHDRAWAL_MARKERS = ("withdraw", "superseded", "retracted", "was wrong", "no longer")

#: ⛔ Mechanism 4. Every document states, in its head, which part of Simon's question it serves.
_SERVES_MARKER = "**Serves:**"

pytestmark = pytest.mark.skipif(
    not _DOSSIER.is_dir(),
    reason="the ranking-head dossier has not been created yet",
)


def _docs() -> list[Path]:
    """-> every markdown document of the dossier, sorted. Excludes results/ and tools/."""
    return sorted(p for p in _DOSSIER.glob("*.md"))


def _readme() -> str:
    return (_DOSSIER / "00_README.md").read_text(encoding="utf-8")


def _names_the_finish_line(text: str) -> bool:
    """⭐ The checker, factored out so the discriminating half can attack IT and not a file."""
    low = text.lower()
    return _FINISH_LINE_NUMBER in text and all(t.lower() in low for t in _FINISH_LINE_TERMS)


def _unindexed(index_block: str, docs: list[Path]) -> list[str]:
    """-> the documents that exist on disk but are absent from the README's index."""
    return [p.name for p in docs if p.name not in index_block]


# ----------------------------------------------------------------------------------------------
# ⛔ Mechanism 5 — the destination. "Does not remember where we are going."
# ----------------------------------------------------------------------------------------------


def test_the_readme_names_the_finish_line():
    """⭐⭐ The single most important assertion in this repo for this programme.

    The finish line is **a trained ranking head's globally-measured ``rho_event`` against the
    body's +0.3127 +/- 0.0713**. If that sentence leaves ``00_README``, the programme has lost its
    destination and nothing else here matters.
    """
    assert _names_the_finish_line(_readme()), (
        f"00_README no longer names the finish line. It must state {_FINISH_LINE_NUMBER} and all "
        f"of {_FINISH_LINE_TERMS} -- a TRAINED ranking head's GLOBAL rho_event against the body's "
        "+0.3127. Without it the dossier cannot say what would end the programme."
    )


def test_the_finish_line_check_refuses_a_readme_without_it():
    """⛔ The discriminating half. A check that accepts everything proves nothing."""
    assert not _names_the_finish_line(
        "# 00 - README\n\nWe will train a ranking head and see if it ranks better.\n"
    ), "the checker accepted a README with no number, so the assertion above is vacuous"
    assert not _names_the_finish_line(
        f"the body ranks at {_FINISH_LINE_NUMBER} and we will compare the gate to it\n"
    ), "the checker accepted the number without 'trained', so a frozen-cube result would pass"


def test_every_dossier_document_is_in_the_readme_index():
    """⛔ Drift catcher: a document that exists but is not indexed is work nobody agreed to.

    This is the mechanism against silent growth -- the shape that produced 4,409 append-only lines
    in the evaluation report with retractions inline and no way to tell live from retracted.
    """
    docs = _docs()
    assert docs, "no markdown documents found in the dossier"
    missing = _unindexed(_readme(), docs)
    assert not missing, (
        f"{len(missing)} document(s) exist on disk but are absent from 00_README's index: "
        f"{missing}. Either index them or they should not be there."
    )


def test_the_index_check_refuses_an_unindexed_document():
    """⛔ The discriminating half for the index check."""
    assert _unindexed("| 00 | 00_README.md | spine |", [Path("02_design.md")]) == [
        "02_design.md"
    ], "the index checker missed a document that is plainly absent, so it cannot catch drift"


# ----------------------------------------------------------------------------------------------
# ⛔ Mechanism 1 — the closed budget. "Goes in circles."
# ----------------------------------------------------------------------------------------------


def test_the_next_actions_list_is_the_approved_length():
    """⛔ Four withdrawn definitions of done in one programme. The budget is the countermeasure.

    ⚠️ This test failing is NOT necessarily a defect -- it means the next-actions list changed.
    If Simon approved the change, ``_APPROVED_NEXT_ACTIONS`` moves with it. If he did not, the
    assistant added work to its own budget, which is the failure mode.
    """
    items = re.findall(r"^\s*-\s*\[[ x]\]", _readme(), flags=re.MULTILINE)
    assert len(items) == _APPROVED_NEXT_ACTIONS, (
        f"00_README lists {len(items)} next actions; the approved budget is "
        f"{_APPROVED_NEXT_ACTIONS}. Adding a row is a request to Simon, not an edit."
    )


# ----------------------------------------------------------------------------------------------
# ⛔ Mechanism 2 — the idea gets no veto from up-front work. "Kills the idea before it is tried."
# ----------------------------------------------------------------------------------------------


def test_no_document_concludes_against_B_without_a_trained_measurement():
    """⛔⛔ Simon's standing instruction, verbatim: *"any idea we propose, if you throw it around
    long enough, you end up refuting it and killing it. 100% of the time. It is only once I force
    you to do the empirics that we ever keep ideas."*

    ⇒ A kill requires a number from a **trained ranking head**. None exists, so no document here
    may conclude against B. Quoting a *withdrawn* past claim is history and is allowed -- but only
    on a line that says so.
    """
    offenders: list[str] = []
    for doc in _docs():
        for n, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            low = line.lower()
            hit = next((p for p in _KILL_PHRASES if p in low), None)
            if hit and not any(m in low for m in _WITHDRAWAL_MARKERS):
                offenders.append(f"{doc.name}:{n} ({hit!r})")
    assert not offenders, (
        "a document concludes against B without citing a trained-head measurement, and without "
        f"marking the claim withdrawn: {offenders}. Up-front work informs; it gets no veto."
    )


def test_the_kill_phrase_check_refuses_an_unmarked_conclusion():
    """⛔ The discriminating half: the marker exemption must not swallow everything."""
    unmarked = "The simulation shows we should not build B."
    marked = "⛔ WITHDRAWN 2026-11-09: 'do not build' -- wrong comparator, persistence not A."
    assert any(p in unmarked.lower() for p in _KILL_PHRASES)
    assert not any(m in unmarked.lower() for m in _WITHDRAWAL_MARKERS), (
        "the unmarked sentence looks withdrawn to the checker, so the exemption is too wide"
    )
    assert any(m in marked.lower() for m in _WITHDRAWAL_MARKERS)


# ----------------------------------------------------------------------------------------------
# ⛔ Mechanism 3 — the BODY is the target. "Slides to the gate instead of the body."
# ----------------------------------------------------------------------------------------------


def test_the_design_document_is_body_named_not_gate_named(capsys):
    """⛔ Measured bias: **3 of the last 7 pre-registrations are gate-named, 0 are body-named**
    (``feedback_the_body_is_the_target_not_the_gate``). Simon named the bias and the record agreed.

    The gate is licensed in exactly three roles -- it composes (``ADR-069``), it is the ordering
    benchmark (+0.4418), it defines the firing set. A fourth use is drift. This counts rather than
    argues.
    """
    ratios = []
    for doc in _docs():
        low = doc.read_text(encoding="utf-8").lower()
        body = low.count("body") + low.count("ranking head")
        gate = low.count("gate")
        ratios.append((doc.name, body, gate))
    with capsys.disabled():
        print("\n  document                      body+ranking-head   gate   ratio")
        for name, body, gate in ratios:
            r = f"{body / gate:.2f}" if gate else "inf"
            print(f"  {name:<30} {body:>13}   {gate:>4}   {r}")

    design = next((t for t in ratios if t[0] == "02_design.md"), None)
    assert design is not None, "02_design.md is missing"
    _, body, gate = design
    assert body >= 2 * gate, (
        f"02_design.md mentions the body/ranking head {body} times and the gate {gate} -- the "
        "design document has drifted onto the gate again. Required: body >= 2x gate."
    )


# ----------------------------------------------------------------------------------------------
# ⛔ Mechanism 4 — scope. "Loses focus and goes off on a tangent."
# ----------------------------------------------------------------------------------------------


def test_every_document_opens_by_naming_the_question_it_serves():
    """⛔ A document that cannot say which part of Simon's question it serves does not belong.

    Simon's question, 2026-11-10: *"if we changed the 'magnitude head' to a 'ranking head' do we
    think we could rank better?"* -- narrowed by him to *"Only ranking."*
    """
    missing = [
        doc.name
        for doc in _docs()
        if _SERVES_MARKER not in "\n".join(doc.read_text(encoding="utf-8").splitlines()[:12])
    ]
    assert not missing, (
        f"these documents do not declare what they serve in their first 12 lines: {missing}. "
        f"Every document opens with a '{_SERVES_MARKER}' line naming the part of the question it "
        "answers."
    )
