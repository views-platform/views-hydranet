"""⛔ FAILING STUBS — /falsify audit of "we are ready to write up an R&D dossier on this".

Audited 2026-11-10. Claim: everything an R&D dossier for the **ranking-head** design requires --
its maths, its references, its harness, its pre-registered criterion -- is on disk or derivable
now, without a new measurement.

⛔ Verdict: **FALSIFIED.** ⚠️ **Seven** hard findings (plus one soft and one survival) -- this
docstring said "Five" and the audit report said "six"; both were miscounts of the same seven tests,
corrected 2026-11-10. Every test in this file asserts a thing the dossier
would have to state and **currently fails**. They are not regression guards; they are the gate on
writing docs ``02_design``, ``03_harness_and_invariants`` and ``05_analysis_plan``.
Each one passes only when the corresponding hole is closed, and then it becomes the guard.

⭐ Contract under test: ``~/.claude/skills/rnd-dossier/references/structure.md`` -- the canonical
8-doc set (``00_README`` … ``07_experiment_log``) -- plus
``reports/2026-11-09_HANDOVER_ranking_body.md`` §2 (the five constraints on the decision rule).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_DOSSIER = _REPO / "reports/2026-11-10_rank_coupling_verdict"
#: The ranking-head dossier this audit gated the writing of.
_DOSSIER_DIR = _REPO / "reports/2026-11-10_ranking_head_dossier"
_GLOSSARY = _REPO / "reports/GLOSSARY.md"

#: ⛔ The marker a REAL pre-registration carries. Deliberately an unmistakable literal: prose
#: *about* the absence of a primary quantity must not satisfy the check. See the corrected test.
_DECLARATION_MARKER = "PRIMARY QUANTITY DECLARED:"

#: ⭐ The vocabulary this programme has used in committed artifacts and in the chat record.
#: ``feedback_use_fixed_glossary``: use ONLY ``reports/GLOSSARY.md``; if a word is missing, EDIT it
#: -- never invent a synonym. ``06_glossary`` of a dossier is written FROM this file.
_PROGRAMME_TERMS = (
    "ranking head",
    "rank-coupled",
    "empirical draw",
    "firing set",
    "midrank",
    "MCB",
    "DSC",
    "Schaake",
    "ensemble copula coupling",
    "listwise",
)


def test_glossary_defines_the_programme_vocabulary():
    """⛔ HARD. ``06_glossary`` must be written from the LOCKED glossary, which defines none of it.

    Measured 2026-11-10: 0 of 10 terms present in ``reports/GLOSSARY.md`` (500 lines). A bare
    ``grep -ci UNC`` returns 21 and is a substring artifact ("function", "uncertainty"), not a
    definition -- which is why this test matches a definition line, not a substring.
    """
    text = _GLOSSARY.read_text(encoding="utf-8").lower()
    missing = [t for t in _PROGRAMME_TERMS if t.lower() not in text]
    assert not missing, (
        f"{len(missing)} of {len(_PROGRAMME_TERMS)} programme terms are undefined in the locked "
        f"glossary: {missing}. The standing rule is to EDIT the glossary, not to invent "
        "vocabulary "
        "per-report -- and this programme invented all of it."
    )


def test_dossier_artifacts_are_tracked_in_git():
    """⛔ HARD. 2 of 22 files under the dossier are in git; ``reports/`` is gitignored so
    ``git status`` reports NOTHING about the other 20.

    Both tracked files are pre-registrations (``3b807bb``, ``9ab37ba``). Every tool, every result,
    ``03_FINDING.md`` and ``07_experiment_log.md`` are untracked and INVISIBLE. ⇒ A dossier citing
    them cites evidence that does not exist in any clone. The epic plan's own EVIDENCE RULE:
    *"Nothing counts unless the code AND its inputs are both on disk and I can re-run it."*
    """
    files = sorted(p for p in _DOSSIER.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert files, f"no dossier files found under {_DOSSIER}"
    untracked = [
        p.relative_to(_REPO).as_posix()
        for p in files
        if subprocess.run(
            ["git", "ls-files", "--error-unmatch", p.relative_to(_REPO).as_posix()],
            cwd=_REPO,
            capture_output=True,
        ).returncode
        != 0
    ]
    assert not untracked, (
        f"{len(untracked)} of {len(files)} dossier files are not tracked by git, and ``reports/`` "
        f"is gitignored so they do not appear in ``git status`` either: {untracked}"
    )


def test_the_ranking_loss_has_a_defined_label_transform():
    """⛔⛔ HARD — the biggest hole. Zhu2020 Eq.2 is the formulation the literature pass chose.
    It consumes **rating levels** ``r_t``. Our truth is **raw counts**. Nothing on disk says how
    one becomes the other.

    ⭐ The equation itself IS available, verbatim, from
    ``~/brain/9_library/papers/Zhu2020_ListwiseUniqueRatings.pdf``:

        L(f; R, D) = -1/(|R|-1) * sum_{t=1}^{|R|-1} (2^{r_t} - 1) * sum_{d in c_t} ln P_t(d)

    ⇒ the maths of the LOSS is in hand. The maths of its INPUT is not.
    """
    hits = subprocess.run(
        ["git", "grep", "-l", "-iE", "label transform|rating level|relevance grade"],
        cwd=_REPO,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert hits, (
        "no tracked file defines the map from raw counts to the rating levels the ranking loss "
        "consumes. Until it exists, ``02_design`` cannot state the loss and ``05_analysis_plan`` "
        "cannot pre-register it."
    )


def test_the_dyadic_transform_keeps_the_gain_weight_finite_over_the_whole_truth_range():
    """⭐ The DECIDED design, guarded. ``r = ceil(log2(1 + y))`` then ``2^r - 1`` must be finite in
    float32 for every count in ``[0, y_max]``.

    ⚠️ CORRECTED 2026-11-10. This test previously asserted that ``2^y - 1`` was finite on **raw
    counts** -- which can never hold, so it was a permanently-red documentation stub rather than a
    guard. The hole is closed by the label transform (``02_design`` §2), not by arithmetic, so the
    guard now encodes the decision. The discriminating half below keeps it honest.

    Measured on the S11 fixed population (n = 6,689 cell-months, ``gate >= 0.05``, months 463-469):
    truth maximum **681**, **115** distinct raw values -> **10** levels, weights in **[0, 1023]**.
    """
    truth_max = 681  # results/s11_tau_sweep.out line 65
    y = np.arange(0, truth_max + 1, dtype=np.float64)

    r = np.ceil(np.log2(1.0 + y))
    w = np.float32(2.0) ** r.astype(np.float32) - np.float32(1.0)
    assert np.all(np.isfinite(w)), "the dyadic transform must keep every gain weight finite"
    assert w.max() == 1023.0, f"expected a top weight of 1023, got {w.max()}"
    assert w[0] == 0.0, "a zero count must carry zero gain weight"
    assert np.unique(r).size == 11, f"expected levels 0..10, got {np.unique(r).size}"

    # ⛔ The discriminating half: raw counts must FAIL, or the transform is guarding nothing.
    raw = np.float32(2.0) ** np.float32(truth_max) - np.float32(1.0)
    assert not np.isfinite(raw), (
        "2^681 - 1 is finite in float32 on this machine, so the transform is not load-bearing and "
        "this guard proves nothing. C-420 would not reproduce."
    )


def test_the_design_declares_which_set_the_loss_ranks():
    """⭐ The DECIDED design, guarded. ``02_design`` must state the ranked list explicitly and name
    the globally-measured instrument for the gap it opens.

    ⚠️ CORRECTED 2026-11-10. This test previously asserted ``'window_dim': 32`` was ABSENT from the
    vehicle config -- i.e. it demanded the training geometry change. That is not the decision:
    ``batch_size=1`` means there is no batch to concatenate (``training_engine.py:1750``), so the
    list IS the 1,024-cell patch-month, declared, with the patch->global ordering gap turned into a
    pre-registered falsifier measured by ``rho_event``. The guard now requires that declaration
    rather than a config change that was never going to happen.
    """
    design = _DOSSIER_DIR / "02_design.md"
    if not design.exists():
        pytest.skip("02_design.md has not been written")
    text = design.read_text(encoding="utf-8")
    low = text.lower()

    assert "1,024-cell patch-month" in low or "1024-cell patch-month" in low, (
        "02_design does not state the ranked list. The set the loss orders must be named, because "
        "it is NOT the set the emission couples."
    )
    assert "batch_size=1" in low or "batch_size: 1" in low, (
        "02_design must cite why the batch cannot be the list -- there is no batch dimension"
    )
    assert "rho_event" in low or "ρ_event" in text, (
        "02_design must name the globally-measured instrument for the patch->global gap it opens"
    )
    assert "falsifier" in low, (
        "the patch->global gap must be carried as a pre-registered falsifier, not an assumption"
    )


def test_a_primary_superiority_quantity_is_declared_on_disk():
    """⛔ HARD, and it bites my own proposal. No tracked file names a primary quantity for the
    B-vs-A decision -- and ``MCB`` has already been MEASURED (``MCB_A`` 0.138 vs ``MCB_B`` 2.952 at
    tau=0.20, ``results/s11_tau_sweep.out``).

    ``11_PREREG_tau_and_decomposition.md`` introduces ``MCB`` as an **attribution** device only.
    Handover §2 constraint 3 (``C-3531``, ``Gneiting2011``): the scoring rule is declared **EX
    ANTE**; §3 records *"I violated this four times in one day by picking the metric after the
    fact."* Declaring ``MCB`` primary now, knowing the sign and the scale, is the fifth.

    ⚠️ CORRECTED 2026-11-10: this test was a FALSE GREEN. It grepped for the phrase "primary
    quantity", which the glossary's §4d then satisfied **while saying that none has been named** --
    a check met by prose about its own subject's absence. It now requires an unmistakable
    declaration marker that only a real pre-registration would carry.
    """
    hits = subprocess.run(
        ["git", "grep", "-l", "-F", _DECLARATION_MARKER],
        cwd=_REPO,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert hits, (
        f"no tracked file carries '{_DECLARATION_MARKER}'. 05_analysis_plan cannot be "
        "written, and "
        "the quantity cannot be chosen after its value is known (C-3531). This is blocked on "
        "Simon, not on work."
    )


def test_the_margin_is_in_units_of_the_primary_quantitys_own_dispersion():
    """⛔ HARD. Handover §2 constraint 4: *"The margin is in units of MEASURED seed dispersion,
    never percent."* The only measured dispersion is ``sigma-hat_paired = 0.4441`` in **fair-CRPS**
    units (``results/s7_sigma_hat.out`` line 66) -- the dispersion of a different quantity.

    ``MCB``'s own seed dispersion has never been measured. Handover §5 item 1 (the ``baseline``
    session) was commissioned for the OLD candidate criteria and is still pending.
    """
    out = _DOSSIER / "results/s7_sigma_hat.out"
    if not out.exists():
        pytest.skip("S7 output absent")
    text = out.read_text(encoding="utf-8")
    assert "sigma-hat_paired" in text and "MCB" in text, (
        "the measured dispersion is in fair-CRPS units, not in units of the proposed primary "
        "quantity. A margin in the wrong units is constraint 4's failure mode exactly."
    )
