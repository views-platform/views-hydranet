#!/usr/bin/env python3
"""Verify EXP-02 after every arm — and never touch EXP-01's results.

⛔⛔ WHY THIS FILE EXISTS. `launch_exp02.sh` previously handed `run_queue.sh` the EXP-01 dossier's
`verify_bakeoff.py`, on the reasoning that reusing an instrument unchanged is better than copying
it. That was wrong in a way that **destroys committed data**:

  * `verify_bakeoff.py:33` hardcodes `RES = _D / "results"` — EXP-01's directory — and contains
    **zero** references to the `RES_DIR` the launcher passes it.
  * It writes `results/VERDICT.md` and `results/bakeoff_state.json` at `:145-146`, and **both are
    git-tracked.**
  * All twelve EXP-01 arm directories are gone from views-models.

⇒ The first EXP-02 arm would have burned ~3 GPU-h, then written a twelve-row *"arm directory
missing"* table **over EXP-01's committed read-out**, then returned 1 and aborted the queue blaming
EXP-01's absent arms. ⭐ Found by an independent `/code-review max`, verified here before the fix.

⚠️ AND IT MADE A PRE-REGISTERED GATE FALSE. `PREREG.md:226` claims gate **F1**: *"verify_bakeoff.py
re-asserts the arm's `model` independently of the builder's declaration."* `verify_bakeoff.py`'s
`ARCHES` names none of EXP-02's arms, so F1 never ran for this experiment. It does now, here.

WHAT THIS DOES, in the parent's three legs
------------------------------------------
1. **Identity** — every arm's `model` is read from its own config and must match the architecture
   its label claims. The builder declares `arm_identity`; a `/falsify guard` audit showed the
   declaration itself is unguarded, so this re-asserts it independently.
2. **Postflight** — `scripts.arm_postflight.audit_arm` on every scored arm, against its own-seed
   control's score CSV.
3. **Read-out** — ⛔ at **h1 and h36**, never h18. EXP-01's own log names h18 as its design error:
   *"it sits almost exactly at the sign change and reports 'nothing is happening' for a pattern
   that is unmistakable at h1 and h36."*

⭐ `_hp`, `_ap` and `audit_arm` are IMPORTED from the parent, not copied — the duplication that
matters is a gate's REQUIREMENTS, not its helpers, and a third copy of a helper is how two
instruments drift.

⛔ EXP-02's control is an ARM OF THIS QUEUE (`ctlfullzero_<seed>`), not a committed CSV from an
earlier programme: the `fullzero_*` directories no longer exist and pairing against their August
scores would vary the code version as well as `model`.

Never raises: a crashed verifier must stop the queue with a clear message, not a traceback the
queue records as "verify failed" without saying why.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_D = _HERE.parent
_HN = _D.parents[1]
_BAKEOFF_TOOLS = _HN / "reports" / "2026-08-24_architecture_bakeoff_dossier" / "tools"
sys.path.insert(0, str(_HN))
sys.path.insert(0, str(_BAKEOFF_TOOLS))

from verify_bakeoff import _ap, _hp  # noqa: E402

from scripts.arm_postflight import audit_arm  # noqa: E402

#: ⛔ THE FIX. `RES_DIR` is what `run_queue.sh` hands every verifier, and honouring it is the whole
#: difference between writing into this dossier and writing over another one's committed verdict.
#: No default: an unset RES_DIR must stop the queue, not silently pick a directory.
_ENV_RES = os.environ.get("RES_DIR")

ARCHES = {"HydraBNUNet06_LSTM4": "ctl", "UpsampleConv2": "upctwo"}
CONTROL_ARCH = "HydraBNUNet06_LSTM4"
SEEDS = {42: "fortytwo", 43: "fortythree"}
#: ⛔ PREREG §6: primary endpoints are h1 and h36. h18 is NOT an endpoint.
HORIZONS = (1, 36)


def main() -> int:
    if not _ENV_RES:
        print(
            "verify_exp02: RES_DIR is not set — refusing to guess a results directory. "
            "run_queue.sh passes it; if it is absent the queue is misconfigured."
        )
        return 1
    res = Path(_ENV_RES)
    if (
        res.resolve()
        == (_HN / "reports" / "2026-08-24_architecture_bakeoff_dossier" / "results").resolve()
    ):
        print(
            f"verify_exp02: RES_DIR points at EXP-01's results ({res}). Refusing — that "
            "directory holds a committed verdict from a different experiment."
        )
        return 1
    res.mkdir(parents=True, exist_ok=True)

    problems: list[str] = []
    rows: list[dict] = []
    for arch, tag in ARCHES.items():
        for seed, word in SEEDS.items():
            arm = f"{tag}fullzero_{word}"
            ctrl = f"{ARCHES[CONTROL_ARCH]}fullzero_{word}"
            ctrl_csv = res / f"score_{ctrl}.csv"

            #: A score CSV is what separates "this arm was scored" from "this arm is queued".
            scored = (res / f"score_{arm}.csv").exists()

            try:
                hp = _hp(arm)
            except (Exception, SystemExit) as exc:  # noqa: BLE001 - `_hp` exec()s the arm config
                # ⛔ FAIL CLOSED. `_hp` runs `exec` on the arm's own config module
                # (`verify_bakeoff.py:57`), so anything in that file can raise — including
                # `SystemExit`, which is a `BaseException` and would sail straight through the
                # `except Exception` at the bottom of this file, exiting 0 having written no
                # verdict at all. A gate that fails OPEN is the recurring defect in this
                # programme (the VOID-on-NaN gate was the fifth); this one fails closed.
                problems.append(
                    f"{arm}: its config could not be read — {type(exc).__name__}: {exc}"
                )
                continue
            if hp is None:
                # ⛔⛔ A MISSING ARM DIRECTORY IS NOT A PROBLEM WHILE THE ARM IS STILL QUEUED.
                # `run_queue.sh` builds each arm INSIDE its own loop iteration (`ensure_arm`),
                # immediately before that arm trains, and runs this verifier after EVERY arm,
                # aborting on a non-zero exit (`run_queue.sh:200-204`). `smoke_exp02.sh` builds
                # `lessons=2` labels (`ctltinyzero_*`), not the queue's `lessons=300` ones, so
                # nothing pre-builds them.
                #
                # An earlier version appended a problem here unconditionally. Measured against an
                # empty results directory it returned exit 1 with four "arm directory is missing"
                # rows. ⇒ After arm 1 the three un-built arms would each raise one, the verifier
                # would return 1, and the queue would log `ABORT — verify_exp02.py failed`:
                # ~2.5 GPU-h spent and three of four runs never started.
                #
                # ⭐ GATE F1 IS UNCHANGED. What makes an absent config incoherent rather than
                # merely early is a score CSV: that arm WAS scored, so its identity must still be
                # re-assertable, and it is not.
                if scored:
                    problems.append(
                        f"{arm}: scored, but the arm directory is gone — the identity behind "
                        f"score_{arm}.csv can no longer be re-asserted (gate F1)"
                    )
                continue
            # (1) identity, re-asserted independently of the builder's declaration — gate F1
            if hp.get("model") != arch:
                problems.append(
                    f"{arm}: model is {hp.get('model')!r}, expected {arch!r} — this arm would be "
                    "scored as an architecture it was not built with"
                )
                continue

            if scored:
                # (2) the setup audit. ⚠️ The control is its own reference, so it is audited
                # against itself only once its own CSV exists; skip the self-comparison.
                ref = None if arm == ctrl else (ctrl_csv if ctrl_csv.exists() else None)
                if ref is None and arm != ctrl:
                    problems.append(
                        f"{arm}: its own-seed control {ctrl} has no score CSV yet — the "
                        "control is queued FIRST (PREREG §10d) so this cannot happen"
                    )
                else:
                    for p in audit_arm(res, arm, reference=ref):
                        problems.append(f"{arm}: {p}")

            row = {"arch": arch, "seed": seed, "arm": arm}
            for h in HORIZONS:
                row[f"ap_h{h}"] = _ap(res / f"score_{arm}.csv", h)
                row[f"ctrl_h{h}"] = _ap(ctrl_csv, h)
            rows.append(row)

    # ⛔⛔ RECONCILE THE EXPECTED ARMS AGAINST WHAT IS ACTUALLY SCORED. `ARCHES` x `SEEDS` is a
    # hardcoded universe, `launch_exp02.sh` hardcodes the specs it queues, and NOTHING tied the two
    # together. An independent `/falsify guard` audit changed one word in the launcher —
    # `UpsampleConv2` -> `UpsampleConv1`, a registered architecture whose tag (`upcone`) is already
    # in `make_arch_arm._ARCH_TAG` — and four arms trained, scored, and were never identity-checked
    # while this verifier exited 0. Reproduced here: with `score_upconefullzero_*.csv` present, the
    # arms that ran were not flagged, not tabulated, and not MENTIONED.
    #
    # The same hole swallows a changed seed, a changed lesson count or eps (`fullzero` silently
    # encodes lessons=300 / eps=0.0 via `_LESSON_WORD` / `_EPS_WORD`), and a renamed tag.
    #
    # ⇒ Gate F1 would read as HAVING RUN when it had not — a pre-registered gate silently false,
    # for the second time in this file. ⚠️ This does NOT replace the real fix, which is for `G-F`
    # to write a sentinel naming the selected arm and for the three tools to read it instead of
    # hardcoding; that remains open. This is the backstop that notices when they disagree.
    for csv_path in sorted(res.glob("score_*.csv")):
        label = csv_path.stem[len("score_") :]
        if label.endswith("_use_real"):
            continue  # the oracle companion `run_queue.sh` scores alongside each arm
        if label not in {f"{t}fullzero_{w}" for t in ARCHES.values() for w in SEEDS.values()}:
            problems.append(
                f"{label}: scored in this results directory but is not an EXP-02 arm, so gate F1 "
                "never ran for it. Either the launcher queues something this verifier does not "
                "know about, or two experiments are sharing RES_DIR."
            )

    done = [r for r in rows if r[f"ap_h{HORIZONS[0]}"] is not None]
    expected = len(ARCHES) * len(SEEDS)
    status = "COMPLETE" if len(done) == expected else f"IN PROGRESS ({len(done)}/{expected})"

    out = [f"# EXP-02 — candidate 7 · {status}", ""]
    out.append("⛔ Endpoints are h1 and h36. h18 is NOT an endpoint — EXP-01's design error.")
    out.append("")
    hdr = ["arm", "arch", "seed"] + [f"dAP h{h}" for h in HORIZONS]
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "---|" * len(hdr))
    for r in rows:
        cells = [r["arm"], r["arch"], str(r["seed"])]
        for h in HORIZONS:
            a, c = r[f"ap_h{h}"], r[f"ctrl_h{h}"]
            cells.append("—" if a is None or c is None else f"{a - c:+.4f}")
        out.append("| " + " | ".join(cells) + " |")
    if problems:
        out += ["", "## ⛔ Problems", ""] + [f"- {p}" for p in problems]

    (res / "VERDICT_EXP02.md").write_text("\n".join(out) + "\n")
    (res / "exp02_state.json").write_text(
        json.dumps({"status": status, "rows": rows, "problems": problems}, indent=2) + "\n"
    )
    print("\n".join(out))
    return 1 if problems else 0


if __name__ == "__main__":
    # ⛔ THE DOCSTRING'S "never raises" PROMISE, with the mechanism that delivers it. The parent
    # does exactly this (`verify_bakeoff.py:154-159`) and the first version of this file copied the
    # promise while dropping the code — the same drift-between-two-copies-of-a-gate that the
    # `PREFLIGHT_OK` incident was. `sys.exit` raises `SystemExit`, which is a `BaseException` and
    # therefore passes straight through `except Exception`, so a real verdict still sets the code.
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - a crashed verifier must stop the queue, loudly
        print(
            f"verify_exp02: CRASHED -> stopping the queue: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
