"""Read the gate convention FROM THE ARM CONFIGS and assert it. ⛔ Never transcribe it.

Epic #579, story S4 #583. Pre-registration: ``../PREREG.md``, committed before this file existed.

⛔⛔ WHY THIS FILE EXISTS. ``tau = 0.20`` was a module constant at
``reports/2026-11-04_rank_coupled_body/tools/sim_rank_coupling.py:79`` and a sentence in
``reports/2026-11-04_delivered_gate_lookup/tools/e1_data.py:26``'s docstring. **Nothing read it
from the thing that produced the cubes.** A transcribed threshold that silently disagreed with the
arm would make every open-set-dependent number in the epic wrong, with no error signal -- and the
whole epic rests on A and B being scored on the SAME open set.

⭐ MEASURED 2026-11-10: the transcription was CORRECT -- all eleven delivered arms carry
``gate_threshold = 0.20`` and ``forecast_composition = 'threshold_gate'``. ⚠️ It was correct and
**unverified**, which are different things, and the second is what this file fixes.

⚠️ WHAT THE CONFIG SAYS ABOUT tau, quoted because it bears on the epic's scope
(``calm_orbiter/configs/config_hyperparameters.py:46-49``):

    "#465: 0.5 fired on 3-6x too few cells (views-hydranet M75). Measured calibrated tau for this
     model was 0.26; set BELOW it on purpose -- the platform undershoots fatalities even at tau=0,
     so lean toward firing more. A prior, not a measurement: to be swept."

⇒ tau is an **unswept prior**, and it is deliberately below this model's measured calibrated value.
⛔ Sweeping it is NOT in this epic's scope (``PREREG.md`` §10); this note exists so that no number
here is read as though tau were optimised.

⭐⭐ THE DECISION THIS FILE RECORDS, and it is S4's whole point. A ships ``threshold_gate``: a cell
with ``gate >= tau`` keeps its body draws and a cell below is zeroed. **The composition is
DETERMINISTIC given the gate -- there is no coin.** The 2026-11-04 simulation instead drew
``rng.random(n_cells) < pi`` (``sim_rank_coupling.py:138,141``), a Bernoulli gate, while using tau
only to define the REAL arm's ordering set (``:185``). ⇒ "same gate, one varying thing" was **not
true**: the firing MECHANISM differed between A and B.

⇒ **B fires deterministically on ``gate >= tau``, exactly as A does.** Consequences, all of them
wanted:

  · A and B are scored on an IDENTICAL open set, by construction rather than by assertion.
    ``PREREG.md`` V2 (different cells / n / S) becomes structurally unsatisfiable.
  · B's open set does not vary across samples, so the gate contributes **zero** per-cell variance.
    The only randomness left in B is the empirical draw -- which is the thing under test.
  · ⚠️ The 2026-11-04 Bernoulli figures are therefore **not** comparable to this epic's, and are
    not quoted as though they were. The committed REAL gap of -0.0137 is a Bernoulli-gate number.

Run: ``conda run --no-capture-output -n views-hydranet-env python <this file>``
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "2026-11-04_delivered_gate_lookup/tools"))

import e1_data  # noqa: E402

# ⭐ Every delivered arm, control form and arm form both. The gate convention must agree across ALL
# of them, not merely on the one arm this epic scores: sigma-hat is measured across the controls
# (``PREREG.md`` §1), so a disagreement anywhere makes the margin itself ill-defined.
# ⭐ PRE-REGISTERED names, not directory names. ``exp03_arms`` explains why: views-models renamed
# every arm on 2026-11-03, and the pre-registered spelling is what ``PREREG.md`` and
# ``PREDICTION_LEDGER.md`` use. The directory is RESOLVED, never spelled.
CONTROL_ARMS = (
    "ctlfullzero_fortytwo",
    "ctlfullzero_fortythree",
    "ctlfullzero_fortyfour",
    "ctlfullzero_fortyfive",
    "ctlfullzero_fortysix",
    "ctlfullzero_fortyseven",
)
ARM_FORM_ARMS = (
    "upctwofullzero_fortytwo",
    "upctwofullzero_fortythree",
    "upctwofullzero_fortyfour",
    "upctwofullzero_fortyfive",
    "upctwofullzero_fortyseven",
)
# ⚠️ ``upctwofullzero_fortysix`` (``quiet_lander``) is a ghost: renamed but never run, so it carries
# no ``gate_endpoints.tar.zst`` and no ``*_lr.tar.zst``. ``arm_directories()`` omits it entirely.
# Verified on disk 2026-11-10. It removes an ARM-form seed, never a control one, which is why
# sigma-hat has n = 6 and not n = 5.
GHOST_ARMS = ("upctwofullzero_fortysix",)

EXPECTED_COMPOSITION = "threshold_gate"


def read_arm_gate_convention(arm: str) -> tuple[float, str]:
    """-> (gate_threshold, forecast_composition) read from the arm's OWN config module.

    ⛔ Imported and called, not regex-scraped: a value assembled at call time by the config would
    be invisible to a text search, and a text search is exactly the failure this file removes.

    ⭐ ``arm`` is the PRE-REGISTERED name; the directory comes from ``exp03_arms.arm_directories()``,
    which derives it from the delivery's two manifests and refuses on any ambiguity.
    """
    cfg = e1_data.arm_dir(arm, arms=e1_data.arm_directories()) / "configs/config_hyperparameters.py"
    if not cfg.exists():
        raise FileNotFoundError(f"arm_config: no config for {arm!r} at {cfg}")
    spec = importlib.util.spec_from_file_location(f"_cfg_{arm}", cfg)
    if spec is None or spec.loader is None:
        raise ImportError(f"arm_config: cannot load {cfg}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hp = mod.get_hp_config()
    missing = [k for k in ("gate_threshold", "forecast_composition") if k not in hp]
    if missing:
        raise KeyError(f"arm_config: {arm} config is missing {missing}")
    return float(hp["gate_threshold"]), str(hp["forecast_composition"])


def resolve_tau(arms: tuple[str, ...] = CONTROL_ARMS + ARM_FORM_ARMS) -> float:
    """-> the single tau every arm agrees on. ⛔ Raises if they disagree; never picks a winner.

    ⭐ G-T1. A disagreement here is not a value to reconcile -- it means the arms do not share an
    open-set definition, so no paired A-vs-B number over them is meaningful. The right response is
    to stop, which is why this raises instead of returning a majority or a default.
    """
    seen: dict[tuple[float, str], list[str]] = {}
    for arm in arms:
        seen.setdefault(read_arm_gate_convention(arm), []).append(arm)
    if len(seen) != 1:
        detail = "; ".join(f"(tau={t}, composition={c!r}) <- {a}" for (t, c), a in seen.items())
        raise ValueError(f"arm_config: G-T1 FAILED, arms disagree on the gate convention: {detail}")
    (tau, composition), _ = next(iter(seen.items()))
    if composition != EXPECTED_COMPOSITION:
        raise ValueError(
            f"arm_config: G-T1 FAILED, composition is {composition!r}, not "
            f"{EXPECTED_COMPOSITION!r}. ⛔ The deterministic open set this epic relies on is a "
            "property of threshold_gate; under another composition the whole design changes."
        )
    return tau


def open_set(gate: "object", tau: float):
    """-> the boolean open set A actually ships: ``gate >= tau``, deterministic, no coin.

    ⭐ The one-line replacement for ``rng.random(n_cells) < pi``. It is a function rather than an
    inline expression so that B cannot acquire a second firing convention by copy-paste.
    """
    import numpy as np

    return np.asarray(gate) >= tau


def main() -> None:
    print(__doc__)
    print("=" * 100)
    print("S4 · THE GATE CONVENTION, READ FROM THE ARM CONFIGS")
    print("=" * 100)

    dirs = e1_data.arm_directories()
    print(f"  {'pre-registered arm':<28}{'directory':<17}{'form':<9}{'tau':>8}{'composition':>18}")
    for arm in CONTROL_ARMS + ARM_FORM_ARMS:
        tau, comp = read_arm_gate_convention(arm)
        form = "control" if arm in CONTROL_ARMS else "arm"
        print(f"  {arm:<28}{dirs[arm].name:<17}{form:<9}{tau:>8.4f}{comp:>18}")
    for arm in GHOST_ARMS:
        note = "absent from arm_directories() — never ran"
        print(f"  {arm:<28}{'(ghost)':<17}{'ghost':<9}{note:>26}")

    tau = resolve_tau()
    print(f"\n  PASS  G-T1  all {len(CONTROL_ARMS + ARM_FORM_ARMS)} arms agree: ")
    print(f"              tau = {tau}, forecast_composition = {EXPECTED_COMPOSITION!r}")
    print(f"  ⭐ tau is now READ, not transcribed. The transcribed value was {tau} and it was right.")
    print("  ⚠️ tau is an UNSWEPT PRIOR, deliberately below this model's measured calibrated 0.26.")
    print("  ⭐ B fires on gate >= tau, deterministically — the same open set A ships. No coin.")
    print(f"  ⚠️ sigma-hat is therefore over n = {len(CONTROL_ARMS)} control-form arms, not 5.")


if __name__ == "__main__":
    main()
