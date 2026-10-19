"""PRE-FLIGHT for the pos_weight arm — does the config key reach the BUILT loss?

The denominator arm's F1 was a loss-MAGNITUDE assertion. This arm changes no reduction, so that
gate is silent here and would pass VACUOUSLY (prereg section 6a). Nothing in views-models checks
that the key reached the loss either — its pin-floor guard checks only that the pinned library
COULD honour it. So this is the only check between the config and a full-length run of the
control, and on #398 exactly this class of gap shipped a dead feature.

No GPU, no spend.
"""
import importlib.util
import sys
from pathlib import Path

import torch

MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
ARM, CONTROL = "copper_kestrel", "amber_kestrel"
EXPECTED_DIFF = {"loss_class_pos_weight", "trajectory_log_path"}


def load_hp(name: str) -> dict:
    path = MODELS / name / "configs" / "config_hyperparameters.py"
    spec = importlib.util.spec_from_file_location(f"hp_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_hp_config()


print("=" * 86)
print("PRE-FLIGHT — the pos_weight arm")
print("=" * 86)
arm, ctrl = load_hp(ARM), load_hp(CONTROL)

# ── 1. the config diff is exactly the one variable (plus the log path) ──
keys = set(arm) | set(ctrl)
diff = {k for k in keys if arm.get(k) != ctrl.get(k)}
print(f"\n1. CONFIG DIFF  {ARM} vs {CONTROL}")
for k in sorted(diff):
    print(f"   {k}: {ctrl.get(k)!r} -> {arm.get(k)!r}")
extra = diff - EXPECTED_DIFF
if extra:
    print(f"\n⛔ VOID — unexpected differences: {sorted(extra)}")
    sys.exit(1)
missing = {"loss_class_pos_weight"} - diff
if missing:
    print("\n⛔ VOID — loss_class_pos_weight does NOT differ. This arm is the control.")
    sys.exit(1)
print("   OK — exactly the pre-registered variable (+ the log path)")

# ── 2. the prereg's held-constant list really is held constant ──
HELD = {
    "output_distribution": "truncated_nb",
    "forecast_composition": "soft_gate",
    "freeze_multitask_balancer": True,
    "body_supervision": "all",
    "total_lessons": 300,
    "windows_per_lesson": 3,
    "window_dim": 32,
    "sampling_strategy": "sigmoid",
    "min_events": 5,
}
print("\n2. HELD CONSTANT (prereg section 2)")
bad = []
for k, v in HELD.items():
    got = arm.get(k)
    ok = got == v
    print(f"   {'OK ' if ok else '⛔ '} {k}: {got!r}" + ("" if ok else f"  expected {v!r}"))
    if not ok:
        bad.append(k)
den = arm.get("body_nll_denominator")
print(f"   {'OK ' if den in (None, 'positives') else '⛔ '} body_nll_denominator: {den!r} "
      f"(must be absent/positives — the denominator arm must not ride along)")
if den not in (None, "positives"):
    bad.append("body_nll_denominator")
if bad:
    print(f"\n⛔ VOID — held-constant values moved: {bad}")
    sys.exit(1)

# ── 3. ⛔ THE GATE: does the key reach the BUILT loss? ──
print("\n3. DOES THE KEY REACH THE BUILT LOSS?  (the only check that catches a dead config key)")
from views_hydranet.utils.utils import choose_loss  # noqa: E402

dev = torch.device("cpu")
got = {}
for name, cfg in ((ARM, arm), (CONTROL, ctrl)):
    _, crit_cls, _ = choose_loss(cfg, dev)
    w = getattr(crit_cls, "pos_weight_value", None)
    got[name] = w
    print(f"   {name:<16} config says {cfg.get('loss_class_pos_weight')!r:>5}"
          f"   ->  built loss carries {w!r}   ({type(crit_cls).__name__})")

if got[ARM] is None:
    print("\n⛔ VOID — the built criterion exposes no pos_weight_value; cannot verify.")
    sys.exit(1)
if float(got[ARM]) != 1.0:
    print(f"\n⛔ VOID — the arm's built loss carries {got[ARM]}, not 1.0 —"
          " the key did NOT reach it.")
    sys.exit(1)
if float(got[CONTROL]) != 2.0:
    print(f"\n⛔ VOID — the control's built loss carries {got[CONTROL]}, not 2.0.")
    sys.exit(1)
if float(got[ARM]) == float(got[CONTROL]):
    print("\n⛔ VOID — arm and control build the SAME pos_weight. The arm is the control.")
    sys.exit(1)

print("\n" + "=" * 86)
print("VERDICT: ✅ CLEARED. The key reaches the built loss (1.0 vs the control's 2.0),")
print("         the diff is")
print("         exactly the pre-registered variable, and every held-constant value holds.")
print("=" * 86)
