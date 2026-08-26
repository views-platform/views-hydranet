"""
TDD tests for ReproducibilityGate — entropy locking and seed control.

C-42: HydraNet has no lock_entropy() — CUDA seeds uncontrolled, Python
random.seed() not called. Results silently differ across environments.

The gate should:
1. Lock all 4 RNG sources (random, numpy, torch, torch.cuda)
2. Be callable before training starts
3. Produce deterministic results across calls with the same seed
"""

import random

import numpy as np
import torch


# ---------------------------------------------------------------------------
# RED TEST 1: Module exists
# ---------------------------------------------------------------------------
def test_reproducibility_gate_importable():
    """ReproducibilityGate must exist in views_hydranet.infrastructure."""
    from views_hydranet.infrastructure.reproducibility_gate import (  # noqa: F401
        ReproducibilityGate,
    )


# ---------------------------------------------------------------------------
# RED TEST 2: lock_entropy method exists
# ---------------------------------------------------------------------------
def test_lock_entropy_exists():
    """ReproducibilityGate must have a lock_entropy static method."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    assert hasattr(ReproducibilityGate, "lock_entropy")
    assert callable(ReproducibilityGate.lock_entropy)


# ---------------------------------------------------------------------------
# RED TEST 3: lock_entropy resets Python random
# ---------------------------------------------------------------------------
def test_lock_entropy_resets_python_random():
    """After lock_entropy(42), Python's random produces deterministic values."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    ReproducibilityGate.lock_entropy(42)
    a = [random.random() for _ in range(5)]

    ReproducibilityGate.lock_entropy(42)
    b = [random.random() for _ in range(5)]

    assert a == b, "Python random not deterministic after lock_entropy"


# ---------------------------------------------------------------------------
# RED TEST 4: lock_entropy resets numpy random
# ---------------------------------------------------------------------------
def test_lock_entropy_resets_numpy_random():
    """After lock_entropy(42), numpy produces deterministic values."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    ReproducibilityGate.lock_entropy(42)
    a = np.random.rand(5).tolist()

    ReproducibilityGate.lock_entropy(42)
    b = np.random.rand(5).tolist()

    assert a == b, "NumPy random not deterministic after lock_entropy"


# ---------------------------------------------------------------------------
# RED TEST 5: lock_entropy resets torch random
# ---------------------------------------------------------------------------
def test_lock_entropy_resets_torch_random():
    """After lock_entropy(42), torch produces deterministic values."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    ReproducibilityGate.lock_entropy(42)
    a = torch.randn(5).tolist()

    ReproducibilityGate.lock_entropy(42)
    b = torch.randn(5).tolist()

    assert a == b, "PyTorch random not deterministic after lock_entropy"


# ---------------------------------------------------------------------------
# RED TEST 6: training_engine uses lock_entropy instead of manual seeds
# ---------------------------------------------------------------------------
def test_training_engine_uses_lock_entropy():
    """
    training_engine.training_loop() must call lock_entropy rather than
    setting seeds manually. Verify by checking source code — the manual
    np.random.seed / torch.manual_seed calls should be replaced.
    """
    import importlib
    from pathlib import Path

    # This assertion is over the module's SOURCE TEXT, so the module object is irrelevant.
    # It used to `del sys.modules[...]` first and re-import, which replaced the live
    # training_engine module for the rest of the session: `train_model.training_loop` kept
    # referring to the OLD module's globals, so any later test that patched a training_engine
    # attribute was silently patching an object nothing called. Three tests in
    # tests/train/test_backward_gate.py failed only in a full-suite run because of it.
    mod = importlib.import_module("views_hydranet.train.training_engine")
    source = Path(mod.__file__).read_text()

    # Should NOT have manual seed calls
    assert "np.random.seed(" not in source, (
        "training_engine.py still uses np.random.seed() instead of lock_entropy()"
    )
    assert "torch.manual_seed(" not in source, (
        "training_engine.py still uses torch.manual_seed() instead of lock_entropy()"
    )

    # Should import and use lock_entropy
    assert "lock_entropy" in source, "training_engine.py does not reference lock_entropy"


def test_the_live_training_engine_module_is_the_one_training_loop_uses():
    """Guard on the fix above: no test may leave a stale training_engine in sys.modules.

    ``test_training_engine_uses_lock_entropy`` used to ``del sys.modules[...]`` before
    re-importing. That swapped the module object for the rest of the session while
    ``train_model.training_loop`` — already bound — kept using the OLD module's globals. Any later
    test that patched a ``training_engine`` attribute was then patching an object nothing called,
    and three tests in ``tests/train/test_backward_gate.py`` failed **only** in full-suite runs.

    Those tests are now written against ``training_loop.__globals__`` and so are immune, which is
    exactly why this guard is needed: with them hardened, re-introducing the delete breaks nothing
    and nothing notices. This asserts the invariant directly.
    """
    import sys

    from views_hydranet.train.train_model import training_loop

    live = sys.modules["views_hydranet.train.training_engine"]
    assert training_loop.__globals__ is live.__dict__, (
        "training_loop is running against a DIFFERENT training_engine module object than the one "
        "in sys.modules. Something deleted or reloaded the module without rebinding its "
        "importers; monkeypatching training_engine attributes will silently do nothing."
    )


# ---------------------------------------------------------------------------
# RED TEST 7: Different seeds produce different results
# ---------------------------------------------------------------------------
def test_different_seeds_produce_different_results():
    """Sanity check: different seeds must produce different random sequences."""
    from views_hydranet.infrastructure.reproducibility_gate import (
        ReproducibilityGate,
    )

    ReproducibilityGate.lock_entropy(42)
    a = torch.randn(10).tolist()

    ReproducibilityGate.lock_entropy(99)
    b = torch.randn(10).tolist()

    assert a != b, "Different seeds produced identical results"
