"""Silence-vs-fade (2026-09-02): the un-composed body-mean dump and its parity guarantee.

The question "does the free-running field lose CELLS or lose SIZE" needs the body mean
``E[Y|body]`` **separately from** the gate ``P(y>0)``, because every composed readout conflates
them. The cube cannot supply it: ``compose_samples`` applies a per-draw ``Bernoulli(gate)``
mask to family DRAWS, so ``expm1(cube)/gate`` is unbiased for ``mu`` but its variance is
inflated by ``1/gate`` — unusable exactly in the collapsed-gate regime the question is about.

The dump therefore writes both factors raw. These tests assert the two properties a diagnostic must
have before it may be believed:

1. **It cannot perturb what it measures** — a run with the dump on produces a byte-identical
   cube to the same run with it off (blocking gate C.2 of the pre-flight checklist).
2. **It is the body actually emitted** — ``gate * mu_dumped`` reproduces
   ``expm1(_emit_magnitude())`` exactly under ``soft_gate``, so the dumped field is not merely
   *a* mean but *the* one composed into the forecast.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from views_hydranet.distributions import resolve_family  # noqa: E402

from .test_sampler_dxk import (  # noqa: E402
    _FEATURES,
    _activated_params,
    _mock_handler,
)


def _make_inference(tmp_dir, *, composition="soft_gate", dist="nb", d=2, k=2):
    """A tiny real-model inference, optionally dumping the body mean to ``tmp_dir``."""
    from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
    from views_hydranet.utils.hydranet_inference import HydraNetInference

    torch.manual_seed(0)
    model = HydraBNUNet06_LSTM4(3, 16, 1, 0.0, output_distribution=dist).float()
    config = {
        "steps": [1, 2],
        "time_steps": 2,
        "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
        "classification_targets": ["by_sb", "by_ns", "by_os"],
        "features": ["lr_sb", "lr_ns", "lr_os"],
        "static_channels": [],
        "n_posterior_samples": d,
        "n_head_samples": k,
        "np_seed": 42,
        "torch_seed": 1234,
        "forecast_composition": composition,
    }
    return HydraNetInference(
        model,
        config,
        device="cpu",
        body_mean_dump_dir=(str(tmp_dir) if tmp_dir is not None else None),
    )


def test_dump_defaults_off_and_writes_nothing(tmp_path):
    """The production path is unchanged unless a caller explicitly opts in."""
    inf = _make_inference(None)
    assert inf.body_mean_dump_dir is None
    inf.generate_posterior_samples(_mock_handler(_FEATURES), origin=1)
    assert not list(tmp_path.iterdir())


def test_dump_on_is_byte_identical_to_dump_off(tmp_path):
    """BLOCKING (dossier 03 C.2): the instrument must not perturb the run it measures.

    An earlier diagnostic in this repo ran an extra forward in ``train()`` mode and silently wrote
    BatchNorm running stats, which would have confounded an A/B at the BN layer. This asserts the
    class of failure cannot recur here: same seed, same handler, cube byte-identical.
    """
    handler = _mock_handler(_FEATURES)
    mag_off, prob_off = _make_inference(None).generate_posterior_samples(handler, origin=1)
    mag_on, prob_on = _make_inference(tmp_path).generate_posterior_samples(handler, origin=1)

    np.testing.assert_array_equal(mag_on, mag_off)
    np.testing.assert_array_equal(prob_on, prob_off)
    assert list(tmp_path.glob("bodymean_origin*.npz")), "the dump ran but wrote nothing"


def test_dumped_mu_is_the_body_that_was_emitted(tmp_path):
    """``gate * mu_dumped`` must reproduce the emitted composed mean exactly.

    Guards the failure that would make the whole experiment meaningless: dumping *a* mean that is
    not *the* mean composed into the forecast (e.g. missing the ADR-068 core switch, or reading the
    wrong param slice). Under ``soft_gate``, ``compose_mean`` is ``gate * mean``, so the
    emitted field is ``log1p(gate * mu)`` and the identity is exact, not approximate.
    """
    inf = _make_inference(tmp_path, composition="soft_gate")
    fam = resolve_family("nb")
    t, h, w, n_reg = 2, 4, 4, 3
    params = _activated_params(fam, t, h, w, n_reg, seed=7)
    gate = torch.rand(t, h, w, n_reg)

    inf._dump_body_mean(params, gate, origin=5)
    loaded = np.load(tmp_path / "bodymean_origin5.npz")
    mu = torch.as_tensor(loaded["mu"])  # [T, n_reg, H, W]

    emitted = inf._emit_magnitude(params, gate.permute(0, 3, 1, 2))  # log1p(gate * mu)
    composed_from_dump = gate.permute(0, 3, 1, 2) * mu

    torch.testing.assert_close(torch.expm1(emitted), composed_from_dump, rtol=1e-6, atol=1e-6)


def test_dumped_gate_is_stored_verbatim(tmp_path):
    """Occurrence is read straight off the gate field: it must round-trip exactly."""
    inf = _make_inference(tmp_path)
    fam = resolve_family("nb")
    params = _activated_params(fam, 2, 4, 4, 3, seed=3)
    gate = torch.rand(2, 4, 4, 3)

    inf._dump_body_mean(params, gate, origin=0)
    loaded = np.load(tmp_path / "bodymean_origin0.npz")

    np.testing.assert_array_equal(loaded["gate"], gate.numpy())
    assert int(loaded["origin"]) == 0
    assert int(loaded["n_reg"]) == 3


def test_dump_without_a_family_fails_loud(tmp_path):
    """A silent no-op would make an experiment report 'no data' when it never ran."""
    from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
    from views_hydranet.utils.hydranet_inference import HydraNetInference

    model = HydraBNUNet06_LSTM4(3, 16, 1, 0.0, output_distribution="hurdle_nb").float()
    config = {
        "steps": [1],
        "time_steps": 1,
        "regression_targets": ["lr_sb"],
        "classification_targets": ["by_sb"],
        "features": ["lr_sb"],
        "static_channels": [],
        "n_posterior_samples": 1,
        "np_seed": 42,
        "torch_seed": 1,
        "hurdle_theta": 1.0,
    }
    with pytest.raises(ValueError, match="needs a registered distribution family"):
        HydraNetInference(model, config, device="cpu", body_mean_dump_dir=str(tmp_path))
