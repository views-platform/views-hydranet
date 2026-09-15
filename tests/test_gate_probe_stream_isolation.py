"""S7/#360: the gate-structure probe must not draw from the stream it observes.

`hydranet_inference.py` keeps deliberately separate RNG streams so an intervention can never be
drawn from the stream it perturbs — the file legislates this at `_FB_TRANSFORM_SEED_NAMESPACE` and
`_FB_CORRELATED_SEED_NAMESPACE`, citing the C-113 shared-generator coupling.

The probe had no namespace of its own. It drew from `_fb_transform_gen`, the generator `thin`,
`inject`, `magnitude_perturb` and both splices consume, and the two interleave within a single
rollout step. So switching the observer on changed the treatment it was observing, and — because
the length-scale sweep runs on posterior sample 0 only — by a *different* amount on sample 0 than
on the rest.

The decisive test is byte identity, not a distributional check: a distributional check passes on a
broken fix.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.utils.hydranet_inference import (  # noqa: E402
    _FB_CORRELATED_SEED_NAMESPACE,
    _FB_GATE_PROBE_SEED_NAMESPACE,
    _FB_TRANSFORM_SEED_NAMESPACE,
    HydraNetInference,
)

SEED = 4242
N_STEPS = 3


class _Rollout:
    """The interleave the real rollout performs: transform the fed field, then record the probe.

    Both methods are the production ones, unbound — a reimplementation here would test this file
    rather than the model.
    """

    _month_shuffle: dict[int, int] = {}
    _scramble_perm = None

    def __init__(self, *, arm, record_probe: bool, sample_idx: int):
        self._feedback_arm = arm
        self._record_gate_probe = record_probe
        self.config = {"regression_targets": ["sb", "ns"], "torch_seed": SEED}
        self.gate_structure_stats: list[dict] = []
        self.diagnostic_stats_dropped: dict[str, int] = {}
        self.sample_idx = sample_idx
        # Seeded exactly as `_forecast` seeds them, so the streams here are the production streams.
        self._fb_transform_gen = torch.Generator(device="cpu").manual_seed(
            SEED + _FB_TRANSFORM_SEED_NAMESPACE + sample_idx
        )
        self._fb_gate_probe_gen = (
            torch.Generator(device="cpu").manual_seed(
                SEED + _FB_GATE_PROBE_SEED_NAMESPACE + sample_idx
            )
            if record_probe
            else None
        )

    def _real_dynamic(self, full_tensor, model_in_indices, n_dyn, step):
        return HydraNetInference._real_dynamic(self, full_tensor, model_in_indices, n_dyn, step)

    def _refuse_if_diagnostic_buffer_full(self, buffer, *, label):
        return HydraNetInference._refuse_if_diagnostic_buffer_full(self, buffer, label=label)

    def _append_diagnostic_stat(self, buffer, record, *, label):
        return HydraNetInference._append_diagnostic_stat(self, buffer, record, label=label)

    def run(self, full, idx, t0, gate):
        """Returns the transformed field at every step — the thing an arm actually feeds back."""
        fed = []
        for step in range(1, N_STEPS + 1):
            # `if self._feedback_arm:` mirrors the rollout's own guard at hydranet_inference:1218 —
            # a probe-only run (no arm) transforms nothing.
            if self._feedback_arm:
                fed.append(
                    HydraNetInference._apply_feedback_transform(
                        self, t0, full, idx, 1, step=step, origin=0
                    ).clone()
                )
            if self._record_gate_probe:
                HydraNetInference._record_gate_structure(
                    self, gate, origin=0, sample_idx=self.sample_idx, step=step
                )
        return fed


def _fixture():
    """A SPARSE field, because the real one is ~99.94% exactly zero.

    A dense fixture silently disarms half these arms: `inject` only touches cells that are zero, so
    on a dense field it is the identity and its parametrisation proves nothing. The first draft of
    this file made exactly that mistake.
    """
    torch.manual_seed(0)
    B, M, C, H, W = 1, 8, 3, 12, 12
    full = torch.rand(B, M, C, H, W) * 4.0
    full[torch.rand_like(full) > 0.08] = 0.0
    idx = [0, 1]
    t0 = torch.rand(B, 3, H, W)
    t0[torch.rand_like(t0) > 0.08] = 0.0
    gate = torch.rand(B, 2, H, W)
    return full, idx, t0, gate


def test_the_fixture_is_sparse_enough_for_inject_to_do_anything():
    """Guards the parametrisation above: `inject` activates zero cells, so if the fixture has none
    the `inject` arm is a decorative test case that cannot fail."""
    full, idx, t0, _ = _fixture()
    real = torch.expm1(full[:, 1, :2]).clamp(min=0.0)
    assert (real == 0).any(), "no zero cells: the inject arm would be the identity"
    assert (real > 0).any(), "no active cells: inject has no pool to draw values from"


class TestTheNamespacesAreDistinct:
    """The acceptance criterion is 'asserted, not asserted-in-a-comment'."""

    def test_the_four_feedback_streams_have_their_own_namespace(self):
        """Scope: the four streams the FEEDBACK path draws from. `arm_gen` (seeded `torch_seed`
        alone) and the cube sampler's generator are outside it and not asserted here — on
        posterior sample 0 they share a seed with `fb_gen`, a pre-existing overlap (#372)."""
        namespaces = {
            "fb_gen (family.sample / compose_samples)": 0,
            "transform": _FB_TRANSFORM_SEED_NAMESPACE,
            "correlated": _FB_CORRELATED_SEED_NAMESPACE,
            "gate probe": _FB_GATE_PROBE_SEED_NAMESPACE,
        }
        assert len(set(namespaces.values())) == len(namespaces), (
            f"two RNG streams share a seed namespace, so they emit the same uniforms: {namespaces}"
        )

    def test_the_probe_generator_is_not_the_transform_generator(self):
        r = _Rollout(arm=("thin", 0.25), record_probe=True, sample_idx=0)
        assert r._fb_gate_probe_gen is not r._fb_transform_gen
        assert r._fb_gate_probe_gen.initial_seed() != r._fb_transform_gen.initial_seed()


@pytest.mark.parametrize("sample_idx", [0, 1])
@pytest.mark.parametrize(
    "arm",
    [("thin", 0.25), ("inject", 0.05), ("magnitude_perturb", 0.5)],
    ids=lambda a: a[0],
)
class TestTheObserverDoesNotPerturbTheTreatment:
    """`sample_idx` is parametrised because the length-scale sweep runs on sample 0 only, so the
    two desynchronise the shared stream by different amounts."""

    def test_the_fed_field_is_byte_identical_with_the_probe_on_and_off(self, arm, sample_idx):
        full, idx, t0, gate = _fixture()
        off = _Rollout(arm=arm, record_probe=False, sample_idx=sample_idx).run(full, idx, t0, gate)
        on = _Rollout(arm=arm, record_probe=True, sample_idx=sample_idx).run(full, idx, t0, gate)
        for step, (a, b) in enumerate(zip(off, on), start=1):
            assert torch.equal(a, b), (
                f"arm={arm[0]} sample_idx={sample_idx} step={step}: enabling the gate probe "
                f"changed the field the arm feeds back "
                f"({int((a != b).sum())} of {a.numel()} values differ). The probe is drawing from "
                "the stream it observes (S7/#360)."
            )

    def test_the_probe_recorded_something_to_be_worth_testing(self, arm, sample_idx):
        """Guards the test above against passing because the probe never ran."""
        full, idx, t0, gate = _fixture()
        r = _Rollout(arm=arm, record_probe=True, sample_idx=sample_idx)
        r.run(full, idx, t0, gate)
        assert len(r.gate_structure_stats) == N_STEPS * 2, (
            f"expected {N_STEPS * 2} probe records (steps x targets), got "
            f"{len(r.gate_structure_stats)}"
        )


class TestTheProbeIsReproducibleIndependentlyOfTheArm:
    """The probe's own numbers must not depend on which treatment is running beside it."""

    @pytest.mark.parametrize("sample_idx", [0, 1])
    def test_the_same_seed_gives_the_same_probe_under_different_arms(self, sample_idx):
        full, idx, t0, gate = _fixture()
        records = {}
        for arm in [("thin", 0.25), ("inject", 0.05), None]:
            r = _Rollout(arm=arm, record_probe=True, sample_idx=sample_idx)
            r.run(full, idx, t0, gate)
            records[arm] = r.gate_structure_stats
        reference = records[("thin", 0.25)]
        for arm, recs in records.items():
            assert recs == reference, (
                f"the probe's output changed when the arm changed to {arm} — it is still coupled "
                "to the treatment stream (S7/#360)"
            )
