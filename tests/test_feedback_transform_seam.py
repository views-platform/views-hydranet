"""The feedback-transform seam: the three guards that make the dose-response programme valid.

These are falsifiers F1-F3 of `reports/2026-08-16_feedback_realism_dossier/05_analysis_plan.md`,
implemented as tests rather than as run-time checks so they run on every commit:

* **F1** — ``feedback_transform='use_real'`` must feed **byte-identically** what
  ``rollout_feedback='teacher_forced'`` feeds. If it does not, the transform is reading the wrong
  month or the wrong channels, and every arm built on it is confidently wrong while looking
  healthy.
* **F2** — no transform may touch the **static** channels. They are geometry-constant
  (ADR-060 I3); perturbing them would confound every axis with "the map moved".
* **F3** — ``feedback_transform=None`` must be byte-identical to ``'identity'``, i.e. the seam
  must not perturb the production path (notably not the RNG streams).

The model here is a **recording mock**: it stores exactly what it was fed at every step, so the
guards are checked against the real inputs rather than inferred from outputs.
"""

import pytest
import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput
from views_hydranet.utils.hydranet_inference import HydraNetInference


class _RecordingModel(nn.Module):
    """Echoes the dynamic channels and records every input it is given."""

    def __init__(self, n_reg=1, base=32):
        super().__init__()
        self.base = base
        self.n_reg = n_reg
        self.seen: list[torch.Tensor] = []

    def init_hTtime(self, hidden_channels, H, W):
        return torch.zeros(1, hidden_channels, H, W)

    def forward(self, x, h):
        self.seen.append(x.clone())
        reg = x[:, : self.n_reg].clone()
        return ModelOutput(reg=reg, cls=torch.zeros_like(reg), h_next=h + 0.01)


def _cfg(*, statics=False):
    cfg = {
        "steps": list(range(1, 6)),
        "time_steps": 5,
        "features": ["feat"],
        "regression_targets": ["feat"],
        "classification_targets": ["class"],
        "sampling_strategy": "threshold",
        "n_posterior_samples": 1,
        "diagnostic_visualizations": False,
        "torch_seed": 7,
        # required once a feedback arm is set: the transforms round-trip with
        # expm1/log1p, so every dynamic feature must be in log1p space.
        "transformations": {"log1p": ["feat"], "asinh": [], "identity": []},
    }
    if statics:
        cfg["static_channels"] = ["stat"]
    return cfg


def _tensor(*, statics=False, n_months=9, h=6, w=6):
    """A volume in log1p space. Dynamic varies per month; the static is time-CONSTANT."""
    n_ch = 2 if statics else 1
    vol = torch.zeros(1, n_months, n_ch, h, w)
    for m in range(n_months):
        vol[0, m, 0, m % h, (m * 2) % w] = torch.log1p(torch.tensor(float(m + 1)))
        vol[0, m, 0, (m + 1) % h, (m * 3) % w] = torch.log1p(torch.tensor(float(m + 5)))
    if statics:
        vol[0, :, 1] = 0.5  # geometry-constant across time, as ADR-060 I3 requires
    return vol


def _run(*, feedback_transform=None, rollout_feedback="mean", statics=False, origin=3):
    cfg = _cfg(statics=statics)
    cfg["rollout_feedback"] = rollout_feedback
    model = _RecordingModel()
    inf = HydraNetInference(model, cfg, device="cpu", feedback_transform=feedback_transform)
    names = ["feat", "stat"] if statics else ["feat"]
    mags, _ = inf.predict(_tensor(statics=statics), origin, 0, names)
    return model.seen, mags, inf


# ------------------------------------------------------------------ F1


@pytest.mark.parametrize("statics", [False, True])
def test_F1_use_real_feeds_exactly_what_teacher_forced_feeds(statics):
    """The self-test the whole programme rests on. A wrong month here poisons every arm."""
    oracle, _, _ = _run(rollout_feedback="teacher_forced", statics=statics)
    viaxform, _, _ = _run(rollout_feedback="mean", feedback_transform="use_real", statics=statics)
    assert len(oracle) == len(viaxform)
    for step, (a, b) in enumerate(zip(oracle, viaxform)):
        assert torch.equal(a, b), (
            f"step {step}: use_real fed a different field than teacher_forced"
        )


def test_F1_fails_when_the_month_is_off_by_one():
    """The guard must be able to fail — `wrong_month:1` is a deliberate one-month error."""
    oracle, _, _ = _run(rollout_feedback="teacher_forced")
    shifted, _, _ = _run(rollout_feedback="mean", feedback_transform="wrong_month:1")
    assert any(not torch.equal(a, b) for a, b in zip(oracle, shifted)), (
        "an off-by-one month produced identical inputs — F1 cannot detect a month error"
    )


# ------------------------------------------------------------------ F2


@pytest.mark.parametrize(
    "spec",
    [
        "identity",
        "use_real",
        "thin:1.0",
        "inject:0.5",
        "magnitude_perturb:1.0",
        "spatial_scramble",
    ],
)
def test_F2_no_transform_touches_the_static_channels(spec):
    """Statics are geometry-constant; a transform that moved them would confound every axis."""
    baseline, _, _ = _run(feedback_transform="identity", statics=True)
    seen, _, _ = _run(feedback_transform=spec, statics=True)
    for step, (a, b) in enumerate(zip(baseline, seen)):
        assert torch.equal(a[:, 1:], b[:, 1:]), f"step {step}: {spec} altered a static channel"


def test_F2_fails_if_a_transform_writes_into_the_static_prefix():
    """A hand-built violation must be caught, or F2 is decorative."""
    baseline, _, _ = _run(feedback_transform="identity", statics=True)
    tampered = [s.clone() for s in baseline]
    tampered[-1][:, 1:] += 1.0
    assert any(not torch.equal(a[:, 1:], b[:, 1:]) for a, b in zip(baseline, tampered))


# ------------------------------------------------------------------ F3


def test_F3_none_is_byte_identical_to_identity():
    """The seam must not perturb production — including not disturbing any RNG stream."""
    off, mags_off, inf_off = _run(feedback_transform=None)
    ident, mags_ident, _ = _run(feedback_transform="identity")
    assert inf_off.feedback_transform is None
    for step, (a, b) in enumerate(zip(off, ident)):
        assert torch.equal(a, b), f"step {step}: the seam perturbed the default path"
    assert torch.equal(torch.as_tensor(mags_off), torch.as_tensor(mags_ident))


def test_F3_the_default_records_no_statistics():
    """No arm set => no observer overhead and no accidental state on the production path."""
    _, _, inf = _run(feedback_transform=None)
    assert inf.feedback_field_stats == []


# ------------------------------------------------- the transforms actually bite on a rollout


def test_thin_at_p_one_feeds_an_empty_field():
    seen, _, _ = _run(feedback_transform="thin:1.0")
    fed = [s[:, :1] for s in seen[4:]]  # free-running steps only (origin=3 => t>3)
    assert fed and all(float(f.abs().sum()) == 0.0 for f in fed)


def test_shuffle_months_feeds_real_fields_in_a_different_order():
    """Every field stays real and geo-located; only the temporal order moves."""
    oracle, _, _ = _run(rollout_feedback="teacher_forced")
    shuffled, _, _ = _run(feedback_transform="shuffle_months")
    free = slice(4, None)
    oracle_set = {tuple(f.flatten().tolist()) for f in oracle[free]}
    shuffled_set = {tuple(f.flatten().tolist()) for f in shuffled[free]}
    assert shuffled_set <= oracle_set, "shuffle_months must feed REAL fields, not modified ones"
    assert [f.tolist() for f in oracle[free]] != [f.tolist() for f in shuffled[free]], (
        "the order did not change — the shuffle is inert"
    )


def test_statistics_are_recorded_for_every_free_running_step():
    _, _, inf = _run(feedback_transform="identity")
    stats = inf.feedback_field_stats
    assert len(stats) == 4, f"expected one record per (step, target), got {len(stats)}"
    assert stats[0]["persistence"] == -1.0, "the first step has no predecessor"
    assert all(s["n_cells"] == 36 for s in stats)  # per-target grid, not pooled
    assert all(s["target_idx"] == 0 for s in stats)


def test_recorded_statistics_show_thin_actually_thinned():
    """The on-real-data check: an arm that did not bite would otherwise be read as 'no effect'."""
    _, _, ident = _run(feedback_transform="identity")
    _, _, thinned = _run(feedback_transform="thin:1.0")
    assert max(s["active_fraction"] for s in ident.feedback_field_stats) > 0
    assert all(s["active_fraction"] == 0.0 for s in thinned.feedback_field_stats)


def test_unknown_arm_raises_before_any_rollout():
    with pytest.raises(ValueError, match="unknown feedback transform"):
        _run(feedback_transform="thinn:0.5")
