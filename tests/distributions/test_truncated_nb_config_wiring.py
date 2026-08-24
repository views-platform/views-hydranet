"""Config/head/loss/composition wiring for `truncated_nb` — the coverage gap before it is trained.

`test_config_integration.py`, `test_head_wiring.py` and `test_loss_wiring.py` all parametrize only
`["nb", "zinb"]`. `truncated_nb` has a thorough *unit* suite
(`test_truncated_negative_binomial.py`) but, until this file, **nothing asserted it survives the
path a real run takes**: config validation, head sizing, loss routing, and composition.

That gap matters now because the family has never been trained at scale — its only vehicle,
That gap matters now because the family has never been trained at scale — its only vehicle,
A wiring defect would surface ~2 hours into a GPU run, four times over.

Pre-registration: `reports/2026-08-24_truncated_nb_dossier/05_analysis_plan.md` §3, §6.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (  # noqa: E402
    HydraBNUNet06_LSTM4,
)
from views_hydranet.distributions import get_family, resolve_family  # noqa: E402
from views_hydranet.distributions.composition import compose_samples  # noqa: E402
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.distributions.truncated_negative_binomial import (  # noqa: E402
    TruncatedNBFamily,
)

H = W = 16
IN_CH, HID_CH = 8, 32


# ── config: the one-key change must validate with everything else untouched ──


def test_config_accepts_truncated_nb_with_soft_gate(valid_config_dict):
    """The exact change the experiment makes, through the real validator.

    `truncated_nb` is NOT self-zeroed, so `validate_forecast_composition` requires a gate. This
    pins that `soft_gate` is accepted — the composition the arms actually run.
    """
    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = {
        **valid_config_dict,
        "output_distribution": "truncated_nb",
        "forecast_composition": "soft_gate",
    }
    resolved = HydraNetConfig(**cfg)
    assert resolved.output_distribution == "truncated_nb"
    assert resolved.forecast_composition == "soft_gate"


def test_config_rejects_truncated_nb_with_self_zeroed(valid_config_dict):
    """`self_zeroed` has no zero mechanism for a body that never draws zero — must fail loud.

    Without this the misconfiguration would emit a forecast that is never zero anywhere, which on
    a 99.7%-zero grid is a catastrophic silent error rather than a crash.
    """
    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = {
        **valid_config_dict,
        "output_distribution": "truncated_nb",
        "forecast_composition": "self_zeroed",
    }
    with pytest.raises(ValueError, match="not self-zeroed|self_zeroed"):
        HydraNetConfig(**cfg)


# ── head: reg width must be targets × n_params, exactly as for nb ──


def test_head_reg_width_matches_n_params():
    torch.manual_seed(7)
    model = (
        HydraBNUNet06_LSTM4(IN_CH, HID_CH, 1, 0.0, output_distribution="truncated_nb")
        .float()
        .train()
    )
    torch.manual_seed(11)
    out = model(torch.randn(1, IN_CH, H, W).float(), model.init_hTtime(HID_CH, H, W).float())
    assert out.reg.shape[1] == 3 * 2, "3 targets x (mu, theta)"
    assert out.cls.shape[1] == 3, "classification / AR-feedback width must be unchanged"


# ── loss: choose_loss must route to the family NLL, not to loss_reg ──


def test_choose_loss_routes_to_truncated_family(valid_config_dict):
    """`loss_reg='mse'` is present and valid but MUST be ignored for a family head.

    If this regressed, the run would silently optimise MSE on (mu, theta) channels — trainable,
    plausible-looking, and wrong.
    """
    from views_hydranet.utils.utils import choose_loss

    cfg = {**valid_config_dict, "output_distribution": "truncated_nb", "loss_reg": "mse"}
    criterion_reg, _, _ = choose_loss(cfg, torch.device("cpu"))
    assert isinstance(criterion_reg, FamilyLoss)
    assert isinstance(criterion_reg.family, TruncatedNBFamily)
    assert criterion_reg.n_params == 2


# ── composition: the whole point — the gate becomes the ONLY zero source ──


def test_soft_gate_composition_makes_the_gate_the_only_zero_source():
    """The mechanism the experiment tests, measured rather than assumed.

    `nb` composes to a small FRACTION of the gate because its own draw is often 0; `truncated_nb`
    composes to the gate itself. Tolerance is set from the Bernoulli standard error at this
    prevalence, not tuned: ~4300 expected actives in 1.5e6 cells => se/mean ~1.5%, 6% is ~4 sd.
    """
    n = 1_500_000
    params = torch.stack([torch.full((n,), 0.05), torch.full((n,), 1.0)], dim=-1)
    gate_p = 0.0034
    gate = torch.full((n,), gate_p)

    fracs = {}
    for name in ("nb", "truncated_nb"):
        fam = get_family(name)
        gen = torch.Generator().manual_seed(4711)
        draws = fam.sample(params, 1, gen).unsqueeze(-2)  # [n, 1(target), 1(k)]
        composed = compose_samples(draws, gate.unsqueeze(-1), "soft_gate", None, gen)
        fracs[name] = float((composed > 0).float().mean())

    assert fracs["truncated_nb"] == pytest.approx(gate_p, rel=0.06), (
        f"truncated_nb composed occurrence {fracs['truncated_nb']:.5f} should equal the gate "
        f"{gate_p} — the body must contribute no zeros"
    )
    assert fracs["nb"] < 0.25 * gate_p, (
        f"nb composed occurrence {fracs['nb']:.5f} should be far BELOW the gate {gate_p} — this"
        " is the double-applied zero process the experiment removes; if this fails, the"
        " premise is gone"
    )


def test_truncated_family_never_draws_zero_and_prob_positive_is_one():
    """The two structural properties the arm builder also asserts before spending GPU time."""
    fam = resolve_family("truncated_nb")
    params = torch.stack(
        [torch.full((5000,), 1e-4), torch.full((5000,), 0.5)], dim=-1
    )  # the mu->0 background
    draws = fam.sample(params, 1, torch.Generator().manual_seed(3))
    assert int((draws == 0).sum()) == 0
    assert torch.equal(fam.prob_positive(params), torch.ones(5000))
