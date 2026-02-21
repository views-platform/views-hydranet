import pytest
import torch

from views_hydranet.utils.mtloss import MultiTaskLoss


def test_mtloss_initialization():
    """Verify that MTLoss initializes with correct parameter count."""
    is_reg = torch.Tensor([True, True, False])
    mt = MultiTaskLoss(is_reg)

    assert mt.n_tasks == 3
    assert isinstance(mt.log_vars, torch.nn.Parameter)
    assert mt.log_vars.shape == (3,)


def test_mtloss_weighting_logic():
    """Verify that MTLoss weights regression and classification differently."""
    # Logic: coeffs = 1 / ((is_regression + 1) * stds^2)
    # Regression (True=1): 1 / (2 * stds^2)
    # Classification (False=0): 1 / (1 * stds^2)
    is_reg = torch.Tensor([True, False])
    mt = MultiTaskLoss(is_reg)

    # losses: reg_loss=10, class_loss=10
    losses = torch.tensor([10.0, 10.0])

    combined = mt(losses)

    # At init, log_vars=0, so stds=1.
    # reg_coeff = 1 / (2 * 1) = 0.5
    # class_coeff = 1 / (1 * 1) = 1.0
    # Expected: 0.5*10 + ln(1) + 1.0*10 + ln(1) = 5 + 10 = 15
    assert pytest.approx(combined.sum().item(), abs=1e-5) == 15.0


def test_mtloss_parameters_are_learnable():
    """Verify that log_vars parameters receive gradients."""
    is_reg = torch.Tensor([True, True])
    mt = MultiTaskLoss(is_reg)
    losses = torch.tensor([1.0, 2.0], requires_grad=True)

    out = mt(losses).sum()
    out.backward()

    assert mt.log_vars.grad is not None
    assert not torch.all(mt.log_vars.grad == 0)


def test_mtloss_rejects_mismatched_task_count():
    """
    RED GATE: MultiTaskLoss.forward() must raise ValueError when the number
    of losses differs from the number of tasks in is_regression.

    Production context: is_regression has 6 elements (3 regression + 3
    classification). If the model ever outputs a different head count, the
    losses tensor has a different length. Without an explicit guard, PyTorch
    raises a generic RuntimeError deep inside the computation with a message
    that does not identify the task balancer as the source.

    This test asserts that MultiTaskLoss raises ValueError (not RuntimeError)
    with a message referencing 'task', so the failure is immediately actionable.

    Mismatches to test:
    - Too few losses (3 instead of 6) — the most likely production failure
    - Too many losses (8 instead of 6) — catches the symmetric case
    """
    is_reg = torch.Tensor([True, True, True, False, False, False])  # 6 tasks
    mt = MultiTaskLoss(is_reg, reduction="sum")

    # Case A: Too few losses
    with pytest.raises(ValueError, match=r"[Tt]ask|[Mm]ismatch|[Ee]xpected"):
        mt(torch.tensor([1.0, 2.0, 3.0]))  # 3 != 6

    # Case B: Too many losses
    with pytest.raises(ValueError, match=r"[Tt]ask|[Mm]ismatch|[Ee]xpected"):
        mt(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]))  # 8 != 6

    # Case C: Correct count must NOT raise
    mt(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))  # 6 == 6, OK
