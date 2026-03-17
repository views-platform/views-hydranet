# from https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
# inpsired by https://arxiv.org/abs/1705.07115

import torch


class MultiTaskLoss(torch.nn.Module):
    """
    Learnable Multi-Task Loss balancer.

    Automatically weights multiple regression and classification losses using
    homoscedastic uncertainty. This allows the model to balance the relative
    contribution of different tasks without manual hyperparameter tuning.

    Paper: https://arxiv.org/abs/1705.07115

    Attributes:
        log_vars (torch.nn.Parameter): Learnable log-variance coefficients per task.
    """

    def __init__(self, is_regression, reduction="none"):
        """
        Initializes the MultiTaskLoss balancer.

        Args:
            is_regression (torch.Tensor): Boolean tensor indicating if task i is regression.
            reduction (str): 'mean', 'sum', or 'none'. Default: 'none'.
        """
        super(MultiTaskLoss, self).__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, losses):
        """
        Balances the input losses using learnable coefficients.

        Args:
            losses (torch.Tensor): Tensor of raw loss values from each task.

        Returns:
            torch.Tensor: Weighted and regularized combined loss.
        """
        if len(losses) != self.n_tasks:
            raise ValueError(
                f"MultiTaskLoss task count mismatch: expected {self.n_tasks} losses "
                f"(matching is_regression length), but received {len(losses)}. "
                f"Check that the model's output head count matches the loss mask."
            )
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars) ** (1 / 2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)
        # Add epsilon to prevent division by zero
        eps = 1e-8
        coeffs = 1 / ((self.is_regression + 1) * (stds**2) + eps)
        multi_task_losses = coeffs * losses + torch.log(stds + eps)

        if self.reduction == "sum":
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == "mean":
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses
