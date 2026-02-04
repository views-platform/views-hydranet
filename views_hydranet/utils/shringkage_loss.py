import torch
import torch.nn as nn


class ShrinkageLoss(nn.Module):
    """
    Shrinkage Loss for regression tasks with noise.

    This loss function is a variation of Mean Squared Error (MSE) that 
    penalizes small deviations (below threshold 'c') significantly less 
     than larger ones. It is designed to 'shrink' the contribution 
    of noise to the total loss.

    Formula: loss = (l^2) / (1 + exp(a * (c - l)))
    where l = |target - input|

    Args:
        a (float): Shrinkage factor (steepness of the shrinkage transition). Default: 10.
        c (float): Threshold below which errors are aggressively shrunk. Default: 0.2.
        size_average (bool): If True, returns the mean loss. Otherwise returns sum.
    """
    def __init__(self, a=10, c=0.2, size_average=True):
        super(ShrinkageLoss, self).__init__()
        self.a = a
        self.c = c
        self.size_average = size_average

    def forward(self, input, target):
        """
        Calculates the shrinkage loss.

        Args:
            input (torch.Tensor): Predicted values.
            target (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Scalar loss.
        """
        input, target = input.unsqueeze(0), target.unsqueeze(0)

        diff = torch.abs(target - input)
        exp_term = torch.exp(self.a * (self.c - diff))
        loss = (diff ** 2) / (1 + exp_term)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

