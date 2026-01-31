# https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance in binary classification.

    This implementation reduces the relative loss for well-classified examples, 
    putting more focus on hard, misclassified examples. It reduces to 
    Binary Cross Entropy (BCE) when gamma=0 and alpha=0.5.

    Paper: https://arxiv.org/abs/1708.02002

    Args:
        alpha (float): Balancing parameter for the rare class. Default: 0.25.
        gamma (float): Focusing parameter. Higher values reduce focus on 
                       easy examples. Default: 2.0.
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Calculates the focal loss.

        Args:
            logits (torch.Tensor): Predicted unnormalized logits.
            targets (torch.Tensor): Ground truth binary targets (0 or 1).

        Returns:
            torch.Tensor: Scalar loss if reduction is 'mean' or 'sum', 
                          otherwise a tensor of shape [1, *logits.shape].
        """
        # Internal unsqueeze matches expected pipeline volume format
        logits, targets = logits.unsqueeze(0), targets.unsqueeze(0)

        # since you are not taking log(p) anywhere, you don't need to clamp it for numerical stability.
        p = torch.sigmoid(logits)

        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")# Calculate the cross-entropy loss. inputs should be Predicted unnormalized logits according to the documentation
        p_t = p * targets + (1 - p) * (1 - targets) # Calculate the probability of the true class
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss # multiple alpha_t with targets here to balance the loss

        if self.reduction == 'mean':
            return loss.mean()  # Average the loss if reduction is set to 'mean'
        elif self.reduction == 'sum':
            return loss.sum()  # Sum the loss if reduction is set to 'sum'
        else:
            return loss  # Return the focal loss without reduction

