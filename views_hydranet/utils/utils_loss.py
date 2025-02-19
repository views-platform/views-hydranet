# https://pytorch.org/vision/stable/_modules/torchvision/ops/focal_loss.html

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Focal loss balancing parameter
        self.gamma = gamma  # Focal loss focusing parameter
        self.reduction = reduction  # Loss reduction method

    def forward(self, logits, targets):

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

# from https://github.com/ywatanabe1989/custom_losses_pytorch/blob/master/multi_task_loss.py
# inpsired by https://arxiv.org/abs/1705.07115

class MultiTaskLoss(torch.nn.Module):
  '''https://arxiv.org/abs/1705.07115'''
  def __init__(self, is_regression, reduction='none'):
    super(MultiTaskLoss, self).__init__()
    self.is_regression = is_regression
    self.n_tasks = len(is_regression)
    self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
    self.reduction = reduction

  def forward(self, losses):
    dtype = losses.dtype
    device = losses.device
    stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
    self.is_regression = self.is_regression.to(device).to(dtype)
    coeffs = 1 / ( (self.is_regression+1)*(stds**2) )
    multi_task_losses = coeffs*losses + torch.log(stds)

    if self.reduction == 'sum':
      multi_task_losses = multi_task_losses.sum()
    if self.reduction == 'mean':
      multi_task_losses = multi_task_losses.mean()

    return multi_task_losses

'''
usage
is_regression = torch.Tensor([True, True, False]) # True: Regression/MeanSquaredErrorLoss, False: Classification/CrossEntropyLoss
multitaskloss_instance = MultiTaskLoss(is_regression)
params = list(model.parameters()) + list(multitaskloss_instance.parameters())
torch.optim.Adam(params, lr=1e-3)
model.train()
multitaskloss.train()
losses = torch.stack(loss0, loss1, loss3)
multitaskloss = multitaskloss_instance(losses)
'''


class ShrinkageLoss(nn.Module):
    def __init__(self, a=10, c=0.2, size_average=True):
        super(ShrinkageLoss, self).__init__()
        self.a = a  # Shrinkage factor
        self.c = c  # Threshold
        self.size_average = size_average

    def forward(self, input, target):

        input, target = input.unsqueeze(0), target.unsqueeze(0) 

        l = torch.abs(target - input)  # Absolute difference between target and input
        exp_term = torch.exp(self.a * (self.c - l))  # Exponential term to control the sensitivity of the loss to deviations from the target values.
        loss = (l ** 2) / (1 + exp_term)  # Shrinkage loss calculation

        if self.size_average:
            return loss.mean()  # Average the loss if size_average is True
        else:
            return loss.sum()  # Sum the loss if size_average is False






