import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
    
class ELBOLoss(nn.Module):

    def __init__(self, mse_loss,eps=1e-8,reduction='mean'):
        super(ELBOLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.mse_loss = mse_loss
    def log_normal(self, x, mu, var):
        var = var + self.eps
        return -0.5 * torch.sum(
            np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)
    def forward(self, x_rec,x ,z, z_mu, z_var, z_mu_prior, z_var_prior,logits, prob):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)
        log_q = F.log_softmax(logits, dim=-1)     
        if self.reduction=='sum':
            loss  = loss.sum() 
            loss  = loss + torch.sum(torch.sum(prob * log_q, dim=-1)- np.log(0.1))
            loss_rec = self.mse_loss(x_rec,x).sum(-1).sum()
        elif self.reduction=='mean':
            loss = loss.mean()
            loss  = loss + torch.mean(torch.sum(prob * log_q, dim=-1)) - np.log(0.1)
            loss_rec = self.mse_loss(x_rec,x).sum(-1).mean()
        loss =loss + loss_rec
        return loss
    
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
    def forward(self, output, target,target_smooth):
        log_preds = F.log_softmax(output, dim=-1)
        loss = -1*torch.sum(target_smooth*log_preds, 1)
        if self.reduction=='sum':
            loss = loss.sum()
        else:
            loss = loss.mean()
        return loss*self.eps + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


