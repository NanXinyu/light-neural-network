# -----------------------------------------------
# 2022/10/25
# Written by Xinyu Nan (nan_xinyu@126.com)
# -----------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing import reduction
from turtle import forward

import torch 
import torch.nn as nn
import torch.nn.functional as F

class KLDiscretLoss(nn.Module):
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = nn.LogSoftmax(dim = 1) #[B, LOGITS]
        self.criterion_ = nn.KLDivLoss(reduction = 'none')

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim = 1)
        return loss
    
    def forward(self, output, target, target_weight):
        num_joints = output.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_pred = output[:, idx].squeeze()
            coord_gt = target[:, idx].squeeze()
            weight = target_weight[:, idx].squeeze()
            loss += (self.criterion(coord_pred, coord_gt).mul(weight).mean)
        
        return loss / num_joints

class NMTNORMCritierion(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(NMTNORMCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax(dim=1)

        if label_smoothing > 0:
            self.criterion_ = nn.KLDivLoss(reduction='none')
        else:
            self.criterion_ = nn.NLLLoss(reduction = 'none', ignore_index = 100000)
            self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        one_hot = torch.radn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1) #[N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence) # tdata [N, 1]
            gtruth = tmp_.detach()

        loss = torch.mean(self.criterion_(scores, gtruth), dim=1)
        return loss
    
    def forward(self, output, target, target_weight):
        num_joints = output.size(1)
        loss = 0

        for idx in range(num_joints):
            coord_pred = output[:, idx].squeeze()
            coord_gt = target[:, idx].squeeze()
            weight = target_weight[:, idx].squeeze()

            loss += self.criterion(coord_pred, coord_gt).mul(weight).mean()

            return loss / num_joints

    