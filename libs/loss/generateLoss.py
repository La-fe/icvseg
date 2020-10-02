# encoding: utf-8
import torch
import torch.nn as nn
# from libs.loss.lovasz_losses as lovasz
from libs.loss.lovasz_losses_base import lovasz_softmax
import torch.nn.functional as F




class CE_loss:
    def __init__(self, ignore_index=255):
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, )

    def __call__(self, output, target):
        return self.criterion(output, target)

class CE_edge_loss:
    def __init__(self, weight=5):
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')

    def __call__(self, output, target):
        loss_map = self.criterion(output, target['mask'])
        edges = target['edge']
        edges_map = ((edges / edges.max()) * self.weight + 1)
        edges_map = edges_map.to(self.cfg.GPUS_ID[1])
        edges_loss_map = torch.mul(loss_map, edges_map)
        loss = edges_loss_map.mean()

        return loss

# class Generate_loss:
#     def __init__(self, cfg):
#         self.loss_type = cfg.LOSS_NAME
#         if self.loss_type == 'lovaszloss' :
#             self.criterion = L
#
#         elif self.loss_type == 'CEloss':
#             self.criterion = nn.CrossEntropyLoss(ignore_index=255)
#
#         elif self.loss_type  == 'CEloss_edgeWeight' and hasattr(cfg, 'edge_loss_weight'):
#             self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
#         else:
#             pass
#             # raise ValueError('generateNet.py: network %s is not support yet' % cfg.MODEL_NAME)
#         self.cfg = cfg
#
#     def comput_loss(self, predicts_batched, labels_batched):
#
#         if self.loss_type == 'lovasz_softmax':
#             # loss = lovasz_losses.lovasz_hinge(predicts_batched, labels_batched, ignore=255)
#             loss = L.lovasz_softmax(predicts_batched, labels_batched, classes=[1], ignore=255)
#         elif self.loss_type == 'lovasz_biner':
#             loss = lovasz_biner(predicts_batched, labels_batched)
#         elif self.loss_type == 'CE':
#             loss = self.criterion(predicts_batched, labels_batched)
#         else:
#             raise ('no type of loss type')
#
#         return loss

def lovasz_biner(logit, truth):
    # 2分类
    # logit = logit.squeeze(1)
    # truth = truth.squeeze(1)
    loss = L.lovasz_hinge(logit, truth,ignore=255)  # 计算损失每张图
    return loss



def edges_loss_weight(loss, edges, weight=5):
    '''
    边缘损失加权, weight -1
    :param loss:
    :param edges:
    :param weight:
    :return:
    '''
    edges_map = ((edges / edges.max())*weight + 1)
    edges_map = edges_map.to(cfg.GPUS_ID[1])
    edges_loss_map = torch.mul(loss, edges_map)

    return edges_loss_map


import torch
import torch.nn as nn

from torch.autograd import Function


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

# from https://github.com/Hsuxu/Loss_ToolBox-PyTorch

'''
self.INIT_loss = edict({
            "type": "LovaszSoftmax",
            "args": {
                'reduction': 'mean'
            }
        })
'''
class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',
                 ignore_lb=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        ignore = label.data.cpu() == self.ignore_lb
        n_valid = (ignore == 0).sum()
        label[ignore] = 0

        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        mask = torch.ones_like(logits)
        mask[[a, torch.arange(mask.size(1)), *b]] = 0

        # compute loss
        probs = torch.sigmoid(logits)
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        pt = torch.where(lb_one_hot == 1, probs, 1 - probs)
        alpha = self.alpha * lb_one_hot + (1 - self.alpha) * (1 - lb_one_hot)
        loss = -alpha * ((1 - pt) ** self.gamma) * torch.log(pt + 1e-12)
        loss[mask == 0] = 0
        # print(loss.size())
        if self.reduction == 'mean':
            loss = loss.sum(dim=1).sum() / n_valid
        return loss



class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, loss_type='softmax'):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.loss_type = loss_type

    def forward(self, logit, target):
        B, C, H, W = logit.size()
        target = target.view(-1, 1).long()
        onehot_target = torch.FloatTensor(B*H*W, C).zero_().cuda()
        onehot_target.scatter_(1, target, 1.) + 1

        logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
        if self.loss_type == 'sigmoid':
            prob = torch.sigmoid(logit)
        elif self.loss_type == 'softmax':
            prob = F.softmax(logit, 1)
        prob = torch.clamp(prob,1e-8,1-1e-8)

        prob = (prob * onehot_target).sum(1).view(-1,1)
        prob = torch.clamp(prob, 1e-8, 1-1e-8)
        batch_loss = - (torch.pow((1-prob), self.gamma))*prob.log()
        loss = batch_loss.mean()

        return loss

class BCELoss2d(nn.Module):
    def __init__(self, loss_type='softmax'):
        super(BCELoss2d, self).__init__()
        self.loss_type = loss_type

    def forward(self, logit, target):
        B, C, H, W = logit.size()
        target = target.view(-1, 1).long()
        onehot_target = torch.FloatTensor(B*H*W, C).zero_().cuda()
        onehot_target.scatter_(1, target, 1.)
        logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
        if self.loss_type == 'sigmoid':
            loss = nn.BCEWithLogitsLoss(reduction='sum')(logit, onehot_target)
        elif self.loss_type == 'softmax':
            prob = F.softmax(logit, 1)
            loss = nn.BCELoss(reduction='sum')(prob, onehot_target)

        loss /= (B*H*W)

        return loss


'''
 self.INIT_loss = edict({
            "type": "Lovasz_focal2_loss",
            "args": {
                'focal_weight': 0.5,
                'lovasz_weight': 0.5,

                'gamma':2,
                'loss_type':"sigmoid"
            }
        })
'''


class Lovasz_focal2_loss(nn.Module):
    def __init__(self, focal_weight=0.5, lovasz_weight=0.5,
                 gamma=2,
                 loss_type='sigmoid'):
        super(Lovasz_focal2_loss, self).__init__()


        self.focal_loss = FocalLoss2d(gamma, loss_type)

        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        inputs_softmax = F.log_softmax(inputs, dim=1)
        return self.focal_weight * self.focal_loss(inputs_softmax, targets) + self.lovasz_weight * lovasz_softmax(inputs_softmax,
                                                                                                            targets)


'''
 self.INIT_loss = edict({
            "type": "Lovasz_CE_loss",
            "args": {
                'ce_weight': 0.5,
                'lovasz_weight': 0.5
            }
        })
'''
class Lovasz_CE_loss(nn.Module):
    def __init__(self, ce_weight=0, lovasz_weight=0):
        super(Lovasz_CE_loss, self).__init__()

        self.lovasz_loss = LovaszSoftmax()
        self.ce_loss = CE_loss()

        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, inputs, targets):
        return self.ce_weight * self.ce_loss( inputs, targets) + self.lovasz_weight * self.lovasz_loss( inputs, targets)


'''
 self.INIT_loss = edict({
            "type": "Lovasz_focal_loss",
            "args": {
                'focal_weight': 1,
                'lovasz_weight': 1,
                
                'alpha':0.25,
                'gamma':2,
                'reduction':"mean",
                'ignore_lb':255
            }
        })
'''

class Lovasz_focal_loss(nn.Module):
    def __init__(self, focal_weight=0, lovasz_weight=0,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',
                 ignore_lb=255):
        
        super(Lovasz_focal_loss, self).__init__()
        
        self.lovasz_loss = LovaszSoftmax()
        self.focal_loss = FocalLoss(alpha, gamma, reduction, ignore_lb)

        self.focal_weight = focal_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal_loss(inputs, targets) + self.lovasz_weight * self.lovasz_loss(inputs, targets)

