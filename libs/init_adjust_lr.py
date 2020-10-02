import torch.nn as nn
class LRsc_poly:
    def __init__(self, power, max_itr, lr, optimizer):
        self.power = power
        self.max_itr = max_itr
        self.lr = lr
        self.optimizer = optimizer

    def __call__(self, itr):
        now_lr = self.lr * (1 - itr / (self.max_itr + 1)) ** self.power
        if len(self.optimizer.param_groups) == 1:
            self.optimizer.param_groups[0]['lr'] = now_lr
        else:
            self.optimizer.param_groups[0]['lr'] = now_lr
            self.optimizer.param_groups[1]['lr'] = 10 * now_lr
        return now_lr 

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']



class Param_change:
    def __init__(self, lr, net=None):

        self.params = [
                    {'params': self.get_params(net.module, key='1x'), 'lr': lr},
                    {'params': self.get_params(net.module, key='10x'), 'lr': lr*10}
                ]


    def get_params(self, model, key):
        for m in model.named_modules():
            if key == '1x':
                if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
            elif key == '10x':
                if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p

class Param_default:
    def __init__(self, lr, net=None):
        self.params = [{'params':net.parameters(), 'lr':lr}]
