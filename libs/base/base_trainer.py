# encoding: utf-8

import os, sys
sys.dont_write_bytecode = True
import argparse
import numpy as np
import math
import random
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from libs.datasets.RemoDataset import RemoDataset
import libs.net.generateNet as net_gener
from libs.net.sync_batchnorm.replicate import patch_replication_callback
import libs.utils.train_utils as utils

import libs.init_adjust_lr as  init_adjustlr
import libs.loss.generateLoss as  init_loss
from libs.init_metric import Evaluator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True  # 设置cudnn基准模式 输入相同时效果更好.

# torch.cuda.set_device(cfg.GPUS_ID[1])  # 设置主gpu



class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.GPUS_ID[0])

        # ---------------- dataset ---------------------
        self.dataset = RemoDataset(cfg, 'train')
        self.val_dataset = RemoDataset(cfg, 'val')

        self.dataloader = DataLoader(self.dataset,
                                     batch_size=cfg.TRAIN_BATCHES,
                                     shuffle=cfg.TRAIN_SHUFFLE,
                                     num_workers=cfg.DATA_WORKERS,
                                     collate_fn=self.dataset.collate_fn,
                                     drop_last=True
                                     )

        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=cfg.VAL_BATCHES,
                                         shuffle=cfg.VAL_SHUFFLE,
                                         num_workers=cfg.DATA_WORKERS,
                                         collate_fn=self.dataset.collate_fn,
                                         drop_last=False
                                         )

        # ---------------- net ---------------------
        net = cfg.initialize_args(net_gener, 'INIT_model')
        print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))  # 计算参数量

        if len(cfg.GPUS_ID) > 1:
            if flag_debug:
                net = nn.DataParallel(net, device_ids=[0])  # 测试用
            else:
                net = nn.DataParallel(net, device_ids=cfg.GPUS_ID, output_device=cfg.GPUS_ID[0])
            patch_replication_callback(net)
        self.net = net.to(self.device)

        # ---------------- resume checkpoint ---------------------
        if cfg.TRAIN_CKPT is not None:
            pretrained_dict = torch.load(cfg.TRAIN_CKPT)
            net_dict = self.net.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               (k in net_dict) and (v.shape == net_dict[k].shape)}
            net_dict.update(pretrained_dict)
            self.net.load_state_dict(net_dict)


        # ---------------- loss init ---------------------
        self.max_itr = self.cfg.TRAIN_EPOCHS * len(self.dataloader)

        params_ge = cfg.initialize(init_adjustlr, "INIT_params", net=self.net)
        self.optimizer = cfg.initialize(torch.optim, "INIT_optim", params_ge.params)

        self.adjust_lr = cfg.initialize(init_adjustlr, "INIT_adjust_lr", max_itr=self.max_itr, lr=self.cfg.TRAIN_LR,
                                        optimizer=self.optimizer)

        self.criterion = cfg.initialize(init_loss, "INIT_loss", )

        self.best_pred = 0.0

    def train(self):

        ave_total_loss = utils.AverageMeter()

        itr = self.cfg.TRAIN_MINEPOCH * len(self.dataloader)
        tblogger = SummaryWriter(self.cfg.LOG_DIR)
        for epoch in range(self.cfg.TRAIN_MINEPOCH, self.cfg.TRAIN_EPOCHS):
            for i_batch, sample_batched in enumerate(self.dataloader):

                now_lr = self.adjust_lr(itr)  # 学习率的改变，参数传递
                inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']

                self.optimizer.zero_grad()
                labels_batched = labels_batched.long().to(self.cfg.GPUS_ID[1])
                predicts_batched = self.net(inputs_batched)
                predicts_batched = predicts_batched.to(self.cfg.GPUS_ID[1])
                loss = self.criterion(predicts_batched, labels_batched)

                loss.backward()
                self.optimizer.step()
                ave_total_loss.update(loss.item())

                if i_batch % 10 == 1:
                    print('Epoch: [{}][{}/{}] |\t'
                          'itr: {} |\t'
                          'lr: {:.6f} |\t'
                          'avg_Loss: {:.6f} |\t'
                          'loss: {:.6f} |\t'
                          .format(epoch, i_batch, len(self.dataset) // self.cfg.TRAIN_BATCHES,
                                  itr + 1,
                                  now_lr, ave_total_loss.average(), loss.item())
                          )

                if self.cfg.TRAIN_TBLOG and itr % self.cfg.display == 0:
                    inputs = inputs_batched.numpy()[0] / 2.0 + 0.5
                    labels = labels_batched[0].cpu().numpy()
                    labels_color = self.dataset.label2colormap(labels).transpose((2, 0, 1))
                    predicts = torch.argmax(predicts_batched[0], dim=0).cpu().numpy()
                    predicts_color = self.dataset.label2colormap(predicts).transpose((2, 0, 1))
                    pix_acc = np.sum(labels == predicts) / (self.cfg.DATA_RESCALE ** 2)

                    tblogger.add_scalar('loss', loss.data.item(), itr)
                    tblogger.add_scalar('avg_loss', ave_total_loss.average(), itr)
                    tblogger.add_scalar('lr', now_lr, itr)
                    tblogger.add_scalar('pixel acc', pix_acc, itr)
                    tblogger.add_image('Input', inputs, itr)
                    tblogger.add_image('Label', labels_color, itr)
                    tblogger.add_image('Output', predicts_color, itr)

                if itr % self.cfg.TRAIN_SAVE_CHECKPOINT == 0:
                    save_path = os.path.join(self.cfg.MODEL_SAVE_DIR, '%s_itr%d.pth' % (self.cfg.EXP_NAME, itr))
                    torch.save(self.net.state_dict(), save_path)
                    print('%s has been saved' % save_path)

                if itr % self.cfg.VAL_CHECKPOINT == 0 and itr != 0:
                    miou, iou_list, macc = self.validation_matrix(itr)  # use time 16

                    self.net.train()
                    tblogger.add_scalar('mIoU', miou, itr)
                    tblogger.add_scalar('macc', macc, itr)
                    tblogger.add_scalar('b', iou_list[0], itr)

                    for i in range(1, len(iou_list)):
                        tblogger.add_scalar('class %s' % str(i), iou_list[i], itr)

                itr += 1

        save_path = os.path.join(self.cfg.MODEL_SAVE_DIR,
                                 '%s_epoch%d_all.pth' % (self.cfg.EXP_NAME, self.cfg.TRAIN_EPOCHS))
        torch.save(self.net.state_dict(), save_path)
        if self.cfg.TRAIN_TBLOG:
            tblogger.close()
        print('%s has been saved' % save_path)

    def validation_matrix(self, itr):
        evaler = Evaluator(self.cfg.MODEL_NUM_CLASSES)

        is_best = False
        self.net.eval()

        torch.cuda.empty_cache()
        with torch.no_grad():
            for sample_batched in tqdm(self.val_dataloader, desc='val'):
                inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']

                predicts_batched = self.net(inputs_batched)

                result = torch.argmax(predicts_batched, dim=1).cpu().numpy().astype(np.int)
                gt = labels_batched.cpu().numpy().astype(np.int)
                evaler.add_batch(result, gt)

        IoU, Miou_list = evaler.Mean_Intersection_over_Union()
        macc = evaler.Pixel_Accuracy_Class()

        c_list = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            if i == 0:
                c_list.append(Miou_list[i] * 100)
                print('%11s:%7.3f%%' % ('b', c_list[0]), end='\t')

            else:
                c_list.append(Miou_list[i] * 100)
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (str(i), Miou_list[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (str(i), Miou_list[i] * 100))

        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', IoU * 100))
        print('%11s:%7.3f%%' % ('acc', macc * 100))
        torch.cuda.empty_cache()

        new_pred = IoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred

        if is_best:
            best_filename = os.path.join(self.cfg.MODEL_SAVE_DIR, 'best_model_%s_miou.pth' % (self.cfg.EXP_NAME))
            torch.save(self.net.state_dict(), best_filename)

        return IoU, Miou_list, macc

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

    def vis_loss_map(self, loss):
        '''
        可视化 loss 热图
        :param loss:
        :return:
        '''
        loss_map = loss.cpu().detach().numpy()
        bs, *_ = loss_map.shape

        # bs_img = list()
        for i in range(bs):
            loss_map_0 = loss_map[i]
            print(loss_map_0.max())
            print(loss_map_0.min())
            l = (loss_map_0).astype(np.uint8)
            l_img = cv2.applyColorMap(l, cv2.COLORMAP_JET)
            cv2.imshow("img", l_img)
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()
                break

    # 参考 https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    def find_lr(self, init_value=1e-8, final_value=10., beta=0.98, num=None):

        trn_loader = self.val_dataloader
        # trn_loader = self.dataloader

        if num is None:
            num = len(trn_loader) - 1

        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        self.optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        losses = []
        log_lrs = []

        iterator = iter(trn_loader)
        for batch_num in tqdm(range(num)):
            try:
                sample = next(iterator)
            except StopIteration:
                iterator = iter(iterator)
                sample = next(iterator)

            # As before, get the loss for this mini-batch of inputs/outputs
            inputs, labels = sample['image'], sample['segmentation']
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()

            outputs = self.net(inputs)
            outputs = outputs.to(self.cfg.GPUS_ID[1])
            labels = labels.long().to(self.cfg.GPUS_ID[1])
            loss = self.criterion(outputs, labels)
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.data.cpu()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            # Stop if the loss is exploding
            # if batch_num > 1 and smoothed_loss > 4 * best_loss:
            #     return log_lrs, losses

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))

            # Do the SGD step
            loss.backward()
            self.optimizer.step()

            # Update the lr for the next step
            lr *= mult
            self.optimizer.param_groups[0]['lr'] = lr

        print('draw the lr grap')
        plt.plot(log_lrs[10:-5], losses[10:-5])
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.savefig("base_lr:%f,end_lr:%f_num:%d.png" % (init_value, final_value, num))
        # return log_lrs, losses


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="remo_unet_oc.py",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0, 0",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # Parse gpu ids
    gpus = utils.parse_devices(args.gpus)
    num_gpus = len(gpus)

    # import config
    sys.path.insert(0, "config")
    try:
        config_name, ext = os.path.splitext(os.path.basename(args.cfg))
        config_file = __import__(config_name)
    except ImportError:
        raise ("not find config")

    cfg = config_file.cfg


    cfg.TRAIN_BATCHES = num_gpus * cfg.batch_size_per_gpu
    cfg.VAL_BATCHES = num_gpus * cfg.val_batch_size_per_gpu

    if flag_debug:
        cfg.GPUS_ID = [0, 0]
    else:
        cfg.GPUS_ID = gpus


    set_seed()
    trainer = Trainer(cfg)
    trainer.train()
    # trainer.find_lr(1e-5, 1, num=100)
