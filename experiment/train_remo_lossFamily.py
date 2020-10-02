# encoding: utf-8
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
sys.path.append("../..") # Unet/lib
from config import cfg
from libs.datasets.RemoDataset import RemoDataset
from libs.net.generateNet import generate_net
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from libs.loss.generateLoss import Generate_loss
#from net.loss import MaskCrossEntropyLoss, MaskBCELoss, MaskBCEWithLogitsLoss
from libs.net.sync_batchnorm.replicate import patch_replication_callback
# from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

torch.cuda.set_device(cfg.GPUS_ID[1]) # 设置主gpu

# !!!!!!!!!!!!!!! 服务器上要修改为 False
flag_debug = True
def train_net():
    dataset = RemoDataset( cfg, 'train')

    # 自定义的组合函数，可以使用自带的组合函数，为了 debug
    def collate_fn(batch):
        images = []
        seg = []
        edg = []
        cs = []
        rs = []
        names = []
        for _,sample in enumerate(batch):
            images.append(sample['image'])
            seg.append(sample['segmentation'])
            rs.append(sample['row'])
            cs.append(sample['col'])
            names.append(sample['name'])
        if hasattr(cfg, 'edge_loss_weight'):
            edg.append(sample['edges'])

            return {
                'image': torch.stack(images,0),
                'segmentation': torch.stack(seg,0),
                'edges': torch.stack(edg,0),
                    }
        else:
            return {
                'image': torch.stack(images, 0),
                'segmentation': torch.stack(seg, 0),
            }

    
    dataloader = DataLoader(dataset,
                batch_size=cfg.TRAIN_BATCHES, 
                shuffle= cfg.TRAIN_SHUFFLE,
                num_workers= cfg.DATA_WORKERS,
                collate_fn = collate_fn,
                drop_last = True
                )

    net = generate_net(cfg)
    if cfg.TRAIN_TBLOG:
        from tensorboardX import SummaryWriter
        # Set the Tensorboard logger
        tblogger = SummaryWriter(cfg.LOG_DIR)
    print('    Total params: %.2fM' % (sum(p.numel() for p in net.parameters())/1000000.0)) # 计算参数量

    print('Use %d GPU'%cfg.TRAIN_GPUS)
    
    # 多 GPU 的模型训练问题
    device = torch.device(cfg.GPUS_ID[0])
    if cfg.TRAIN_GPUS > 1:
        if flag_debug:
            net = nn.DataParallel(net, device_ids=[0]) # 测试用
        else:
            net = nn.DataParallel(net, device_ids=cfg.GPUS_ID)
        patch_replication_callback(net)
    net.to(device)		
    
    if cfg.TRAIN_CKPT:
        pretrained_dict = torch.load(cfg.TRAIN_CKPT)
        net_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        print(f' * load {cfg.TRAIN_CKPT}')
        # net.load_state_dict(torch.load(cfg.TRAIN_CKPT),False)

    loss_func = Generate_loss(cfg)
    # baseline、head使用不同的学习率
    optimizer = optim.SGD(
        params = [
            {'params': get_params(net.module,key='1x'), 'lr': cfg.TRAIN_LR},
            {'params': get_params(net.module,key='10x'), 'lr': 10*cfg.TRAIN_LR}
        ],
        momentum=cfg.TRAIN_MOMENTUM
    )
    itr = cfg.TRAIN_MINEPOCH * len(dataloader)
    max_itr = cfg.TRAIN_EPOCHS * len(dataloader)
    running_loss = 0.0
    tblogger = SummaryWriter(cfg.LOG_DIR)
    for epoch in range(cfg.TRAIN_MINEPOCH, cfg.TRAIN_EPOCHS):
        for i_batch, sample_batched in enumerate(dataloader):
            now_lr = adjust_lr(optimizer, itr, max_itr)  # 学习率的改变，参数怎么传递的
            if hasattr(cfg, 'edge_loss_weight'):
                inputs_batched, labels_batched, edges_batched = sample_batched['image'], sample_batched['segmentation'], sample_batched['edges']
            else:
                inputs_batched, labels_batched = sample_batched['image'], sample_batched['segmentation']

            optimizer.zero_grad()
            labels_batched = labels_batched.long().to(cfg.GPUS_ID[1])
            predicts_batched = net(inputs_batched)
            predicts_batched = predicts_batched.to(cfg.GPUS_ID[1])
            loss = loss_func.comput_loss(predicts_batched, labels_batched)
            # -------- 边缘损失加权 -----------------
            if hasattr(cfg, 'edge_loss_weight'):
                loss = edges_loss_weight(loss, edges_batched, weight=cfg.edge_loss_weight)
                # -------- 损失图 可视化 -----------------
                # vis_loss_map(loss)
                loss = loss.mean()

            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            if i_batch % 10 ==1:
                print('epoch:%d/%d\tbatch:%d/%d\titr:%d\tlr:%g\tloss:%g ' %
                    (epoch, cfg.TRAIN_EPOCHS, i_batch, dataset.__len__()//cfg.TRAIN_BATCHES,
                    itr+1, now_lr, running_loss))
            if cfg.TRAIN_TBLOG and itr%100 == 0:
                inputs = inputs_batched.numpy()[0]/2.0 + 0.5
                labels = labels_batched[0].cpu().numpy()
                labels_color = dataset.label2colormap(labels).transpose((2,0,1))
                predicts = torch.argmax(predicts_batched[0],dim=0).cpu().numpy()
                predicts_color = dataset.label2colormap(predicts).transpose((2,0,1))
                pix_acc = np.sum(labels==predicts)/(cfg.DATA_RESCALE**2)
    
                tblogger.add_scalar('loss', running_loss, itr)
                tblogger.add_scalar('lr', now_lr, itr)
                tblogger.add_scalar('pixel acc', pix_acc, itr)
                tblogger.add_image('Input', inputs, itr)
                tblogger.add_image('Label', labels_color, itr)
                tblogger.add_image('Output', predicts_color, itr)
            running_loss = 0.0
            
            if itr % 10000 == 0:
                # save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_itr%d.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,itr))
                save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_itr%d.pth'%(cfg.EXP_NAME, itr))
                torch.save(net.state_dict(), save_path)
                print('%s has been saved'%save_path)
    
            itr += 1
        
    # save_path = os.path.join(cfg.MODEL_SAVE_DIR,'%s_%s_%s_epoch%d_all.pth'%(cfg.MODEL_NAME,cfg.MODEL_BACKBONE,cfg.DATA_NAME,cfg.TRAIN_EPOCHS))
    save_path = os.path.join(cfg.MODEL_SAVE_DIR, '%s_epoch%d_all.pth' % (cfg.EXP_NAME, cfg.TRAIN_EPOCHS))
    torch.save(net.state_dict(),save_path)
    if cfg.TRAIN_TBLOG:
        tblogger.close()
    print('%s has been saved'%save_path)

def adjust_lr(optimizer, itr, max_itr):
    now_lr = cfg.TRAIN_LR * (1 - itr/(max_itr+1)) ** cfg.TRAIN_POWER
    optimizer.param_groups[0]['lr'] = now_lr
    optimizer.param_groups[1]['lr'] = 10*now_lr
    return now_lr

def get_params(model, key):
    for m in model.named_modules():
        if key == '1x':
            if 'backbone' in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if 'backbone' not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p

def vis_loss_map(loss):
    '''
    可视化 loss 热图
    :param loss:
    :return:
    '''
    loss_map = loss.cpu().detach().numpy()
    bs, *_= loss_map.shape

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

if __name__ == '__main__':
    train_net()


