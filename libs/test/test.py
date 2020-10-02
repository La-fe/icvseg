# coding=utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
from libs.test.miou import *
from config import cfg
from lib.datasets.transform import ToTensor
from libs.datasets import RemoDataset
from libs.net.generateNet import generate_net
import torch.optim as optim
from libs.net.sync_batchnorm.replicate import patch_replication_callback
from tqdm import tqdm

from torch.utils.data import DataLoader


def test_net():
    dataset = RemoDataset.RemoDataset(cfg, 'val')

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            # collate_fn=collate_fn,
                            # drop_last=True
                            )

    net = generate_net(cfg)
    print('net initialize')
    if cfg.TEST_CKPT is None:
        raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')

    print('Use %d GPU' % cfg.TEST_GPUS)
    device = torch.device('cuda')
    if cfg.TEST_GPUS > 1:
        net = nn.DataParallel(net)
        patch_replication_callback(net)
    net.to(device)

    print('start loading model %s' % cfg.TEST_CKPT)
    model_dict = torch.load(cfg.TEST_CKPT, map_location=device)
    from collections import OrderedDict
    new_model_dict = OrderedDict()
    mod = net.state_dict()
    for k, v in model_dict.items():
        if k[7:] in mod.keys():
            name = k[7:]  # remove module.
            new_model_dict[name] = v

    net.load_state_dict(new_model_dict)
    net.eval()
    result_list = []
    with torch.no_grad():
        hist = np.zeros((4, 4))
        # for i_batch, sample_batched in tqdm(enumerate(dataloader)):
        for sample_batched in tqdm(dataloader):
            name_batched = sample_batched['name']
            row_batched = sample_batched['row']
            col_batched = sample_batched['col']
            labels_batched = sample_batched['segmentation']

            [batch, channel, height, width] = sample_batched['image'].size()
            multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).to(0)
            for rate in cfg.TEST_MULTISCALE:
                inputs_batched = sample_batched['image_%f' % rate]
                # inputs_batched = sample_batched['image']
                inputs_batched = inputs_batched.cuda(0)
                predicts = net(inputs_batched).to(0)
                predicts_batched = predicts.clone()
                del predicts
                if cfg.TEST_FLIP:
                    inputs_batched_flip = torch.flip(inputs_batched, [3])
                    predicts_flip = torch.flip(net(inputs_batched_flip), [3]).to(0)
                    predicts_batched_flip = predicts_flip.clone()
                    del predicts_flip
                    predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0

                predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1 / rate, mode='bilinear',
                                                 align_corners=True)
                multi_avg = multi_avg + predicts_batched
                del predicts_batched

            multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
            result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)

            for i in range(batch):
                row = row_batched[i]
                col = col_batched[i]

                p = result[i, :, :]
                p = cv2.resize(p, dsize=(col, row), interpolation=cv2.INTER_NEAREST)
                labels = labels_batched[i].cpu().numpy()
                # result_list.append({'predict': p, 'name': name_batched[i]})
                result_list.append({'predict':p, 'gt':labels})

        # dataset.save_result(result_list, cfg.MODEL_NAME)
        dataset.do_python_eval(result_list)
        print('Test finished')



if __name__ == '__main__':
    cfg.txt_f = '/home/xjx/data/mask/Kaggle/data_crop/test.txt'

    cfg.TEST_CKPT = '/home/xjx/data/model/Kaggle/kk_ratio_124_itr56000.pth'
    '''
      backbound: 93.590%	          1: 96.135%
              2: 88.532%	          3: 91.124%
    '''

    # cfg.TEST_CKPT = '/home/xjx/data/model/Kaggle/kaggle_627_epoch96_all.pth'
    'all [0.83542668 0.94958946 0.72471457 0.7660923 ]'
    test_net()

    #