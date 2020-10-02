# encoding: utf-8

import sys
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import time
import cv2

from icvseg.libs.datasets.generateData import generate_dataset, generate_test_dataset, generate_img_test
# from libs.net.generateNet import generate_net
import icvseg.libs.net.generateNet as net_gener
import torch.optim as optim
from icvseg.libs.net.sync_batchnorm.replicate import patch_replication_callback
import icvseg.libs.utils.test_utils as func
from icvseg.libs.utils.util_color import tensor2label as tensor2label
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import mmcv
from tqdm import tqdm


def init_cfg(cfg_path):
    try:
        config_path, basename = os.path.split(cfg_path)
        print('import config path: ', config_path)
        sys.path.insert(0, config_path)
        config_name, ext = os.path.splitext(basename)
        config_file = __import__(config_name)
    except ImportError:
        raise ("not find config")

    cfg = config_file.Configuration()
    return cfg




class NetLoader:
    def __init__(self, cfg, model_name='deeplabv3plus', model_backbone='res101_atrous', ckpt='', num_classes=21,
                 flip=False, multi_scale=1, vis_type='person', edge_width=5):
        self.cfg = cfg
        self.cfg.TEST_CKPT = ckpt
        self.cfg.MODEL_NUM_CLASSES = num_classes
        self.cfg.TEST_FLIP = flip
        if isinstance(multi_scale, int):
            self.cfg.TEST_MULTISCALE = [multi_scale]
        elif isinstance(multi_scale, list):
            self.cfg.TEST_MULTISCALE = multi_scale

        self.cfg.MODEL_NAME = model_name
        # cfg.MODEL_NAME = 'deeplabv3plus_fpn_multidecode'
        # cfg.MODEL_NAME = 'deeplabv3plus_fpn'
        self.cfg.MODEL_BACKBONE = model_backbone

        self.vis_type = vis_type
        self.net = self.model_init(self.cfg)
        self.num_class = num_classes
        self.edge_width = edge_width

        self.func_transform = func.Transform(self.cfg)

    def model_init(self, cfg):
        # net = generate_net(cfg)
        net = cfg.initialize_args(net_gener, 'INIT_model')
        print("net initialize model name : ", cfg.MODEL_NAME)
        if cfg.TEST_CKPT is None:
            raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
        print('start loading model %s' % cfg.TEST_CKPT)

        device = torch.device(0)
        model_dict = torch.load(cfg.TEST_CKPT, map_location=device)  # 在其他gpu训练需要用map搞到测试gpu上
        from collections import OrderedDict
        new_model_dict = OrderedDict()
        mod = net.state_dict()
        for k, v in model_dict.items():
            if k[7:] in mod.keys():
                name = k[7:]  # remove module.
                new_model_dict[name] = v

        net.load_state_dict(new_model_dict)

        net.eval()
        net.cuda()
        return net

    def __call__(self, sample):
        row_batched = sample['row']
        col_batched = sample['col']
        raw_img = sample['raw']
        [batch, channel, height, width] = sample['image'].size()
        multi_avg = torch.zeros((batch, self.num_class, height, width), dtype=torch.float32).cuda()  # .to(0)

        for rate in self.cfg.TEST_MULTISCALE:
            inputs_batched = sample['image_%f' % rate].cuda()
            predicts = self.net(inputs_batched)  # .to(0)KK
            predicts_batched = predicts.clone()
            # func.vis_mask_feat(predicts_batched, True, row_batched, col_batched)

            del predicts
            if self.cfg.TEST_FLIP:
                inputs_batched_flip = torch.flip(inputs_batched, [3])
                predicts_flip = torch.flip(self.net(inputs_batched_flip), [3]).to(0)
                predicts_batched_flip = predicts_flip.clone()
                del predicts_flip
                predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0
            predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1 / rate, mode='bilinear',
                                             align_corners=True)
            multi_avg = multi_avg + predicts_batched
            del predicts_batched
        multi_avg = multi_avg / len(self.cfg.TEST_MULTISCALE)
        # func.vis_mask_feat(multi_avg, True, row_batched, col_batched)

        result_torch = torch.argmax(multi_avg, dim=1)

        result = result_torch.cpu().numpy().astype(np.uint8)

        row = row_batched
        col = col_batched
        mask_img = result[0, :, :]
        mask_img = cv2.resize(mask_img, dsize=(col, row), interpolation=cv2.INTER_NEAREST)
        color_mask = tensor2label(result_torch, self.cfg.MODEL_NUM_CLASSES)
        color_mask = cv2.resize(color_mask, dsize=(col, row), interpolation=cv2.INTER_NEAREST)

        # dict(mask_img=mask_img_n, mix_img=mix_img, raw_img=raw_img, person=img_person, back=img_back, change_back_addweight=all_img, change_back=bad_all_img )
        img_dict = self.func_transform(raw_img, mask_img)
        img_dict['color_mask'] = color_mask
        # cv2.imshow('img', img_dict['mask_img'])
        # k = cv2.waitKey(0)
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
        return img_dict


def nothing(emp):
    pass


def loopshow(res, savep=None):
    if len(res) == 0:
        return False
    ind = 0
    if savep is not None:
        os.makedirs(savep, exist_ok=True)

    while True:
        cv2.imshow('img', res[ind])
        # cv2.imshow("raw, fake, enmap", np.concatenate(res, axis=1))
        k = cv2.waitKey(0)
        if k == ord('d'):
            ind += 1
        elif k == ord('a'):
            ind -= 1
        elif k == ord('q'):
            cv2.destroyAllWindows()
            raise TypeError(" ****** STOP *********")
        elif k == ord('s'):
            savep_img = f'{savep}/{ind}.png'
            cv2.imwrite(savep_img, res[ind])
            print(' * save in:', savep_img)
        if ind >= len(res):
            ind = 0


def show_videos_with_trackbar(video_name, nets_dict, generate_img, flag_showIOU=False):
    '''

    :param video_name: list(["video path ", int])
    :param nets_dict:
    :param generate_img:  bgr img -> tensor 等操作变换函数
    :return:
    '''
    video = video_name[0]
    cv2.namedWindow("video", cv2.NORM_HAMMING2)
    cv2.resizeWindow("video", 1920, 1080)
    cap = cv2.VideoCapture(video)
    save_path = os.path.join("/home/xjx/data/videos/mask/mask_test_video/lta", video.split('/')[-1].split('.')[0])
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获得总帧数
    # print cap.get(cv2.CAP_PROP_FPS) # 获得FPS
    loop_flag = 0
    pos = 0
    cv2.createTrackbar('time', 'video', 0, frames, nothing)

    trans_CropRoi = func.CropRoi(cfg)
    while 1:
        if loop_flag == pos:  # 视频起始位置
            loop_flag = loop_flag + 1
            cv2.setTrackbarPos('time', 'video', loop_flag)
        else:  # 设置视频播放位置
            pos = cv2.getTrackbarPos('time', 'video')
            loop_flag = pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)  # 设置当前帧所在位置
        ret, raw_img = cap.read()
        # cv2.imshow("raw", raw_img)
        img_tensor = generate_img(raw_img)

        # im_show_all = dict(mix_img=[], mask_img=[], raw_img=[], person=[], back=[], change_back_addweight=[], change_back=[])# 组合图片
        im_show_vis_type = list()
        mask_list = list()
        raw_list = list()
        for net_name in nets_dict.keys():
            net = nets_dict[net_name]
            pre_dict = net(img_tensor)

            im_show_vis_type.append(pre_dict[str(net.vis_type)])  # 每个网络选择自己的输出模式
            # im_show_vis_type.append(pre_dict['mix_img']) # 每个网络选择自己的输出模式
            # roiPred = trans_CropRoi(pre_dict, net, generate_img)
            # im_show_vis_type.append(roiPred['mix_img'])

            # mask_list.append(pre_dict['mask_img'][..., 0])
            # mask_list.append(roiPred['mix_img'])
            # raw_list.append(pre_dict['raw_img'

        im_show_vis_type.append(raw_img)  # 每个网络选择自己的输出模式
        list_name = list(nets_dict.keys())
        img_one = func.putImgsToOne(im_show_vis_type, list_name, 4, (0, 20), fonts=1, color=(0, 0, 255))
        # mask_one = func.putImgsToOne(mask_list, list_name, 4, (0, 20), fonts=1,color=(0, 0, 255))

        if flag_showIOU:
            roi_img = func.vis_Image_from_iou(raw_img)
            # roi_im_show_all = dict(mix_img=[], mask_img=[], raw_img=[], person=[], back=[], change_back_addweight=[], change_back=[])# 组合图片
            roi_im_show_all = list()
            if roi_img.shape[0] != 0:  # 如果为空 shape为 (0,0,3)
                roi_img_tensor = generate_img(roi_img)
                for net_name in nets_dict.keys():
                    net = nets_dict[net_name]
                    pre_dict = nets_dict[net_name](roi_img_tensor)
                    roi_im_show_all.append(pre_dict[str(net.vis_type)])  # 每个网络选择自己的输出模式

                roi_img_one = func.putImgsToOne(roi_im_show_all, list(nets_dict.keys()), 4, (0, 20), fonts=1,
                                                color=(0, 0, 255))

                cv2.imshow("Image", roi_img_one)

        cv2.imshow("video", img_one)
        # cv2.imshow("mask", mask_one)

        key = cv2.waitKey(0)
        if key == ord('q') or loop_flag == frames:
            break
        if key == ord('s'):
            mmcv.imwrite(raw_list[-1], "%s/%06d.jpg" % (save_path, loop_flag))
            mmcv.imwrite(mask_list[-1], "%s/%06d.png" % (save_path, loop_flag))


def inference(cfg, nets_dict, imgs_dir=None, mask_save=None, flag_save=False):
    '''

    :param cfg:
    :param nets_dict: {name: <class>Net_loader}
    NetLoader  __call__ return : dict(mask_img, mix_img, raw_img)
    :return:
    '''
    if flag_save:
        os.makedirs(mask_save, exist_ok=True)

    generate_img = generate_img_test(cfg)
    flag_showIOU = False
    '''  mix_img  mask_img  raw_img  person  back '''

    # if flag_img:
    imgs_list = list(os.path.join(imgs_dir, l) for l in list(os.listdir(imgs_dir)))
    imgs_list = sorted(imgs_list, key=lambda x: x)
    vis = []
    for img_dir in tqdm(imgs_list):
        try:
            im = cv2.imread(img_dir)
            basename = os.path.basename(img_dir)
            try_test = im.shape
            # im = im[478:955, 190:351, :]
        except:
            print("read error %s " % img_dir)
            continue
        img_tensor = generate_img(im)  # 图片进行torch 格式转换
        im_show_vis_type = list()
        mask_list = []
        for net_name in nets_dict.keys():
            net = nets_dict[net_name]
            s = time.time()
            pre_dict = net(img_tensor)
            print(' * infere time: ', time.time() - s)
            # im_show_vis_type.append(pre_dict[str(net.vis_type)])  # 每个网络选择自己的输出模式
            for k, v in pre_dict.items():
                im_show_vis_type.append(v)
            mask_list.append(pre_dict['mask_img'])  # 获得mask 单通道

        if flag_save:
            savename = os.path.join(mask_save, basename.replace('jpg', 'png'))
            mask = mask_list[-1]
            if len(mask.shape) == 3:
                mask = mask[..., 0]
            cv2.imwrite(savename, mask)
        else:
            img_one = func.putImgsToOne(im_show_vis_type, list(nets_dict.keys()), 4, (0, 20), fonts=1,
                                        color=(0, 0, 255))
            vis.append(img_one)
            # cv2.imshow('img', img_one)
            # k = cv2.waitKey(0)
            # if k == ord('q'):
            #     cv2.destroyAllWindows()
            #     break
    if not flag_save:
        loopshow(vis)

    print('Test finished')


if __name__ == '__main__':
    # sys.path.insert(0, "./")
    # from config.faceparse.mt_facepars_config2 import Configuration
    # from config.human.mt_human import Configuration

    # from config.human.mu import Configuration
    path = '/home/lafe/mt/segmentation/icvseg/config/human/mt_human_skin_art_celeb.py'
    cfg = init_cfg(path)
    # cfg = Configuration()

    cfg.DATA_WORKERS = 2
    cfg.TEST_BATCHES = 1
    cfg.TEST_GPUS = 1
    # imgs_dir = "/home/lafe/mt/gan/demo_img/helen_sr/HR/x4"
    # imgs_dir = "/home/lafe/tmp"

    # imgs_dir = '/data1/datasets_lafe/mask/human_alpha/id_crop/img'
    # imgs_dir = '/data1/datasets_lafe/a2b_data/zheng_jian_zhao_8_11/train_B_nopad_ali'
    # imgs_dir = '/data1/datasets_lafe/mask/coco_lip_mosaic/coco_LIP_mosaic_img'
    # imgs_dir = '/data1/datasets_lafe/a2b_data/zheng_jian_zhao_7_28/train_A_nopad_ali'
    # imgs_dir = '/data1/datasets_lafe/a2b_data/autops_crop_result/meta_original'
    imgs_dir = '/data2/test/img'
    flag_save = False
    mask_save = '/data2/test/mask_results2'

    '''
        model_name: deeplabv3plus,    deeplabv3plus_fpn_multidecode, deeplabv3plus_fpn, deeplabv3plus_mutilAspp, 
        model_backbone:  res101_atrous  xception

        vis_type: mix_img, mask_img,  raw_img, person, back,  change_back_addweight, change_back,

    '''
    # nets_dict = {
    #     'Decorate': NetLoader(cfg, 'deeplabv3plus', 'res101_atrous',
    #                       ckpt='/home/lafe/mt/segmentation/mmsegment_v1/ckpt/model/mt_human/best_model_mt_human_miou.pth',
    #                       num_classes=2, flip=False, multi_scale=1, vis_type='color_mask'),
    # }
    # nets_dict = {
    #     'Decorate': NetLoader(cfg, 'deeplabv3plus', 'res101_atrous',
    #                       ckpt='/home/lafe/mt/segmentation/mmsegment_v1/ckpt/model/mt_human_predictb/best_model_mt_human_predictb_miou.pth',
    #                       num_classes=2, flip=False, multi_scale=1, vis_type='color_mask'),
    # }
    nets_dict = {
        'Decorate': NetLoader(cfg, 'deeplabv3plus', 'res101_atrous',
                              ckpt='/home/lafe/mt/segmentation/mmsegment_v1/ckpt/model/mt_human_no_idphoto/best_model_mt_human_no_idphoto_miou.pth',
                              # ckpt='/home/lafe/mt/segmentation/mmsegment_v1/ckpt/model/mt_human/best_model_mt_human_miou.pth',
                              # ckpt='/home/lafe/mt/segmentation/mmsegment_v1/ckpt/model/mt_human_skin_art_celeb/best_model_mt_human_skin_art_celeb_miou.pth',

                              # ckpt='/home/lafe/mt/segmentation/mmsegment_v1/ckpt/model/mt_human_skin_art_celeb/best_model_mt_human_skin_art_celeb_miou.pth',
                              # ckpt='/home/lafe/mt/segmentation/mmsegment_v1/ckpt/model/mt_human_mtdata_changebg/best_model_mt_human_mtdata_changebg_miou.pth',
                              # ckpt='/home/lafe/mt/segmentation/mmsegment_v1/ckpt/model/mt_human_skin_art_celeb/mt_human_skin_art_celeb_epoch48_all.pth',
                              # ckpt='/home/lafe/mt/segmentation/mmsegment_v1/ckpt/model/mt_human_mtdata_changebg_purebg/mt_human_mtdata_changebg_purebg_epoch48_all.pth',
                              num_classes=2, flip=False, multi_scale=1, vis_type='color_mask'),

    }

    # 0:img 1:video 2:cap
    s = time.time()
    inference(cfg, nets_dict, imgs_dir, mask_save, flag_save)
    print(' * time: ', time.time() - s)



