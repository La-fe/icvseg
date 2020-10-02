# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import time
import cv2
import os, sys
sys.path.append("../..") # Unet/lib
from libs.datasets.generateData import generate_dataset, generate_test_dataset, generate_img_test
# from libs.net.generateNet import generate_net
import libs.net.generateNet as net_gener
import torch.optim as optim
from libs.net.sync_batchnorm.replicate import patch_replication_callback
import libs.utils.test_utils as func
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import mmcv
# import ipdb

torch.cuda.set_device(0) # 设置主gpu

class NetLoader:
    def __init__(self, cfg, model_name='deeplabv3plus', model_backbone='res101_atrous', ckpt='', num_classes=21, flip=False, multi_scale=1, vis_type='person', edge_width=5):
        self.cfg = cfg
        self.cfg.TEST_CKPT = ckpt
        self.cfg.MODEL_NUM_CLASSES = num_classes
        self.cfg.TEST_FLIP = flip
        if isinstance(multi_scale, int):
            self.cfg.TEST_MULTISCALE = [multi_scale]
        elif isinstance(multi_scale, list):
            self.cfg.TEST_MULTISCALE  = multi_scale

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
            predicts = self.net(inputs_batched)  # .to(0)
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

        result = torch.argmax(multi_avg, dim=1)
        result = result.cpu().numpy().astype(np.uint8)

        row = row_batched
        col = col_batched
        mask_img = result[0, :, :]
        mask_img = cv2.resize(mask_img, dsize=(col, row), interpolation=cv2.INTER_NEAREST)


        # dict(mask_img=mask_img_n, mix_img=mix_img, raw_img=raw_img, person=img_person, back=img_back, change_back_addweight=all_img, change_back=bad_all_img )
        img_dict = self.func_transform(raw_img, mask_img)

        return img_dict

def nothing(emp):
    pass

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
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 获得总帧数
    # print cap.get(cv2.CAP_PROP_FPS) # 获得FPS
    loop_flag = 0
    pos = 0
    cv2.createTrackbar('time', 'video', 0, frames, nothing)

    trans_CropRoi = func.CropRoi(cfg)
    while 1:
        if loop_flag == pos: # 视频起始位置
            loop_flag = loop_flag + 1
            cv2.setTrackbarPos('time', 'video', loop_flag)
        else: # 设置视频播放位置
            pos = cv2.getTrackbarPos('time', 'video')
            loop_flag = pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos) # 设置当前帧所在位置
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

            im_show_vis_type.append(pre_dict[str(net.vis_type)]) # 每个网络选择自己的输出模式
            # im_show_vis_type.append(pre_dict['mix_img']) # 每个网络选择自己的输出模式
            # roiPred = trans_CropRoi(pre_dict, net, generate_img)
            # im_show_vis_type.append(roiPred['mix_img'])

            # mask_list.append(pre_dict['mask_img'][..., 0])
            # mask_list.append(roiPred['mix_img'])
            # raw_list.append(pre_dict['raw_img'

        im_show_vis_type.append(raw_img)  # 每个网络选择自己的输出模式
        list_name = list(nets_dict.keys())
        img_one = func.putImgsToOne(im_show_vis_type, list_name, 4, (0, 20), fonts=1,color=(0, 0, 255))
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
                    roi_im_show_all.append(pre_dict[str(net.vis_type)]) # 每个网络选择自己的输出模式

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


def main(cfg, nets_dict, flag):
    '''

    :param cfg:
    :param nets_dict: {name: <class>Net_loader}
    NetLoader  __call__ return : dict(mask_img, mix_img, raw_img)
    :return:
    '''
    
    generate_img = generate_img_test(cfg)
    # imgs_dir = "/home/xjx/data/videos/mask/rvos/5_mask"
    imgs_dir = "/home/lafe/work/mask/data/alldata/imgsave"
    # imgs_list = ['49bbb9a205de862aa7dced16a6c22ddf_7.jpg', 'aa937c02e2119ec99dbac74ba0906ac3_3.jpg',
    #              '76a73c6827e2a643341d3a6e6a23c92b_3.jpg']
    # imgs_list = [os.path.join(imgs_dir, i) for i in imgs_list]

    # video_root = "/home/xjx/data/videos/mask/test_video"
    # video_root = "/home/xjx/data/videos/mask/mask_test_video/PT_LB_Video"
    video_root = "/home/xjx/data/videos/mask/mask_test_video/sz_dy"
    # video_root = "/home/xjx/data/videos/mask/mask_test_video/PT_LB_Video" # 单人抖音视频
    # video_root = "/home/xjx/data/videos/videoD/det_raw_video"

    # videos = [[os.path.join(video_root, l), 0] for l in list(os.listdir(video_root))]
    # videos = sorted(videos, key=lambda x: x)
    # videos = [["/home/xjx/data/videos/mask/mask_test_video/sz_dy/PM_LB_JWJ_000056.mp4", 0]]



    flag_showIOU = False
    '''  mix_img  mask_img  raw_img  person  back '''

    # if flag_img:
    if flag == 0:
        if imgs_list is None:
            imgs_list = list(os.path.join(imgs_dir, l) for l in list(os.listdir(imgs_dir)))
            imgs_list = sorted(imgs_list , key=lambda x: x)





        for img_dir in imgs_list:
            try:
                im = cv2.imread(img_dir)
                try_test = im.shape
                # im = im[478:955, 190:351, :]
            except:
                print("read error %s " % img_dir)
                continue
            img_tensor = generate_img(im) # 图片进行torch 格式转换
            im_show_vis_type = list()
            mask_list = []
            for net_name in nets_dict.keys():
                net = nets_dict[net_name]
                pre_dict = net(img_tensor)
                im_show_vis_type.append(pre_dict[str(net.vis_type)])  # 每个网络选择自己的输出模式
                mask_list.append(pre_dict['mask_img'][...,0]) # 获得mask 单通道

            img_one = func.putImgsToOne(im_show_vis_type, list(nets_dict.keys()), 4, (0, 20), fonts=1,
                                        color=(0, 0, 255))

            cv2.imshow('img', img_one)
            k = cv2.waitKey(0)
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            if k == ord('s'):
                cv2.imwrite('/home/xjx/data/videos/mask/rvos/5_mask/0.png', mask_list[0])

        print('Test finished')

    # if flag_video:
    if flag == 1:
        for video_name in videos:
            try:
                show_videos_with_trackbar(video_name, nets_dict, generate_img, flag_showIOU)
            except:
                print("%s finised!" %video_name[0])
                cv2.destroyAllWindows()

    # cap
    if flag == 2:
        try:
            cap = cv2.VideoCapture(0)
            ret, im = cap.read()
            zzz = im.shape
        except:
            cap = cv2.VideoCapture(1)

        cap_frame = 0
        cv2.namedWindow("img", cv2.NORM_HAMMING2)
        while 1:
            ret, im = cap.read()
            # 1. ------------------增广操作------------------------------
            # cv2.imshow('raw', im)
            img_tensor = generate_img(im)

            im_show_vis_type = list()
            for net_name in nets_dict.keys():
                net = nets_dict[net_name]
                pre_dict = net(img_tensor)
                im_show_vis_type.append(pre_dict[str(net.vis_type)])  # 每个网络选择自己的输出模式

            img_one = func.putImgsToOne(im_show_vis_type, list(nets_dict.keys()), 4, (0, 20), fonts=1,
                                        color=(0, 0, 255))
            # --------------- 马赛克 --------------------
            # ii = im_show_all['person'][1]
            # h, w, c = ii.shape
            # im_show_all['person'][1] = do_mosaic(ii, 0, 0, w, h, 20)

            # cv2.resizeWindow("img", 1920, 1080)
            cv2.imshow("img", img_one)
            key = cv2.waitKey(5)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    sys.path.insert(0, "./")
    from config.mt_config_A1 import Configuration

    cfg = Configuration()

    cfg.DATA_WORKERS = 2
    cfg.TEST_BATCHES = 1
    cfg.TEST_GPUS = 1

    # img_dir = "/home/xjx/data/videos/mask/LB_DY_000045"
    img_dir = "/home/lafe/work/mask/data/alldata/imgsave"
    # img_dir = "/home/xjx/data/videos/mask/cut_video/LB_DY_000001"
    # img_dir = "/home/xjx/data/videos/mask/green_back"
    # img_dir = ["/home/xjx/data/videos/mask/cut_video"]
    # img_dir = ["/home/xjx/data/videos/mask/cut_video"]
    # img_dir = ["/home/xjx/data/videos/mask/Green_Screen"]
    # img_dir = ["/home/xjx/data/videos/mask/Green_Screen_1"]
    '''
        model_name: deeplabv3plus,    deeplabv3plus_fpn_multidecode, deeplabv3plus_fpn, deeplabv3plus_mutilAspp, 
        model_backbone:  res101_atrous  xception
        
        vis_type: mix_img, mask_img,  raw_img, person, back,  change_back_addweight, change_back,
            
    '''
    # cfg_1024input = Configuration()
    # cfg_1024input.DATA_RESCALE = 512
    # cfg_1024input.DATA_RANDOMCROP = 512
    nets_dict = {
        # 'fpn_multidecode_19': NetLoader(cfg, 'deeplabv3plus_fpn_multidecode', 'res101_atrous',
        #                                ckpt='/home/xjx/work/mask/deeplab/model/deeplabv3plus_fpn_multidecode_res101_atrous_VOC2012_itr190000.pth',
        #                                num_classes=2, flip=False, multi_scale=1),
        # 'raw': NetLoader(Configuration(), 'deeplabv3plus', 'res101_atrous',
        #                                ckpt='/home/xjx/data/model/deeplabv3+voc/deeplabv3plus_res101_atrous_VOC2012_itr190000.pth',
        #                                num_classes=2, flip=False, multi_scale=1, vis_type='mix_img', edge_width=5), #
        # 'mutilAspp': NetLoader(cfg, 'deeplabv3plus_mutilAspp', 'res101_atrous',
        #                   ckpt='/home/xjx/work/mask/deeplab/model/deeplabv3plus_mutilAspp_res101_atrous_VOC2012_itr120000.pth',
        #                        num_classes=2, fli p=False, multi_scale=1),
        # 'mutilAspp_26w': NetLoader(cfg, 'deeplabv3plus_mutilAspp', 'res101_atrous',
        #                   ckpt='/home/xjx/work/mask/deeplab/model/deeplabv3plus_mutilAspp_res101_atrous_VOC2012_itr260000.pth',
        #                   num_classes=2, flip=False, multi_scale=1),
        # 'Decorate': NetLoader(cfg, 'deeplabv3plus', 'res101_atrous',
        #                   ckpt='/home/xjx/work/mask/deeplab/model/deeplab_precise_Decorate_617_itr60000.pth',
        #                   num_classes=2, flip=False, multi_scale=1, vis_type='change_back_addweight'),
        # ' Decorate 15': NetLoader(Configuration(), 'deeplabv3plus', 'res101_atrous',
        #                   ckpt='/home/xjx/data/model/deeplab_precise_Decorate_617_itr150000.pth',
        #                   num_classes=2, flip=True, multi_scale=1, vis_type='mix_img', edge_width=2),
        # 'parse+lip ': NetLoader(Configuration(), 'deeplabv3plus', 'res101_atrous',
        #                   ckpt='/home/xjx/data/model/deeplab_precise_Decorate_LMv2_703_itr300000.pth',
        #                   num_classes=2, flip=True, multi_scale=1, vis_type='mix_img', edge_width=2),
        'Unet Celoss ': NetLoader(Configuration(), 'deeplabv3plus', 'res50_atrous',
                          ckpt='/home/lafe/work/mask/ckpt/best_model_mt_configA1_miou.pth',
                          num_classes=2, flip=True, multi_scale=1, vis_type='mix_img', edge_width=2),
        # 'parse ': NetLoader(Configuration(), 'deeplabv3plus', 'res101_atrous',
        #                   ckpt='/home/xjx/data/model/deeplab_LMv2_703_itr96epoch.pth',
        #                   num_classes=2, flip=True, multi_scale=1, vis_type='mix_img', edge_width=2),
        # '? ': NetLoader(Configuration(), 'deeplabv3plus', 'res101_atrous',
        #                   ckpt='/home/xjx/data/model/deeplab_precise_Decorate_gt_precise_620_epoch96all.pth',
        #                   num_classes=2, flip=False, multi_scale=1, vis_type='mix_img', edge_width=2),
        # '1024 input': NetLoader(cfg_1024input, 'deeplabv3plus', 'res101_atrous',
        #                   ckpt='/home/xjx/data/model/deeplabv3plus_res101_atrous_VOC2012_itr240000.pth',
        #                   num_classes=2, flip=False    , multi_scale=1, vis_type='mix_img', edge_width=2),
        # ' deeplabv3plus_fpn_multidecode 12': NetLoader(Configuration(), 'deeplabv3plus_fpn_multidecode', 'res101_atrous',
        #                   ckpt='/home/xjx/work/mask/deeplab/model/deeplabv3plus_fpn_multidecode_res101_atrous_VOC2012_epoch96_all.pth',
        #                   num_classes=2, flip=True, multi_scale=1, vis_type='change_back_addweight'),
        # ' raw1 ': NetLoader(cfg, 'deeplabv3plus', 'res101_atrous',
        #                   ckpt='/home/xjx/work/mask/deeplab/model/deeplabv3plus_fpn_res101_atrous_VOC2012_itr120000.pth',
        #                   num_classes=2, flip=False, multi_scale=1),
        # ' edge': NetLoader(cfg, 'deeplabv3plus', 'res101_atrous',
        #                   ckpt='/home/xjx/work/mask/deeplab/model/deeplab_precise_Decorate_edgeB4W3_620_itr120000.pth',
        #                   num_classes=2, flip=False, multi_scale=1),
        # # 'xception': NetLoader(cfg, 'deeplabv3plus', 'res101_atrous',
        # #                   ckpt='/home/xjx/work/mask/deeplab/model/deeplabv3+voc/deeplabv3plus_res101_atrous_VOC2012_itr190000.pth',
        # #                   num_classes=2, flip=False, multi_scale=1),
        # 'fpn_multidecode_26': NetLoader(cfg, 'deeplabv3plus_fpn_multidecode', 'res101_atrous',
        #                                ckpt='/home/xjx/work/mask/deeplab/model/deeplabv3plus_fpn_multidecode_res101_atrous_VOC2012_itr260000.pth',
        #                                num_classes=2, flip=False, multi_scale=1),
    }

    # 0:img 1:video 2:cap
    main(cfg, nets_dict, 0)

