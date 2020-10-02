# encoding: utf-8
import torch
import argparse
import os
import os.path as osp
import sys
import cv2
import time
import copy
from easydict import EasyDict as edict

if torch.cuda.device_count() > 1:
    flag_debug = False
else:
    flag_debug = True

class Configuration():
    def __init__(self):
        # self.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname("__file__")))
        self.ROOT_DIR = osp.abspath(os.path.dirname("__file__"))
        self.EXP_NAME = osp.splitext(osp.basename(__file__))[0]  # 获取config.py 名称作为model 名称

        # ----------------- dataset ----------------------
        if flag_debug is True:
            self.txt_f = '/home/lafe/work/mask/data/layout/all.txt'
            self.root_dir = '/home/lafe/work/mask/data/alldata'

            self.val_txt_f = '/home/lafe/work/mask/data/layout/all.txt' # 1.jpg, 1.png
            self.val_root_dir = '/home/lafe/work/mask/data/alldata'
        else:
            self.txt_f =  ['/data1/datasets_lafe/mask/coco_lip_mosaic/layouts/precise_03/precise_03_right_train.txt',
                           ]
            self.root_dir = ['/data1/datasets_lafe/mask/coco_lip_mosaic',
                             ]

            self.val_txt_f = ['/data1/datasets_lafe/mask/coco_lip_mosaic/layouts/precise_03/precise_03_right_val.txt',
                              ]
            self.val_root_dir = ['/data1/datasets_lafe/mask/coco_lip_mosaic',
                                 ]

        self.DATA_WORKERS = 94
        self.DATA_RESCALE = int(512)
        self.DATA_RANDOMCROP = 0
        self.DATA_RANDOMROTATION = 0
        self.DATA_RANDOMSCALE = 1  # 1/r ~ r 的范围
        self.DATA_RANDOM_H = 10
        self.DATA_RANDOM_S = 10
        self.DATA_RANDOM_V = 10
        self.DATA_RANDOMFLIP = 0.0
        self.DATA_gt_precise = 0  # 边缘细化 向内缩进2个像素
        self.edge_width = 0  # 边缘宽度

        # self.INIT_dataset = edict({
        #     "type": "RemoDataset",
        #     "args": {
        #         'txt_f': self.txt_f,
        #         'root_dir': self.root_dir,
        #
        #         'DATA_RESCALE': 512,
        #         'DATA_RANDOMCROP': 0,
        #         'DATA_RANDOMROTATION': 0,
        #         'DATA_RANDOMSCALE': 2, # 1/r ~ r 的范围
        #         'DATA_RANDOM_H': 10,
        #         'DATA_RANDOM_S': 10,
        #         'DATA_RANDOM_V': 10,
        #         'DATA_RANDOMFLIP': 0.5,
        #         'DATA_gt_precise': 0,
        #         'edge_width': 0
        #     }
        # })
        #
        # self.INIT_dataset_val = copy.deepcopy(self.INIT_dataset)
        # self.INIT_dataset_val.args.update(edict({
        #     'val_txt_f': self.val_txt_f,
        #     'val_root_dir': self.val_root_dir,
        #     })
        # )

        # ----------------- model ------------------------
        self.MODEL_SAVE_DIR = osp.join(self.ROOT_DIR, "ckpt", 'model', self.EXP_NAME)
        self.LOG_DIR = osp.join(self.ROOT_DIR, 'ckpt', 'log', self.EXP_NAME)

        self.MODEL_NUM_CLASSES = 2  # class

        self.INIT_model = edict({
            "type": "deeplabv3plus",
            "args": {
                'num_classes': self.MODEL_NUM_CLASSES,
                "MODEL_BACKBONE": 'res50_atrous',
                'MODEL_OUTPUT_STRIDE': 16,
                'MODEL_ASPP_OUTDIM': 256,
                'MODEL_SHORTCUT_DIM': 48,
                'MODEL_SHORTCUT_KERNEL': 1,
            }
        })

        # ----------------- loss -------------------------
        self.INIT_loss = edict({
            "type": "CE_loss",
            "args": {
                'ignore_index': 255
            }
        })

        # ----------------- optmi ------------------------
        self.TRAIN_LR = 0.007  # learning rate

        self.INIT_optim = edict({
            "type": "SGD",
            "args": {
                # 'lr': self.TRAIN_LR,
                "momentum": 0.9,
                "weight_decay": 0.00004,
            }
        })
        self.INIT_params = edict({
            "type": "Param_change",  # backbone 1x, other 10x
            "args": {
                "lr": self.TRAIN_LR
            }
        })

        # ----------------- adjust lr --------------------
        self.INIT_adjust_lr = edict({
            "type": "LRsc_poly",
            "args": {
                'power': 0.9
            }
        })

        self.TRAIN_BN_MOM = 0.0003  # sy bn 参数

        if flag_debug is True:
            self.GPUS_ID = [0, 0]
            self.batch_size_per_gpu = 1  # bs
        else:
            self.GPUS_ID = [2, 3]
            self.batch_size_per_gpu = 36  # 每张卡的bs

        self.TRAIN_SAVE_CHECKPOINT = 1000  # save checkpoint
        self.TRAIN_SHUFFLE = True
        self.display = 50

        self.TRAIN_MINEPOCH = 0
        self.TRAIN_EPOCHS = 48  # epoch
        self.TRAIN_TBLOG = True
        self.TRAIN_CKPT = None

        # =========== val ==================
        if flag_debug:
            self.val_batch_size_per_gpu = 2
        else:
            self.val_batch_size_per_gpu = 24

        self.VAL_CHECKPOINT = 500  # val checkpoint
        self.VAL_SHUFFLE = False

        self.TEST_MULTISCALE = [1]  # multisscale
        self.TEST_FLIP = True
        self.TEST_CKPT = ''
        self.TEST_GPUS = 1
        self.TEST_BATCHES = 12  # bs

        self.__check()
        self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))

    def __check(self):
        if not torch.cuda.is_available():
            raise ValueError('config.py: cuda is not avalable')
        if not os.path.isdir(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        if not os.path.isdir(self.MODEL_SAVE_DIR):
            os.makedirs(self.MODEL_SAVE_DIR)

    def __add_path(self, path):
        if path not in sys.path:
            sys.path.insert(0, path)

    def initialize(self, module, name, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_ = getattr(self, name)
        module_name = module_.type
        # module_args = edict({"args": module_.args})
        module_args = module_.args

        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)

    def initialize_args(self, module, name, *args, **kwargs):
        """
        只传递一个args 参数
        moudle.name(args, )
        """
        module_ = getattr(self, name)
        module_name = module_.type
        module_args = edict({"args": module_.args})
        # module_args = module_.args

        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)

    def mearge(self, args):
        '''
        使用命令行传参, 增加新参数到config中
        :param args:
        :return:
        '''
        pass

cfg = Configuration()
