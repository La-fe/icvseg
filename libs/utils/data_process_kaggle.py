#! -*- coding:utf-8 -*-
"""sjk modified test"""
import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# import mmcv



Image.MAX_IMAGE_PIXELS = 100000000000

class Process:
    def __init__(self):
        self.label_list_dir = [
            #'/home/xjx/data/mask/Kaggle/rematch/jingwei_round2_train_20190726/image_10_label.png',
            #'/home/xjx/data/mask/Kaggle/rematch/jingwei_round2_train_20190726/image_11_label.png',
            #'/home/xjx/data/mask/Kaggle/rematch/jingwei_round2_train_20190726/image_20_label.png',
            '/home/xjx/data/mask/Kaggle/rematch/jingwei_round2_train_20190726/image_21_label.png',
                      ]

        self.img_list_dir = [
            #'/home/xjx/data/mask/Kaggle/rematch/jingwei_round2_train_20190726/image_10.png',
            #'/home/xjx/data/mask/Kaggle/rematch/jingwei_round2_train_20190726/image_11.png',
            #'/home/xjx/data/mask/Kaggle/rematch/jingwei_round2_train_20190726/image_20.png',
            '/home/xjx/data/mask/Kaggle/rematch/jingwei_round2_train_20190726/image_21.png',
                    ]

        self.img_test_list_dir = ['','']

        self.label_list_np = ['/home/lzhpc/kk/dataset/seg/np_file/label_0.npy',
                              '/home/lzhpc/kk/dataset/seg/np_file/label_1.npy']

        self.img_list_np = ['/home/lzhpc/kk/dataset/seg/np_file/image_0.npy',
                            '/home/lzhpc/kk/dataset/seg/np_file/image_1.npy']

        # self.label_list, self.img_list = None, None
        self.label_list, self.img_list = self._create_data()
        # self.label_list, self.img_list = self._create_data_np()

    def _create_test_data(self):
        img_list = []
        for i in range(len(self.img_test_list_dir)):
            img = Image.open(self.img_test_list_dir[i])
            img = np.asarray(img)[..., 0:3]
            img_list.append(img)
        return img_list


    def _create_data(self):
        label_list = []
        img_list = []
        for i in range(len(self.img_list_dir)):
            anno_map = Image.open(self.label_list_dir[i])
            img = Image.open(self.img_list_dir[i])
            anno_map = np.asarray(anno_map)
            img = np.asarray(img)
            print(img.shape)
            img = img[..., 0:3]
            label_list.append(anno_map)
            img_list.append(img)
        return label_list, img_list

    def _create_data_np(self):
        label_list = []
        img_list = []
        for i in range(len(self.label_list_np)):
            anno_map = np.load(self.label_list_np[i])
            img = np.load(self.img_list_np[i])
            label_list.append(anno_map)
            img_list.append(img)
        return label_list, img_list


    def data_crop(self, ratios=None):
        if ratios is None:
            ratios = [1, 2, 4, 5, 10]
        for i in range(len(self.img_list)):
            img = self.img_list[i]
            label = self.label_list[i]
            h, w, c = img.shape
            for ratio in ratios:
                save_name = 'image_%d_ratio%d' % (i, ratio)
                label_resize = cv2.resize(label, dsize=(w//ratio, h//ratio), interpolation=cv2.INTER_NEAREST)
                img_resize = cv2.resize(img, dsize=(w//ratio, h//ratio))
                self.convert(img_resize, label_resize, save_name, stride=500, img_size=512)


    def convert(self, img, label, img_name, stride=500, img_size=512):

        root_path = '/home/xjx/data/mask/Kaggle/rematch/data'
        save_path = os.path.join(root_path, img_name)
        img_save_path = os.path.join(save_path, 'img')
        label_save_path = os.path.join(save_path, 'label')
        if not os.path.exists(save_path):
            os.makedirs(img_save_path)
            os.makedirs(label_save_path)

        height, width, c = img.shape
        layout_file = open('%s/layout.txt' % save_path, '+w')
        ind = 0 # index
        for  x1 in tqdm(range(0, width, stride)):
            for y1 in range(0, height, stride):
                x2, y2 = x1 + img_size, y1 + img_size
                if x2 >= width:
                    x2, x1 = width , width - img_size
                if y2 >= height:
                    y2, y1 = height, height - img_size

                img_roi = img[y1:y2, x1:x2, 0:3]
                label_roi = label[y1:y2, x1:x2]

                if len(np.flatnonzero(img_roi)) / (img_size * img_size) < 0.5:  # 图像占比0.5才保存图像
                    continue
                if len(np.flatnonzero(label_roi)) < 25: # 少于25 个像素剔除
                    continue

                cv2.imwrite('%s/%06d.jpg' % (img_save_path, ind),img_roi )
                cv2.imwrite('%s/%06d.png' % (label_save_path, ind),label_roi )
                layout_file.write('%s/%06d.jpg %s/%06d.png\n' % (img_save_path.split(root_path+'/')[-1], ind,
                                                             label_save_path.split(root_path+'/')[-1], ind))
                ind += 1
        layout_file.close()


if __name__ == '__main__':
    datasets  = Process()
    #datasets.label_list, datasets.img_list = datasets._create_data() # 创建训练数据
    datasets.data_crop()

    z = 'stop'

# def maskAddImg(img, mask):
#     '''
#     bgr
#     1 : 烤烟，蓝色（255,0,0），
#     2：玉米，黄色（0,255,255），
#     3：薏仁米，绿色（0,0,255）
#     '''
#     h, w, c = img.shape
#     black = np.zeros((h, w, c), dtype=np.uint8)
#     color = np.asarray([[255,0,0], [0,255,255], [0,0,255]])
#     black[mask == 1] = color[0]
#     black[mask == 2] = color[1]
#     black[mask == 3] = color[2]
#
#     # mask_img_n = np.stack((mask, mask, mask), axis=2)
#     mix_img = cv2.addWeighted(img, 0.5, black, 0.5, 1)
#     return mix_img
