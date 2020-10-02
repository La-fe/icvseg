import os
import json
import torch
import cv2
from torchvision import transforms
import numpy as np
import PIL


class ImgTestDataset:
    def __init__(self,opt ):
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        self.normalize = transforms.Normalize(
            mean=[102.9801, 115.9465, 122.7717],
            std=[1., 1., 1.])


    def __call__(self, img):
        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image transform
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = ''
        return output

    def img_transform(self, img):
        # image to float
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p

    def __len__(self):
        return 1