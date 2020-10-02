# -*- coding: utf-8 -*-
# @Time    : 2020/8/31
# @Author  : Lafe
# @Email   : wangdh8088@163.com
# @File    : vis_datasets.py

import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm

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


def check_txt_list_add_root(root_dir, txt_f):
    img_dirs = list()
    mask_dirs = list()
    for line in open(txt_f):
        img_dir, mask_dir = line[:-1].split(' ')[0:2] # 后边可能包含 h, w
        if not img_dir.endswith('jpg') or not mask_dir.endswith('png'):
            print("error bug :%s " % img_dir)
            print("error bug :%s " % mask_dir)
            continue
        i = os.path.join(root_dir, img_dir)
        m = os.path.join(root_dir, mask_dir)
        img_dirs.append(i)
        mask_dirs.append(m)
    return img_dirs, mask_dirs

def create_clamap(root, txt_f, savep):
    os.makedirs(savep, exist_ok=True)

    img_path, mpath = check_txt_list_add_root(root, txt_f)
    vis = []
    for ind, (imgname, maskname) in enumerate(tqdm(zip(img_path, mpath), total=len(img_path))):
        img = cv2.imread(imgname)
        mask = cv2.imread(maskname) * 120
        try:
            img, mask = [cv2.resize(i, (512, 512), interpolation=cv2.INTER_NEAREST) for i in [img, mask]]
            color_map = (img * 0.5 + (0, 120, 0)).astype(np.uint8)
            fg = np.where(mask==0, (img*0.7).astype(np.uint8), color_map)

            # **** save image ****
            basename = osp.basename(imgname)
            img_savep = osp.join(savep, basename)
            cv2.imwrite(img_savep, fg)
        except:
            continue

    # loopshow(vis)


def vis_datasets(root, txt_f):
    img_path, mpath = check_txt_list_add_root(root, txt_f)
    vis = []
    for ind, (imgname, maskname) in enumerate(tqdm(zip(img_path, mpath), total=len(img_path))):
        img = cv2.imread(imgname)
        mask = cv2.imread(maskname) * 120
        if ind > 100:
            break
        try:
            img, mask = [cv2.resize(i, (512, 512), interpolation=cv2.INTER_NEAREST) for i in [img, mask]]
            color_map = (img * 0.5 + (0, 120, 0)).astype(np.uint8)
            fg = np.where(mask==0, (img*0.7).astype(np.uint8), color_map)
            vis.extend([img, mask, fg])
        except:
            continue

    loopshow(vis)

if __name__ == '__main__':
    # savep = '/data1/datasets_lafe/mask/CelebA/celebA_skin_clamap'
    savep = '/data1/datasets_lafe/mask/skin_segm/skin_mask/ATR/ATR_skin_clamap'
    # root = '/data1/datasets_lafe/mask/human_alpha/id_crop'
    # txt_f = '/data1/datasets_lafe/mask/human_alpha/layouts/mt_alpha_train.txt'
    #root = '/data1/datasets_lafe/mask/id_human_mt'
    # root = '/data1/datasets_lafe/mask/CelebA'
    root = '/data1/datasets_lafe/mask/skin_segm'
    # txt_f = '/data1/datasets_lafe/mask/coco_lip_mosaic/layouts/precise_03/precise_03_right_train.txt'
    # txt_f = '/data1/datasets_lafe/mask/coco_lip_mosaic/layouts/split_data_precise/unprecise_02_right.txt'
    # txt_f = '/data1/datasets_lafe/mask/CelebA/layout_mt/skin/celebA_skin.txt'
    txt_f = '/data1/datasets_lafe/mask/skin_segm/skin_mask/layouts/ATR.txt'
    # txt_f = '/data1/datasets_lafe/mask/id_human_mt/layouts/id_human_train.txt'
    # txt_f = '/data1/datasets_lafe/mask/skin_segm/skin_mask/layouts/ATR_train.txt'
    # txt_f = '/data1/datasets_lafe/mask/CelebA/layout_mt/mt_CelebA-HQ.txt'
    # vis_datasets(root, txt_f)
    create_clamap(root, txt_f, savep)



