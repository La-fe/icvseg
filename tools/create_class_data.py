# -*- coding: utf-8 -*-
# @Time    : 2020/9/2
# @Author  : Lafe
# @Email   : wangdh8088@163.com
# @File    : create_class_data.py

from easydict import EasyDict as edict
import os
import os.path as osp
from tqdm import tqdm
import cv2
import numpy as np
import ray
import math



"""
0: 'background' 1: 'skin' 2: 'nose'
3: 'eye_g' 4: 'l_eye' 5: 'r_eye'
6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
9: 'r_ear' 10: 'mouth' 11: 'u_lip'
12: 'l_lip' 13: 'hair' 14: 'hat'
15: 'ear_r'  # 耳环``
16: 'neck_l' 17: 'neck'
18: 'cloth'  

skin datasets: 1,2,3,4,5,6,7,8,9,10,11,12
"""

def split_chunks_avg(data, num_process):
    '''
        平均分配list 到不同的 process 上
        e.g. sum_num = 100, num_process = 7
            every = 100 / 7 = 14.285
            every_plus_one =  0.285 * 7 = 1.995 , ceil(1.995) = 2
            chunks_list = [1,1,0,0,0,0,0]  len=7
            if chunks_list item is 1:
                step = 15
            elif chunks_list item is 0:
                step = 14
            return [0:15] [15:30] [30:44] [44:58] ...
    '''

    def chunks(lst, chunksize, step):
        start = 0
        end = step
        i = 0
        sum_num = len(lst)
        for flag in chunksize:
            if flag == 0:
                start = i
                end = i + step
                end = min(end, sum_num)
                i = end
            elif flag == 1:
                start = i
                i = i + 1
                end = i + step
                i = end
            yield lst[start: end]

    sum_num = len(data)
    every = sum_num / float(num_process)
    every_plus_one = math.ceil((every - int(every)) * float(num_process))
    chunks_list = np.zeros(num_process, dtype=np.int8)
    chunks_list[0:every_plus_one] += 1
    data_list = list(chunks(data, chunks_list, int(every)))
    return data_list


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

def cla_map(mask):
    nmask = np.where((mask >= 1) & (mask <= 12) | (mask == 17), 1, 0)

    return nmask.astype(np.uint8)


@ray.remote
def create_new_class_mask(masks, nmaskp=None, flag_show=False):
    num = len(masks)
    # num = 100
    vis = []
    for mname in tqdm(masks[0:num]):
        mask = cv2.imread(mname, cv2.IMREAD_UNCHANGED)
        nmask = cla_map(mask)

        if flag_show:
            heat = cv2.applyColorMap(nmask*255, cv2.COLORMAP_JET)
            vis.append(heat)

        else:
            nname = osp.join(nmaskp, osp.basename(mname))
            cv2.imwrite(nname, nmask)

    if flag_show:
        loopshow(vis)

def multi_process(maskp, nmaskp, num_process):
    os.makedirs(nmaskp, exist_ok=True)
    ray.init(num_cpus=num_process)
    masks = [osp.join(maskp, i) for i in os.listdir(maskp)]
    subdatas = split_chunks_avg(masks, num_process)
    res_id = []
    for sub in subdatas:
        res_id.append(create_new_class_mask.remote(sub, nmaskp))
    for r_id in res_id:
        ray.get(r_id)


if __name__ == '__main__':
    maskp = '/data1/datasets_lafe/mask/CelebA/CelebAMaskHQ-mask'
    nmaskp = '/data1/datasets_lafe/mask/CelebA/CelebA_skin_mask'
    num_process = 95
    multi_process(maskp, nmaskp, num_process)
    # create_new_class_mask(maskp)










