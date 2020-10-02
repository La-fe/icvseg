# -*- coding: utf-8 -*-
# @Time    : 2020/9/30
# @Author  : Lafe
# @Email   : wangdh8088@163.com
# @File    : ps_label_to_mask.py

"""
ps 生成mask 为4通道 r, g, b, a.

mask 生成规则:
    取 a == 255 的区域作为mask.

"""
import math
import os
import os.path as osp
import cv2
import numpy as np
import ray
from tqdm import tqdm
import logging

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

@ray.remote
def process(imgps, savep):
    for imgp in tqdm(imgps):
        try:
            img = cv2.imread(imgp, cv2.IMREAD_UNCHANGED)
            h, w, c = img.shape
            assert c == 4, " * error: channel is not 4"
            mask = img[..., -1]

            mask = np.where(mask>=200, 1, 0).astype(np.uint8)
            if 1 not in np.unique(mask):
                continue

            # **** imwrite ****
            basename = osp.basename(imgp)
            target_path = osp.join(savep, basename)
            cv2.imwrite(target_path, mask)
        except Exception as e:
            print(logging.exception(e))

def multi(datas, savep, num_process):
    ray.init(num_cpus=num_process)
    res_id = []
    sub_datas = split_chunks_avg(datas, num_process)
    for sub_data in sub_datas:
        res_id.append(process.remote(sub_data, savep))

    for r_id in res_id:
        ray.get(r_id)

if __name__ == '__main__':
    savep = '/data1/datasets_lafe/mask/skin_segm/mix-cnt20000-size2000-q95-split/mask2'
    root = '/data1/datasets_lafe/mask/skin_segm/mix-cnt20000-size2000-q95-split/mask_image'
    datas = [osp.join(root, i) for i in os.listdir(root)]
    multi(datas, savep, num_process=40)



