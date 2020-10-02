# -*- coding: utf-8 -*-
# @Time    : 2020/9/1
# @Author  : Lafe
# @Email   : wangdh8088@163.com
# @File    : crop_image.py

import cv2
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import ray
import math

cv2.setNumThreads(0)

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

def trans(img):
    dst = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    return dst

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
def crop_from_txt(lines, imgroot, maskroot, imgp, maskp, txt_f=None):
    vis = []
    for line in tqdm(lines):
        line = line.split('\n')[0]
        name, *box = line.split(',')
        x1, y1, x2, y2 = [int(i) for i in box]
        imgname = osp.join(imgroot, name+'.jpg')
        maskname = osp.join(maskroot, name + '.png')
        # img = cv2.imread(imgname)
        # mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(maskname, cv2.IMREAD_UNCHANGED)

        # img = img[y1:y2, x1:x2, :]
        mask = mask[y1:y2, x1:x2]
        # cv2.imwrite(osp.join(imgp, name+'.jpg'), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cv2.imwrite(osp.join(maskp, name+'.png'), mask)
        # mask_vis = cv2.applyColorMap(mask*255, cv2.COLORMAP_JET)
        # vis.append(trans(img))
        # vis.append(trans(mask_vis))
    # loopshow(vis)

def mult_run(txt_f, num_process,
              imgroot, maskroot, imgp, maskp):
    lines = open(txt_f, 'r').readlines()
    sub_lines = split_chunks_avg(lines, num_process=num_process)

    ray.init(num_cpus=num_process)
    res_id = []
    for sub in sub_lines:
        res_id.append(crop_from_txt.remote(sub, imgroot, maskroot, imgp, maskp))
    for r_id in res_id:
        ray.get(r_id)




if __name__ == '__main__':
    txt_f = '/data1/datasets_lafe/mask/human_alpha/img_croped_bboxes.txt'
    imgroot = '/data1/datasets_lafe/mask/human_alpha/img'
    maskroot = '/data1/datasets_lafe/mask/human_alpha/alpha'
    imgp = '/data1/datasets_lafe/mask/human_alpha/id_crop/img'
    maskp = '/data1/datasets_lafe/mask/human_alpha/id_crop/alpha'
    num_process = 95

    # crop_from_txt(imgroot, maskroot, imgp, maskp, txt_f)
    mult_run(txt_f, num_process, imgroot, maskroot, imgp, maskp)
