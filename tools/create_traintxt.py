

import os
import os.path as osp

def create_txtfor_train(root_path, masks_path, imgs_path, save_txt):
    masksp = osp.join(root_path, masks_path)
    imgsp = osp.join(root_path, imgs_path)

    masks = os.listdir(masksp)
    # imgs = os.listdir(imgsp)

    with open(save_txt, 'w') as f:
        for mask in masks:
            lines = '{imgp}/{img} {maskp}/{mask}\n'.format(
                imgp=imgs_path, maskp=masks_path,
                img=mask.replace('.png', '.jpg'), mask=mask)
            f.writelines(lines)

def save(mask_paths, save_txt, imgs_path, masks_path):
    with open(save_txt, 'w') as f:
        for path in mask_paths:
            lines = '{imgp}/{img} {maskp}/{mask}\n'.format(
                imgp=imgs_path, maskp=masks_path,
                img=path.replace('.png', '.jpg'), mask=path)
            f.writelines(lines)

def create_txtfor_train_val(root_path, masks_path, imgs_path, save_txt, train_ratio=0.8):
    masksp = osp.join(root_path, masks_path)
    print(" * mask: ", masksp)
    masks = os.listdir(masksp)
    train_len = int(len(masks) * train_ratio)

    train_masks = masks[0:train_len]
    val_masks = masks[train_len:-1]

    save(masks, save_txt, imgs_path, masks_path)

    save_train_txt = "{}_train.txt".format(save_txt.split('.txt')[0])
    save(train_masks, save_train_txt, imgs_path, masks_path)

    save_val_txt = "{}_val.txt".format(save_txt.split('.txt')[0])
    save(val_masks, save_val_txt, imgs_path, masks_path)


if __name__ == '__main__':
    """
    mask abspath: root_path + masks_path
    images abspath:  root_path + imgs_path
    
    """

    masks_path = 'mask'
    imgs_path = 'image'
    root_path = '/data1/datasets_lafe/mask/skin_segm/mix-cnt20000-size2000-q95-split'

    save_txt = '/data1/datasets_lafe/mask/skin_segm/mix-cnt20000-size2000-q95-split/layouts/mt_faceskin_mix_cnt20000.txt'
    # create_txtfor_train(root_path, masks_path, imgs_path, save_txt)
    create_txtfor_train_val(root_path, masks_path, imgs_path, save_txt)