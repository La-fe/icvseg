
import os
import random

def save(save_txt, lines):

    with open(save_txt, 'w') as f:
        for line in lines:
            f.writelines(line)


def split_train_val(saveroot, txt_f, train_ratio=0.8):
    os.makedirs(saveroot, exist_ok=True)
    lines = [line for line in open(txt_f)]
    train_n = int(len(lines) * train_ratio)
    val_n = int(len(lines) - train_n)
    vals = random.sample(lines, val_n)
    trains = set(lines) - set(vals)

    base_txt = os.path.basename(txt_f)
    save_val_txt = "{}/{}_val.txt".format(saveroot, base_txt.split('.txt')[0])
    save(save_val_txt, vals)

    save_train_txt = "{}/{}_train.txt".format(saveroot, base_txt.split('.txt')[0])
    save(save_train_txt, trains)



if __name__ == '__main__':
    root_path = '/data1/datasets_lafe/mask/CelebA/layout_mt/skin'
    txt_f = '/data1/datasets_lafe/mask/CelebA/layout_mt/skin/celebA_skin_cla_pure.txt'
    split_train_val(root_path, txt_f)