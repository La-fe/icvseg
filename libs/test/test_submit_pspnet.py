import torch
import numpy as np
from tqdm import tqdm
from libs.datasets.generateData import generate_img_test
from libs.datasets.PspnetDataset import   ImgTestDataset
from libs.net.UperNet.models import ModelBuilder, SegmentationModule
import torch.nn as nn
# from experiment.deeplabv3_rematch_r1.config import cfg
import libs.net.generateNet as net_gener
import torch.nn.functional as F
import cv2
from PIL import Image
import PIL
Image.MAX_IMAGE_PIXELS = 2000000000
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import sys
import os
import  argparse
from defaults import _C as cfg
torch.cuda.set_device(0) # 设置主gpu
import collections

from torch.autograd import Variable

from scipy.io import loadmat

# colors = loadmat('data/color150.mat')['colors']

def async_copy_to(obj, dev, main_stream=None):
    if torch.is_tensor(obj):
        v = obj.cuda(dev, non_blocking=True)
        if main_stream is not None:
            v.data.record_stream(main_stream)
        return v
    elif isinstance(obj, collections.Mapping):
        return {k: async_copy_to(o, dev, main_stream) for k, o in obj.items()}
    elif isinstance(obj, collections.Sequence):
        return [async_copy_to(o, dev, main_stream) for o in obj]
    else:
        return obj


def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='BGR'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def visualize_result(data, pred, cfg):
    (img, info) = data
    imgs_dict = dict()
    # print predictions in descending order
    pred = np.int32(pred)

    if False:
        pixs = pred.size
        uniques, counts = np.unique(pred, return_counts=True)
        print("Predictions in [{}]:".format(info))
        for idx in np.argsort(counts)[::-1]:
            name = names[uniques[idx] + 1]
            ratio = counts[idx] / pixs * 100
            if ratio > 0.1:
                print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    # pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    # im_vis = np.concatenate((img, pred_color), axis=1)

    # pred_mask = cv2.addWeighted(img, 0.7, pred_color, 0.5, 0)
    # im_vis = np.concatenate((pred_color, pred_mask ), axis=1)

    img_name = info.split('/')[-1]
    # imgs_dict['raw'] = img
    # imgs_dict['add'] = pred_mask
    # imgs_dict['mask'] = pred_color
    imgs_dict['pred'] = pred
    return imgs_dict


def test(segmentation_module, batch_data, gpu):
    segmentation_module.eval()

    # process data
    segSize = (batch_data['img_ori'].shape[0],
               batch_data['img_ori'].shape[1])
    img_resized_list = batch_data['img_data']

    with torch.no_grad():
        scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
        scores = async_copy_to(scores, gpu)

        for img in img_resized_list:
            feed_dict = batch_data.copy()
            feed_dict['img_data'] = img
            del feed_dict['img_ori']
            del feed_dict['info']
            feed_dict = async_copy_to(feed_dict, gpu)

            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

        _, pred = torch.max(scores, dim=1)
        pred = as_numpy(pred.squeeze(0).cpu())

    # visualization
    return visualize_result(
        (batch_data['img_ori'], batch_data['info']),
        pred,
        cfg
    )




class NetLoader:
    def __init__(self, cfg,
                 ckpt='',
                 num_classes=5, flip=False, multi_scale=1, flag_dataparall=True):
        self.cfg = cfg
        self.cfg.TEST_CKPT = ckpt
        self.cfg.MODEL_NUM_CLASSES = num_classes
        self.cfg.TEST_FLIP = flip
        self.flag_dataparall = flag_dataparall
        if isinstance(multi_scale, int):
            self.cfg.TEST_MULTISCALE = [multi_scale]
        elif isinstance(multi_scale, list):
            self.cfg.TEST_MULTISCALE  = multi_scale



        self.net = self.model_init(self.cfg)
        self.num_class = num_classes



    def model_init(self, cfg):
        net = cfg.initialize_args(net_gener, 'INIT_model')

        if cfg.TEST_CKPT is None:
            raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')
        print('start loading model %s' % cfg.TEST_CKPT)

        device = torch.device(0)
        model_dict = torch.load(cfg.TEST_CKPT, map_location=device)  # 在其他gpu训练需要用map搞到测试gpu上
        from collections import OrderedDict
        # 如果使用单卡跑的模型，关闭以下model清洗
        # ----------------------------------------
        if not self.flag_dataparall :
            net.load_state_dict(model_dict)
        else:
            new_model_dict = OrderedDict()
            mod = net.state_dict()
            for k, v in model_dict.items():
                if k[7:] in mod.keys():
                    name = k[7:]  # remove module.
                    new_model_dict[name] = v

            net.load_state_dict(new_model_dict)

        # ----------------------------------------
        # net.load_state_dict(model_dict)

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
        # 特征可视化
        # func.vis_mask_feat(multi_avg, True, row_batched, col_batched)


        score_map, mask = torch.max(multi_avg, dim=1)
        score_map_cpu = score_map.cpu().detach().numpy().astype(np.float16)[0, ...]
        mask_cpu = mask.cpu().numpy().astype(np.uint8)[0, :, :]
        mask_img = cv2.resize(mask_cpu, dsize=(col_batched, row_batched), interpolation=cv2.INTER_NEAREST)

        score_map_cpu = Image.fromarray(score_map_cpu.astype(np.float32))
        score_map_cpu = np.asarray(score_map_cpu.resize((col_batched, row_batched)))


        return mask_img, score_map_cpu


def predict(d, model, outputname='tmp.bmp'):
    wy, wx ,c = d.shape  # width
    print(wx, wy)
    od  = np.zeros((wy, wx), np.uint8)
    blocksize = 512
    step = 256
    for cy in tqdm(range(step, wy - blocksize, step)):
        for cx in range(step, wx - blocksize, step):
            x1,y1,x2,y2 = cx - step, cy - step, cx - step + blocksize, cy - step+blocksize
            img = d[y1:y2, x1:x2, 0:3] #[0:3, :, :]  # channel*h*w
            if (img.sum() == 0): continue
            img_tensor = imgTensor(img)
            r, _= model[0](img_tensor)

            od[cy - step // 2:cy + step // 2, cx - step // 2:cx + step // 2] = r[128:step + 128, 128:step + 128]
            # print(cy, cx)
    cv2.imwrite('%s/%s'%('/home/xjx/data/mask/Kaggle/submit/submit', outputname), od)
    return


def ndindex_numpy(x, y ):
    xs = []
    ys = []
    for i in np.ndindex(x, y):
        xs.append(i[0])
        ys.append(i[1])
    return xs, ys

def predict_pyramid(d, model, flag_save_score=False, outputname='image_3', file_name='analy', save_path='', root_path=''):
    wy, wx ,c = d.shape  # width
    print(wx, wy)
    feat_pyramid = []
    all_mask  = np.zeros((wy, wx), np.uint8)
    all_score_map  = np.zeros((wy, wx), np.float16)
    blocksize_list = [ 512] # 不同尺度滑动窗口 扫描测试图片
    step_list = [256]

    # xs, ys = ndindex_numpy(wy, wx)
    for step, blocksize in zip(step_list, blocksize_list):
    # blocksize = 512
    # step = 256
        for cy in tqdm(range(step, wy - blocksize, step)):
            for cx in range(step, wx - blocksize, step):
                x1,y1,x2,y2 = cx - step, cy - step, cx - step + blocksize, cy - step+blocksize
                img = d[y1:y2, x1:x2, 0:3] #[0:3, :, :]  # channel*h*w
                if (img.sum() == 0): continue # 全0 跳过
                img_tensor = imgTensor(img)
                # mask, score_map = test(img_tensor)
                pred = test(model, img_tensor, args.gpu)
                mask = pred['pred']

                all_mask[cy - step // 2:cy + step // 2, cx - step // 2:cx + step // 2] = mask[step//2:step + step//2, step//2:step + step//2]
        # feat_pyramid.append([all_mask, all_score_map])

        if save_path == '':
            save_path = '%s/submit/%s/%d_file'%(root_path, file_name, blocksize)
        else:
            save_path = '%s/%s/%d_file'%(save_path, file_name, blocksize)

        if  not os.path.exists(save_path):
            os.makedirs(save_path)

        # 保存 mask  png图
        print(all_mask.shape)
        cv2.imwrite('%s/%s.png' % (save_path, outputname), all_mask)
        if flag_save_score:
        # 保存 mask np文件
            np.save('%s/%s'%(save_path, '%s_%d_all_mask'%(outputname, blocksize)), all_mask)
        # 保存 mask score 图
            np.save('%s/%s'%(save_path, '%s_%d_all_score_map'%(outputname, blocksize)), all_score_map)



        del all_mask
        del all_score_map
        all_mask  = np.zeros((wy, wx), np.uint8)
        all_score_map  = np.zeros((wy, wx), np.float16)
    # feat_pyramid = np.asarray(feat_pyramid)
    # mask_map = np.stack(feat_pyramid[:, 0], axis=2) # h, w, c
    # score_map = np.stack(feat_pyramid[:, 1], axis=2) # h, w, c
    # ind = np.argmax(score_map, axis=2)
    # mask = mask_map[xs, ys, ind.flatten(), ]
    # mask = mask.reshape(wy, wx)
    z = 1

            # print(cy, cx)
    # cv2.imwrite('%s/%s'%('/home/xjx/data/mask/Kaggle/submit/submit', outputname), mask)
    return



def maskAddImg(img, mask):
    '''
    bgr
    1 : 烤烟，蓝色（255,0,0），
    2：玉米，黄色（0,255,255），
    3：薏仁米，绿色（0,0,255）
    '''

    h, w, c = img.shape
    black = np.zeros((h, w, c), dtype=np.uint8)
    color = np.asarray([[255,0,0], [0,255,255], [0,0,255],[0,128,128]])
    black[mask == 1] = color[0]
    black[mask == 2] = color[1]
    black[mask == 3] = color[2]
    black[mask == 4] = color[3]

    # mask_img_n = np.stack((mask, mask, mask), axis=2)
    mix_img = cv2.addWeighted(img, 0.7, black, 0.3, 1)
    return mix_img



def show_pre_mix(img, ratio, label_path, flag_show=True, save_p=None):
    '''

    :param img:
    :param ratio:  缩小比率
    :param label_path:
    :return:
    '''
    # label = Image.open(label_path)
    # label_np = np.asarray(label)
    if isinstance(label_path, str):
        label_np = np.load(label_path)
    elif isinstance(label_path, PIL.PngImagePlugin.PngImageFile):
        label_np = np.asarray(label_path)
    else:
        label_np = label_path

    h, w = label_np.shape
    label_r10 = cv2.resize(label_np, dsize=(w // ratio, h // ratio), interpolation=cv2.INTER_NEAREST)
    img_r10 = cv2.resize(img, dsize=(w // ratio, h // ratio))
    mix_img = maskAddImg(img_r10, label_r10)
    mix_img_rgb = cv2.cvtColor(mix_img, cv2.COLOR_BGR2RGB)
    if flag_show:
        plt.imshow(mix_img_rgb)
        plt.show()
    if save_p is not None:
        cv2.imwrite(save_p, mix_img_rgb)
    z = 'strop'

# def show_png(img, reteo)


def config_choise(args, flag=9):
    '''
    0:    mobilenetv2  dilated-c1_deepsup
    1:    resnet18 dilated-ppm_deepsup
    2:    resnet50 dilated-ppm_deepsup
    3:    resnet101-upernet
    4:    resnet101 dilated-ppm_deepsup
    '''
    cfg_list = [
        # 0
        'config/ade20k-mobilenetv2dilated-c1_deepsup.yaml',
        # 1
        'config/ade20k-resnet18dilated-ppm_deepsup.yaml',
        # 2
        'config/ade20k-resnet50dilated-ppm_deepsup.yaml',
        # 3
        'config/ade20k-resnet101-upernet.yaml',
        # 4
        'config/ade20k-resnet101dilated-ppm_deepsup.yaml',
        #５ remo psp
        'config/remo_ade20k-resnet50dilated-ppm_deepsup.yaml',
        # 6 remo uper
        'config/ade20k-resnet50-upernet.yaml',
        # 7 remo person stuff
        'config/remo_ade20k-resnet50dilated-ppm_deepsup_person_stuff.yaml',
        # 8 18
        'config/remo_ade20k-resnet18dilated-ppm_deepsup.yaml',
        # 9
        '/home/zhangming/work/mask/scene/semantic-segmentation-pytorch/config/remo_ade20k-resnet50dilated-ppm_deepsup_kk.yaml',


    ]
    weight_list = [
        # 0
        '',
        # 1
        'Module/baseline-resnet18dilated-ppm_deepsup/decoder_epoch_20.pth',
        # 2
        'Module/baseline-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
        # 3
        'Module/baseline-resnet101-upernet/decoder_epoch_50.pth',
        # 4
        # 'Module/baseline-resnet50-upernet/decoder_epoch_30.pth',
        'Module/baseline-resnet101dilated-ppm_deepsup/decoder_epoch_25.pth',
        # 5  remo psp
        'Module/resnet50_ade20k_20/decoder_epoch_20.pth',
        # 6 remo uper
        'Module/resnet50-upernet_40/decoder_epoch_40.pth',
        # 7 remo person stuff
        'Module/ade20k-resnet50dilated-ppm_deepsup_person_stuff/decoder_epoch_40.pth',
        # 8
        'Module/baseline-resnet18dilated-ppm_deepsup/decoder_epoch_20.pth',
        # 9
        'remo_ade20k-resnet50dilated-ppm_deepsup_kk/decoder_epoch_15.pth',

    ]


    args.gpu = 0
    args.cfg = cfg_list[flag]
    args.imgs = '/home/xjx/data/videos/mask/save_img'

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    cfg.MODEL.weights_encoder = weight_list[flag].replace('decoder', 'encoder')
    cfg.MODEL.weights_decoder = weight_list[flag]

    abs_path = '/home/zhangming/work/mask/scene/semantic-segmentation-pytorch/ckpt'
    cfg.MODEL.weights_encoder = os.path.join(abs_path, cfg.MODEL.weights_encoder)
    cfg.MODEL.weights_decoder = os.path.join(abs_path, cfg.MODEL.weights_decoder)
    cfg.list_test = ['x']


    # cfg.DATASET.imgSizes=  (300, 375, 450, 525, 600)
    cfg.DATASET.imgSizes=  [450]
    cfg.DATASET.imgMaxSize = 1000
    cfg.DATASET.random_flip = False
    return args, cfg

def init_mode(cfg, gpu):

    torch.cuda.set_device(gpu)
    print('start main')
    # Network Builders
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    return segmentation_module


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="remo_unet_oc.py",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0, 0",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--ckpt",
        default="",
        type=str,
    )

    args = parser.parse_args()

    # import config
    # sys.path.insert(0, "config")
    # try:
    #     config_name, ext = os.path.splitext(os.path.basename(args.cfg))
    #     config_file = __import__(config_name)
    # except ImportError:
    #     raise ("not find config")

    # cfg = config_file.cfg
    args, cfg = config_choise(args)

    imgTensor = ImgTestDataset(cfg.DATASET)

    flag_show = False  # 可视化图片
    flag_pred = True  # 全图扫描
    flag_com = False   # 多尺度融合
    name = 'PSPNet_resnet50_kk' # 测试文件夹名(自动创建)
    root_path = '/home/zhangming/work/mask/deeplab_jingwei/submit'
    test_file = "/home/zhangming/Datasets/mask/Kaggle/rematch/jingwei_round2_test_a_20190726"
    if flag_show:
        num_pred = 3
        ds = np.load('%s/submit/image_%d.npy' % (root_path, num_pred)) # image3  or image4的numpy 文件，进行可视化
        f = Image.open('%s/submit/%s/512_file/image_%d_predict.png' %(root_path, name, num_pred))
        # f = Image.open('/home/xjx/data/mask/Kaggle/submit/analys_3000data/512_1024_mix_file/image_4_predict.png' )
        show_pre_mix(ds, 5, f)

    elif flag_pred:
        '''
            'deeplabv3plus', 'res101_atrous'
            'unet'
        '''

        model = init_mode(cfg, args.gpu)
        model.cuda()
        # predict_pyramid(np.random.randint(0, 255, (5000, 5000, 3), dtype=np.uint8), model, 'xx')
        ds = np.load('%s/image_3.npy' % (test_file))
        print(ds.shape)
        predict_pyramid(ds, model, flag_save_score=False, outputname='image_3_predict', file_name=name, root_path=root_path)
        del ds
        ds = np.load('%s/image_4.npy' % (test_file))
        print(ds.shape)
        predict_pyramid(ds, model,flag_save_score=False,outputname='image_4_predict', file_name=name, root_path=root_path)

        num_preds = [3, 4]
        for num_pred in num_preds:
            ds = np.load('%s/image_%d.npy' % (test_file, num_pred)) # image3  or image4的numpy 文件，进行可视化
            f = Image.open('%s/submit/%s/512_file/image_%d_predict.png' %(root_path, name, num_pred))
            
            show_pre_mix(ds, 5, f, False, '%s/submit/%s/512_file/mix_%d_predict.jpg' %(root_path, name, num_pred))

    elif flag_com:
        '''
            选取 尺度A和B的 mask 和score图 ，
            
            切割后进行比较替换，不支持>=3个特征图 
        '''
        img_num = 3  # 测试图片id  image_3.png
        pred_np = [
            [
            '%s/submit/%s/512_file/image_%d_512_all_mask.npy' %(root_path, name, img_num),
            '%s/submit/%s/512_file/image_%d_512_all_score_map.npy' %(root_path, name, img_num),
            ],
            [
            '%s/submit/%s/1024_file/image_%d_1024_all_mask.npy' % (root_path, name, img_num),
            '%s/%s/1024_file/image_%d_1024_all_score_map.npy' %(root_path, name, img_num),
            ],
            [
            '%s/submit/%s/2048_file/image_%d_2048_all_mask.npy' % (root_path, name, img_num),
            '%s/submit/%s/2048_file/image_%d_2048_all_score_map.npy' %(root_path, name, img_num),
            ],
        ]


        # 0 代表mask图， 1 代表score图
        img_512_mask = np.load(pred_np[0][0])
        img_512_score = np.load(pred_np[0][1])

        img_1024_mask = np.load(pred_np[1][0])
        img_1024_score = np.load(pred_np[1][1])


        h, w = img_512_mask.shape
        mask = np.zeros((h, w), np.uint8)

        for i in tqdm(range(0, w, 1024)):
            for j in range(0, h, 1024):
                sw = 1024
                sh = 1024
                if i + sw >= w:
                    sw = w
                if j + sh >= h:
                    sh = h

                mask_ind = np.argmax((img_512_score[j:j + sh, i:i + sw], img_1024_score[j:j + sh, i:i + sw]), axis=0)
                mask[j:j+sh, i:i+sw] = np.where(mask_ind==0, img_512_mask[j:j+sh, i:i+sw], img_1024_mask[j:j+sh, i:i+sw])

        mix_path = '%s/submit/%s/512_1024_mix_file' % (root_path, name)
        if not os.path.exists(mix_path):
            os.makedirs(mix_path)

        cv2.imwrite('%s/image_%d_predict.png' % (mix_path, img_num), mask)

z = 1
