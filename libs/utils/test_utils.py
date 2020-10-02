import cv2
import numpy as np
import math
import h5py
import torch
import matplotlib.pyplot as plt


def get_pad169size(image_h, image_w):
    img_width_new = int(max(float(image_h) * 16.0 / 9.0, image_w))
    img_height_new = int(max(float(image_w) * 9.0 / 16.0, image_h))
    return img_height_new, img_width_new


def get_pad916size(image_h, image_w):
    img_width_new = int(max(float(image_h) * 9.0 / 16.0, image_w))
    img_height_new = int(max(float(image_w) * 16.0 / 9.0, image_h))
    return img_height_new, img_width_new


def putImgsToOne(imgs, strs, ncols, txt_org=(20, 10), fonts=1, color=(0, 0, 255)):
    # imgs: list of images
    # strs: strings put on images to identify different models,len(imgs)==len(strs)
    # ncols: columns of images, the code will computed rows according to len(imgs) and ncols automatically
    # txt_org: (xmin,ymin) origin point to put strings
    # fonts: fonts to put strings
    # color: color to put stings
    w_max_win = 1400
    h_max_win = 980
    img_h_max = -1
    img_w_max = -1
    if len(imgs) != len(strs):
        diff_len = int(math.fabs(len(imgs) - len(strs)))
        for i in range(diff_len):
            strs.append('%d' % i)
        # print('！！！！！！！！ error len of imgs != len ot str in putImgsToOne')


    for i in range(len(imgs)):
        h, w = imgs[i].shape[:2]
        if h > img_h_max:
            img_h_max = h
        if w > img_w_max:
            img_w_max = w

    if len(imgs) < ncols:
        ncols = len(imgs)
        n_rows = 1
    else:
        n_rows = int(math.ceil(float(len(imgs)) / float(ncols)))
    x_space = 5
    y_space = 5

    img_one = np.zeros(
        (img_h_max * n_rows + (n_rows - 1) * y_space, img_w_max * ncols + (ncols - 1) * x_space, 3)).astype(np.uint8)
    img_one[:, :, 0] = 255
    cnt = 0
    for i_r in range(n_rows):
        for i_c in range(ncols):
            if cnt > len(imgs) - 1:
                break
            xmin = i_c * (img_w_max + x_space)
            ymin = i_r * (img_h_max + y_space)
            img = imgs[cnt]
            cv2.putText(img, "%d_" % (cnt + 1) + strs[cnt], txt_org, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=fonts,
                        color=color, thickness=1)
            img_h_cur, img_w_cur, _ = img.shape
            # print i_r,i_c,"img_h_cur:", img_h_cur, "img_w_cur:", img_w_cur, "img_w_max:", img_w_max, "img_h_max:", img_h_max
            xmax = xmin + img_w_cur
            ymax = ymin + img_h_cur
            # print i_r, i_c,xmin,xmax,ymin,ymax
            img_one[ymin:ymax, xmin:xmax, :] = img
            cnt += 1
    scale_x = float(w_max_win) / float(img_w_max * ncols)
    scale_y = float(h_max_win) / float(img_h_max * n_rows)
    scale = min(scale_x, scale_y)
    img_one = cv2.resize(img_one, dsize=None, fx=scale, fy=scale)
    return img_one


def read_net(state_dict, path):
    f = h5py.File(path, 'r')
    for i in f:
        if state_dict[i].is_cuda:
            state_dict[i] = torch.Tensor(f[i][:]).cuda()
        else:
            state_dict[i] = torch.Tensor(f[i][:])
    f.close()
    return state_dict


def nothing(emp):
    pass

def do_mosaic(frame, x, y, w, h, neighbor=9):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    :param frame: opencv frame
    :param int x :  马赛克左顶点
    :param int y:  马赛克右顶点
    :param int w:  马赛克宽
    :param int h:  马赛克高
    :param int neighbor:  马赛克每一块的宽
    """
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        return
    for i in range(0, h - neighbor, neighbor):  # 关键点0 减去neightbour 防止溢出
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = frame[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
            cv2.rectangle(frame, left_up, right_down, color, -1)
    return frame


def vis_mask_feat(raw_img, predicts, flag_show=False, flag_crf=False, row=512, col=512):
    pre = predicts.cpu().detach().numpy()
    pre_0 = pre[:, 0, ...]
    pre_1 = pre[:, 1, ...]
    pre_0 = pre_0[0, ...]
    pre_1 = pre_1[0, ...]
    pre_0_t = (pre_0 * 5 + 128).astype(np.uint8)
    pre_1_t = (pre_1 * 5 + 128).astype(np.uint8)
    pre_0_img = cv2.applyColorMap(pre_0_t, cv2.COLORMAP_JET)
    pre_1_img = cv2.applyColorMap(pre_1_t, cv2.COLORMAP_JET)
    p0 = cv2.resize(pre_0_img, dsize=(col, row), interpolation=cv2.INTER_NEAREST)
    p1 = cv2.resize(pre_1_img, dsize=(col, row), interpolation=cv2.INTER_NEAREST)
    # if flag_crf:
    #     crf_mask = dense_crf(raw_img, pre)

    if flag_show:
        cv2.namedWindow('0', 0)
        p_all = np.concatenate((p0, p1), axis=1) # 拼接两张图
        cv2.imshow("0", p_all)
        # cv2.imshow("1", p1)
        # k = cv2.waitKey(0)
        # if k == ord('q'):
        #     cv2.destroyAllWindows()
    pass

def vis_Image_from_iou(im):
    flag_fromCenter = False  # 鼠标以矩形中心为基准
    showCrosshair = True  # 显示网格
    r = cv2.selectROI(im, showCrosshair=showCrosshair, fromCenter=flag_fromCenter)  # return tuple
    z = list(map(int, list(r)))  # box_center x1, y1, w, h
    box_corner = [z[0], z[1], z[2] + z[0], z[3] + z[1]]  # center 2 corner
    imCrop = im[box_corner[1]:box_corner[3], box_corner[0]:box_corner[2], ...]

    return imCrop

def VisEdges(mask, flag_canny=False):
    mask = mask.astype(np.uint8)
    if flag_canny:
        back = cv2.Canny(mask, 0, 1) / 255
        return back.astype(np.byte)

    lEdgesSet = list()
    rEdgesSet = list()

    h, w = mask.shape
    back = np.zeros((h, w))

    for j in range(h):
        for i in range(w):
            if mask[(j, i)] == 1 and mask[(j, i - 1)] == 0: # 0,1 为1
                back[(j, i)] = 1
            if mask[(j, i)] == 0 and mask[(j, i - 1)] == 1: #   0,1
                back[(j, i-1)] = 1

            if mask[(j, i)] == 1 and mask[(j-1, i)] == 0:
                back[(j, i)] = 1
            if mask[(j, i)] == 0 and mask[(j-1, i)] == 1:
                back[(j-1, i)] = 1
    return back
    # plt.subplot(121), plt.imshow(black)
    # # plt.subplot(122), plt.imshow(edges)
    # plt.imshow()

def VisChangeBack(mask, person, flag_canny=False):
    import matplotlib.pyplot as plt

    # mask must 单通道， np.byte
    mask = mask.astype(np.byte)
    edge1 = VisEdges(mask, flag_canny) # canny 使用 uint8
    edge1[edge1 >= 1] = 1
    edge1 = edge1.astype(np.byte)

    mask_noedge1 = mask - edge1
    edge2 = VisEdges(mask_noedge1, flag_canny)

    mask_noedge2 = mask_noedge1 - edge2
    edge2 = edge1 + edge2

    back = cv2.imread("/home/xjx/Pictures/Cg-4V1Cf7B6IH7qFAAdeFxmCJTMAABmTgMFauUAB14v678.jpg")

    pre_noedge2 = cv2.bitwise_and(person, person, mask=mask_noedge2.astype(np.uint8)) # 前景 没边
    pre = cv2.bitwise_and(person, person, mask=mask.astype(np.uint8)) # 前景

    # back = cv2.cvtColor(back, cv2.COLOR_BGR2RGB)
    # pre_noedge2 = cv2.cvtColor(pre_noedge2, cv2.COLOR_BGR2RGB)
    # pre_edge2 = cv2.cvtColor(pre_edge2, cv2.COLOR_BGR2RGB)

    mask_inv = mask - 1
    mask_inv[mask_inv < 0] = 1

    back_mask = cv2.bitwise_and(back, back, mask=mask_inv)  # 背景抠人

    all_img = cv2.add(back_mask, pre_noedge2) # 背景 + 没边界人
    back_edge2 = cv2.bitwise_and(back, back, mask=edge2.astype(np.uint8)) # 背景 边界
    pre_edge2 = cv2.bitwise_and(pre, pre, mask=edge2.astype(np.uint8)) # 前景 边界
    all_edge2 = cv2.addWeighted(back_edge2, 0.1, pre_edge2, 0.5, 0) # 全图 边界
    all_img = cv2.add(all_edge2, all_img) # 全图

    back_fuck = cv2.bitwise_and(back, back, mask=mask_inv.astype(np.uint8)) # 背景 抠人 不考虑边界
    bad_all_img = cv2.add(back_fuck, cv2.cvtColor(person, cv2.COLOR_BGR2RGB)) # 直接

    return all_img, bad_all_img

def VisFeather(mask, person, raw_img, edge_width=5):
    # mask = mask.astype(np.byte)
    if len(mask) == 0:
        return None, None
    mask[mask >0] = 1
    edge = np.zeros(person.shape, np.uint8)

    blurred_img = cv2.GaussianBlur(person, (21, 21), 0)
    gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    try:
        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        _, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(edge, contours, -1, (255, 255, 255), edge_width) # color, width 5

    # output = np.where(mask == np.array([255, 255, 255]), blurred_img, person)

    edge2 = (edge / edge.max())[..., 0]

    mask = mask.astype(np.byte)
    mask_all =( mask + edge2).astype(np.uint8)
    mask_all[mask_all> 0] = 1


    mask_noedge2 =  edge2.astype(np.byte) - mask.astype(np.byte)
    mask_noedge2 = mask_noedge2 == -1 # edge 超出 mask， 有重叠部分， 有超出部分
    mask_noedge2 = 1 * mask_noedge2
    # back = cv2.imread("/home/xjx/Pictures/20171018112826932693.jpg")
    # if back.shape[0] != person.shape[0]:
    back = np.zeros(person.shape, np.uint8)


    pre_noedge2 = cv2.bitwise_and(person, person, mask=mask_noedge2.astype(np.uint8))  # 前景 没边
    pre = cv2.bitwise_and(blurred_img, blurred_img, mask=mask_all.astype(np.uint8))  # 前景 边界 使用模糊边界做

    # back = cv2.cvtColor(back, cv2.COLOR_BGR2RGB)
    # pre_noedge2 = cv2.cvtColor(pre_noedge2, cv2.COLOR_BGR2RGB)
    # pre_edge2 = cv2.cvtColor(pre_edge2, cv2.COLOR_BGR2RGB)

    mask_inv = mask_all.astype(np.byte) - 1
    mask_inv[mask_inv < 0] = 1

    back_mask = cv2.bitwise_and(back, back, mask=mask_inv.astype(np.uint8))  # 背景抠人

    all_img = cv2.add(back_mask, pre_noedge2)  # 背景 + 没边界人
    back_edge2 = cv2.bitwise_and(back, back, mask=edge2.astype(np.uint8))  # 背景 边界
    pre_edge2 = cv2.bitwise_and(pre, pre, mask=edge2.astype(np.uint8))  # 前景 边界
    all_edge2 = cv2.addWeighted(back_edge2, 0.99, pre_edge2, 0.1, 0)  # 全图 边界
    all_img = cv2.add(all_edge2, all_img)  # 全图

    # back_fuck = cv2.bitwise_and(back, back, mask=mask_inv.astype(np.uint8))  # 背景 抠人 不考虑边界
    person_bad = cv2.bitwise_and(raw_img, raw_img, mask=mask_all.astype(np.uint8))
    bad_all_img = cv2.add(back_mask, person_bad)  # 直接


    return all_img, bad_all_img


def VisFeather_2(mask, person, raw_img, edge_width=5):
    '''
    mask 有2个， mask_all 代表扩充边界的mask，  mask代表分割结果
    处理思路是 图片是由3部分组成， 前景，背景，边界过度，
    边界过度区域是来源于： mask 产生的前景和背景融合后， 加入高斯模糊， 在使用扩充的边界提取出一条边界作为边界过度区域。


    edge2 = Func  mask
    mask_all = edge2 + mask
    back_mask = mask_inv + back

    blurred_img = 高斯模糊  (back_mask + person)
    2. blurred_img_edge2 = edge2 + blurred_img

    mask_noedge2 = mask_all - edge2
    3. per_noedge2 = raw_img +   mask_noedge2

    4.back_mask_all =  mask_all_inv + back

    5. all_img = 透明度(  back_mask +  blurred_img_edge2 ) + per_noedge2

    '''
    mask[mask >0] = 1
    # 1.1 edge2
    edge = np.zeros(person.shape, np.uint8)
    gray = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edge, contours, -1, (255, 255, 255), edge_width)  # color, width 5
    edge2 = (edge / edge.max())[..., 0]

    # 1.2 mask_all
    mask = mask.astype(np.byte)
    mask_all =( mask + edge2).astype(np.uint8)
    mask_all[mask_all> 0] = 1

    # 1.3 back_mask
    back = cv2.imread("/home/xjx/Pictures/20171018112826932693.jpg")
    if back.shape[0] != person.shape[0]:
        back = np.zeros(person.shape, np.uint8)
    mask_inv = mask_all.astype(np.byte) - 1
    mask_inv[mask_inv < 0] = 1
    back_mask = cv2.bitwise_and(back, back, mask=mask_inv.astype(np.uint8))  # 背景抠人

    # 2.blurred_img_edge2
    blurred_img = cv2.add(back_mask, person)
    blurred_img = cv2.GaussianBlur(blurred_img, (21, 21), 0)
    blurred_img_edge2 = cv2.bitwise_and(blurred_img, blurred_img, mask=edge2.astype(np.uint8))

    # 3.per_noedge2
    mask_noedge2 = mask_all.astype(np.byte) - edge2.astype(np.byte)
    mask_noedge2 = mask_noedge2.astype(np.uint8)
    per_noedge2 = cv2.bitwise_and(raw_img, raw_img, mask=mask_noedge2.astype(np.uint8))

    # 4. back_mask_all
    mask_all_inv = mask_all.astype(np.byte) - 1
    mask_all_inv[mask_all_inv < 0] = 1
    back_mask_all = cv2.bitwise_and(back, back, mask=mask_all_inv.astype(np.uint8))

    # 5 all_img
    # all_img = cv2.add(blurred_img_edge2, per_noedge2)
    # all_img = cv2.add(raw_img, back_mask_all)

    # 5 all_img # 透明度版本
    all_img = cv2.addWeighted(back_mask, 1, blurred_img_edge2, 1.2, 0)
    all_img = cv2.add(all_img, per_noedge2)



    return all_img


class Transform:

    def __init__(self, cfg):
        self.cfg = cfg
        if hasattr(cfg, 'edge_width'):
            self.edge_width = cfg.edge_width

    def __call__(self, raw_img, mask_img):

        mask_img_n = np.stack((mask_img, mask_img, mask_img), axis=2)
        mix_img = cv2.addWeighted(raw_img, 0.5, mask_img_n * 255, 0.5, 1)

        ret1, mask_person = cv2.threshold(mask_img, 0, 1, cv2.THRESH_BINARY)  # 只有人
        ret2, mask_back = cv2.threshold(mask_img, 0, 1, cv2.THRESH_BINARY_INV)  # 只有背景

        img_person = cv2.bitwise_and(raw_img, raw_img, mask=mask_person)
        img_back = cv2.bitwise_and(raw_img, raw_img, mask=mask_back)
        if hasattr(self.cfg, 'edge_width'):
            all_img, bad_all_img = VisFeather(mask_img, img_person, raw_img, self.edge_width)
        else:
            all_img, bad_all_img = None, None
        # all_img, bad_all_img = func.VisChangeBack(mask_img, img_person, flag_canny=True)


        return dict(mask_img=mask_img_n, mix_img=mix_img, raw_img=raw_img, person=img_person, back=img_back, change_back_addweight=all_img, change_back=bad_all_img )


class CropRoi:

    def __init__(self, cfg):
        self.roi_thre = 0
        self.edge_width = 1
        self.ratio = 1.2
        self.transform = Transform(cfg)

    def __call__(self, pred, net, generate_img):
        '''
        找到mask的外接矩形， 判断矩形大小，较小就扩大一点resize成512*512重新送入网路
        '''
        img = pred['raw_img'].copy()
        mask_img = pred['mask_img'].copy()
        mask = mask_img[..., 0].astype(np.uint8)


        # thresh = (mask * 255).astype(np.uint8)
        # contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_h, img_w, c = img.shape

        for cnt in contours:
            # cnt = contours[0]
            x1, y1, w, h = cv2.boundingRect(cnt)
            x2, y2 = x1 + w, y1 + h

            x_c, y_c = x1 + w // 2, y1 + h // 2
            max_side = int(max(w, h) * self.ratio)
            x1_n, y1_n, x2_n, y2_n = x_c - max_side//2 , y_c - max_side//2 , x_c + max_side//2, y_c + max_side//2
            x1_n, y1_n = max(0, x1_n), max(0, y1_n)
            x2_n, y2_n = min(x2_n, img_w), min(y2_n, img_h)
            w_n, h_n = x2_n - x1_n, y2_n - y1_n

            imgRoi = img[y1_n:y2_n, x1_n:x2_n, :].copy()
            imgRoi_tensor = generate_img(imgRoi)
            predRoi = net(imgRoi_tensor)
            maskRoi = predRoi['mask_img'][..., 0]
            # maskRoi_reshape = cv2.resize(maskRoi, (w_n, h_n), cv2.INTER_NEAREST)
            mask[y1_n:y2_n, x1_n:x2_n] = maskRoi
            cv2.rectangle(img, (x1_n, y1_n), (x2_n, y2_n), (0,255,0), 2)
        img_dict = self.transform(img, mask)

        z = 1

        return img_dict



            # cv2.rectangle(mask, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        # cv2.drawContours(edge, contours, -1, (255, 255, 255), self.edge_width)  # color, width 5
        # edge2 = (edge / edge.max())[..., 0]


