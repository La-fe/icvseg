# coding=utf-8
import numpy as np


# 设标签宽W，长H
def fast_hist(a, b, n):  # a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的标签，形状(H×W,)；n是类别数目，实数（在这里为19）
    '''
    核心代码
    '''
    k = (a >= 0) & (a < n)  # k是一个一维bool数组，形状(H×W,)；目的是找出标签中需要计算的类别（去掉了背景） k=0或1
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n,
                                                                              n)  # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)


def per_class_iu(hist):  # 分别为每个类别（在这里是19类）计算mIoU，hist的形状(n, n)
    '''
    核心代码
    '''
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    miou[miou != miou] = 1
    return miou

# a=np.random.randint(0,3,size=[2,2])
# b=np.random.randint(0,3,size=[2,2])
# print(a)
# print(b)
# hist=fast_hist(a,b,3)
# print(hist)
# print(per_class_iu(hist))
z = 1