from enum import unique
import numpy as np
from sklearn.metrics import silhouette_score
from snc.clustering import SNC
import matplotlib.pyplot as plt
from utils.merge121 import req_numclust
from utils.utils import cluster_acc
from utils.logger import get_logger
import random
import itertools

random.seed(0)
np.random.seed(0)

def sample_each_class(trg, msk, ratio):
    trg_lb = trg[msk == 1]
    cls_id = np.unique(trg_lb)
    sample_clsnum = int(ratio*len(cls_id))
    lb_sample = []
    for i in cls_id:
        if i < sample_clsnum:
            sample = np.where(trg_lb == i)[0]
            lb_sample = lb_sample + sample.tolist()
    msk_sub = np.zeros_like(trg_lb, dtype=np.int64)
    msk_sub[lb_sample] = 1
    return msk_sub

def check_num(trg, msk, msk_new):
    trg_lb = trg[msk == 1]
    trg_lb_sub = trg[msk_new == 1]
    # print(len(np.unique(trg_lb)))
    # print(len(np.unique(trg_lb_sub)))
    for u in np.unique(trg_lb):
        print("{}:{}".format(u, sum(trg_lb == u)))
    for u in np.unique(trg_lb_sub):
        print("{}:{}".format(u, sum(trg_lb_sub == u)))

def norm(s):
    if s.max() != s.min():
        return (s - s.min()) / (s.max() - s.min())
    elif s.max() == s.min():
        return s

font_size = 16
p = ['cifar10', 'cifar100', 'imgnet100', 'cub', 'car', 'herb']
title = ['CIFAR-10', 'CIFAR-100', 'ImageNet-100', 'CUB-200', 'SCars', 'Herbarium19']
class_num = [10, 100, 100, 200, 196, 683]

plt.figure(figsize=(24, 12.8))
for di, pi in enumerate(p):
    dataset = pi

    out = np.load('./features/'+ dataset +'/outputs.npy')
    trg = np.load('./features/'+ dataset +'/targets.npy')
    msk = np.load('./features/'+ dataset +'/masks.npy')

    if pi == 'cifar10':
        ratio = 0.6
    else:
        ratio = 0.8
    msk_sub = sample_each_class(trg, msk, ratio)

    msk_new = np.zeros_like(msk, dtype=np.int64)
    msk_new[msk == 1] = msk_sub

    # check_num(trg, msk, msk_new)
    prd, num_clust, req, d_all = SNC(out, labeled=trg, mask=msk_new)

    # # mask = mask_lb

    CN = []
    S = []
    ACCLB = []

    for i in range(prd.shape[1]):
        c = prd[:,i]
        s = silhouette_score(out[msk_new==0], c[msk_new==0], metric='cosine', sample_size=2048, random_state=0) 
        acc = cluster_acc(trg[msk==1][msk_sub==0], c[msk==1][msk_sub==0])
        S.append(s)
        ACCLB.append(acc)
        CN.append(np.unique(c).shape[0])

    y = norm(np.array(ACCLB))
    s = norm(np.array(S))
    x = np.array(CN)
    metric = s*y

    last = np.argmax(metric)-1
    next = np.argmax(metric)+1
    if pi == 'herb':
        last = 1
        next = 3
    _, c_all, d_all = req_numclust(prd[:,last], out, req_clust=x[next], distance='cosine', labeled=trg, mask=msk_new)


    CN = []
    S = []
    ACCLB = []

    for c in c_all:
        s = silhouette_score(out[msk_new==0], c[msk_new==0], metric='cosine', sample_size=2048, random_state=0) 
        acc = cluster_acc(trg[msk==1][msk_sub==0], c[msk==1][msk_sub==0])
        S.append(s)
        ACCLB.append(acc)
        CN.append(np.unique(c).shape[0])

    y = norm(np.array(ACCLB))
    s = norm(np.array(S))
    x = np.array(CN)
    metric = s*y
    plt.subplot(2,3,di+1)
    plt.axvline(class_num[di], 0, 1, linestyle='dotted', color='red', label='GT={}'.format(class_num[di]), linewidth=2)
    plt.axvline(x[np.argmax(metric)], 0, 1, linestyle='dotted', color='c', label='Est.={}'.format(x[np.argmax(metric)]), linewidth=2)
    plt.plot(x, y, c='g', linestyle='-.', label='Labelled accuracy', linewidth=2)
    plt.plot(x, s, c='b', linestyle='--', label='Silhouette score', linewidth=2)
    plt.plot(x, metric, c='c', linestyle='-', label='Reference score', linewidth=2)
    plt.title(title[di], fontsize=font_size)
    plt.xlabel('Class number', fontsize=font_size)  # Add an x-label to the axes.
    plt.ylabel('Score', fontsize=font_size)  # Add a y-label to the axes.
    plt.legend(fontsize=12)
    
    print(dataset + " Estimated class number is {}".format(x[np.argmax(metric)]))
    print(dataset + " metric score is {}".format(metric[np.argmax(metric)]))
    
plt.savefig('class_number_curve.pdf')