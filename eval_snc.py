import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment 
from sklearn import metrics
import scipy.sparse as sp
import warnings
from snc.clustering import SNC
from sklearn.cluster import KMeans
from utils.logger import get_logger
import time

def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets
    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape (n_samples,)
        y_pred: predicted labels, numpy.array with shape (n_samples,)
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = list(map(list, zip(*ind)))
    
    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc
    
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w) # assignment problem
    # print([w[i, j] for i, j in list(map(list, zip(*ind))) if w[i, j]!=0])
    # print(y_pred.size)
    return sum([w[i, j] for i, j in list(map(list, zip(*ind)))]) * 1.0 / y_pred.size

def cluster_purity(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    D1 = y_pred.max() + 1
    D2 = y_true.max() + 1
    w = np.zeros((D1, D2), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    max = np.max(w, axis=1)
    num = np.sum(max)
    purity = num / y_pred.size
    
    # print([w[i, j] for i, j in list(map(list, zip(*ind))) if w[i, j]!=0])
    # print(y_pred.size)
    return purity

if __name__ == '__main__':
    p = ["cifar10", "cifar100", "cub", "car", "imgnet100", "herb"]
    old = [5, 80, 100, 98, 50, 341]
    new = [5, 20, 100, 98, 50, 342]
    
    for i, pi in enumerate(p):
        print(pi)
        out = np.load("./features/" + pi + "/outputs.npy")
        trg = np.load("./features/" + pi + "/targets.npy")
        msk = np.load("./features/" + pi + "/masks.npy")

        # T1 = time.perf_counter()
        num_old = old[i]
        num_new = new[i]
        num = num_old + num_new

        # mask_unlb_new = (msk == 0) * (trg >= num_old) + (msk == 1)
        # mask_unlb_old = (msk == 0) * (trg < num_old) + (msk == 1)

        # out = out[mask_unlb_old]
        # trg = trg[mask_unlb_old]
        # msk = msk[mask_unlb_old]

        print(out.shape)
        prd, num_clust, req, d_all = SNC(out, req_clust=num, labeled=trg, mask=msk)

        # unlab data
        mask_unlb = msk == 0
        trg_unlb = trg[mask_unlb]
        req_unlb = req[mask_unlb]

        print(trg_unlb.shape[0])

        mask_old = trg_unlb < num_old
        acc_all, acc_old, acc_new = split_cluster_acc_v2(trg_unlb, req_unlb, mask_old)

        # acc = cluster_acc(trg_unlb, req_unlb)

        print('Test acc_old {:.4f}, acc_new {:.4f}, acc {:.4f}'.format(acc_old, acc_new, acc_all))
        # print('Test acc {:.4f}'.format(acc))

        # T2 = time.perf_counter()
        # logger.info(pi + " time: {}".format(T2-T1))