from ntpath import realpath
import time
import argparse
import numpy as np
from sklearn import metrics
import scipy.sparse as sp
import warnings
import math

try:
    from pynndescent import NNDescent

    pynndescent_available = True
except Exception as e:
    warnings.warn('pynndescent not installed: {}'.format(e))
    pynndescent_available = False
    pass

ANN_THRESHOLD = 100000

def compute_N(num):
    if num <= 20:
        N = 3
    elif num <= 50:
        N = 4
    elif num <= 130:
        N = 5
    else:
        N = 7
    return N

def clust_rank(mat, initial_rank=None, distance='cosine', labeled=None, mask=None, level=100, N=10):
    s = mat.shape[0]

    if initial_rank is not None:
        orig_dist = []
    elif s <= ANN_THRESHOLD:
        orig_dist = metrics.pairwise.pairwise_distances(mat, mat, metric=distance)
        np.fill_diagonal(orig_dist, 1e12)
        initial_rank = np.argmin(orig_dist, axis=1)
    else:
        if not pynndescent_available:
            raise MemoryError("You should use pynndescent for inputs larger than {} samples.".format(ANN_THRESHOLD))
        print('Using PyNNDescent to compute 1st-neighbours at this step ...')

        knn_index = NNDescent(
            mat,
            n_neighbors=2,
            metric=distance,
            random_state=0,
        )

        result, orig_dist = knn_index.neighbor_graph
        initial_rank = result[:, 1]
        orig_dist[:, 0] = 1e12
        print('Step PyNNDescent done ...')

    if labeled is not None:
        labeled = labeled.astype(int)
        labeled[~mask.astype(np.bool8)] = -101
        
        # print(len(np.unique(labeled[mask.astype(np.bool8)])))
        label_ref = labeled[mask.astype(np.bool8)]
        index = np.arange(len(labeled))[mask.astype(np.bool8)]

        orig_dist_lb = metrics.pairwise.pairwise_distances(mat[mask.astype(np.bool8)], mat[mask.astype(np.bool8)], metric=distance)
        np.fill_diagonal(orig_dist_lb, 1e12)

        unique, counts = np.unique(label_ref, return_counts=True)
        cls_num = dict(zip(unique, counts))

        for i in range(label_ref.shape[0]):
            if label_ref[i] >= 0:
                N = math.ceil(np.sqrt(cls_num[label_ref[i]]))
                d =[p[0] for p in np.argwhere(label_ref==label_ref[i])]
                if len(d)>N: # 3 for CUB
                    d = d[:N]
                else:
                    d = d
                pt = d.pop(0)
                while len(d) > 0:
                    dists = orig_dist_lb[pt]
                    sort = np.argsort(dists[d])
                    idx = sort[0]
                    next_pt = d.pop(idx)
                    initial_rank[index[pt]] = index[next_pt]
                    label_ref[pt] = -1
                    pt = next_pt
                if len(d) == 0:
                    label_ref[pt] = -1
                    initial_rank[index[pt]] = index[pt]             

    # The Clustering Equation
    A = sp.csr_matrix((np.ones_like(initial_rank, dtype=np.float32), (np.arange(0, s), initial_rank)), shape=(s, s)) # construct first neighbor

    A = A + sp.eye(s, dtype=np.float32, format='csr')
    A = A @ A.T # before or after
    A = A.tolil()
 
    if level != 0 and labeled is not None:
        for i, x1 in enumerate(labeled):
            if x1 >= 0:
                for j, x2 in enumerate(labeled):
                    if x2 >= 0 and x1 != x2:
                        A[i,j] = 0
    A.setdiag(0)
    return A, orig_dist


def get_clust(a, orig_dist, min_sim=None):
    if min_sim is not None:
        a[np.where((orig_dist * a.toarray()) > min_sim)] = 0

    num_clust, u = sp.csgraph.connected_components(csgraph=a, directed=True, connection='weak', return_labels=True)
    return u, num_clust


def cool_mean_old(M, u):
    _, nf = np.unique(u, return_counts=True)
    idx = np.argsort(u)
    M = M[idx, :]
    M = np.vstack((np.zeros((1, M.shape[1])), M))

    np.cumsum(M, axis=0, out=M)
    cnf = np.cumsum(nf)
    nf1 = np.insert(cnf, 0, 0)
    nf1 = nf1[:-1]

    M = M[cnf, :] - M[nf1, :]
    M = M / nf[:, None]
    return M


def cool_mean(M, u):
    s = M.shape[0]
    un, nf = np.unique(u, return_counts=True)
    umat = sp.csr_matrix((np.ones(s, dtype='float32'), (np.arange(0, s), u)), shape=(s, len(un)))
    return (umat.T @ M) / nf[..., np.newaxis]


def get_merge(c, u, data):
    if len(c) != 0:
        _, ig = np.unique(c, return_inverse=True)
        c = u[ig]
    else:
        c = u

    mat = cool_mean(data, c)
    return c, mat


def update_adj(adj, d):
    # Update adj, keep one merge at a time
    idx = adj.nonzero()
    v = np.argsort(d[idx])
    v = v[:2]
    x = [idx[0][v[0]], idx[0][v[1]]]
    y = [idx[1][v[0]], idx[1][v[1]]]
    a = sp.lil_matrix(adj.get_shape())
    a[x, y] = 1
    return a, d[x, y]


def req_numclust(c, data, req_clust, distance='cosine', labeled=None, mask=None, chain_length=10, long_tail=None):
    iter_ = len(np.unique(c)) - req_clust
    c_, mat = get_merge([], c, data)

    if labeled is not None:
        new_label, new_mask = get_new_label(labeled, c_, mask)
    else:
        new_label, new_mask = None, None

    dist_all = []
    c_all = []
    
    for i in range(iter_):
        adj, orig_dist = clust_rank(mat, initial_rank=None, distance=distance, labeled=new_label, mask=new_mask, N=chain_length)
        adj, min_dist = update_adj(adj, orig_dist)
        u, _ = get_clust(adj, [], min_sim=None)
        c_, mat = get_merge(c_, u, data)
        dist_all.append(min_dist)
        c_all.append(c_)
        if new_label is not None:
            new_label, new_mask = get_new_label(labeled, c_, mask)
    return c_, c_all, dist_all

def get_new_label(y_true, y_pred, mask):
    y_true_lb = y_true[mask.astype(np.bool8)].astype(int)
    y_pred_lb = y_pred[mask.astype(np.bool8)].astype(int)
    new_label = -1 * np.ones(y_pred.max()+1).astype(int)

    D_pred = y_pred_lb.max()+1
    D_true = y_true_lb.max()+1
    w = np.zeros((D_pred, D_true), dtype=np.int64)
    for i in range(y_true_lb.shape[0]):
        w[y_pred_lb[i], y_true_lb[i]] += 1
    
    # y_pred_unlb = y_pred[~mask.astype(np.bool8)].astype(int)
    # u = np.zeros(D_pred, dtype=np.int64)
    # for i in range(y_pred_unlb.shape[0]):
    #     if y_pred_unlb[i] < D_pred:
    #         u[y_pred_unlb[i]] += 1
    
    index = np.argmax(w, axis=1)
    index[np.max(w, axis=1) == 0] = -1
    # index[np.max(w, axis=1) < u] = -1
    new_label[:D_pred] = index
    new_mask = new_label >= 0
    return new_label, new_mask

