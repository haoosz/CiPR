import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.utils import BCE, PairSame, minmaxscaler, norm_feat, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch, split_cluster_acc_v2
from utils.heads import MLP, DINOHead
from utils import ramps 
from models.vision_transformer import vit_base, vit_small
from data.cifarloader import CIFAR100LoaderMixGCD, CIFAR10LoaderMixGCD
from data.imgnetloader import IMGNet100LoaderMixGCD
from data.cubloader import CUBLoaderMixGCD
from data.carloader import CarLoaderMixGCD
from data.herbloader import HerbLoaderMixGCD
from data.airloader import AirLoaderMixGCD
from tqdm import tqdm
import numpy as np
import os
import wandb
import math
from torchvision.models import resnet50
from snc.clustering import SNC
import torchvision.transforms as transforms


# ## standard finch
# def generate_pl(model, test_loader):
#     model.eval()
#     outputs=np.array([[]])
#     targets=np.array([])
#     indexes=np.array([])
#     with torch.no_grad():
#         for batch_idx, (x, label, mask, index) in enumerate(tqdm(test_loader)):
#             x = x.to(device)
#             _, output = model(x)
#             mask_lb = mask == 1

#             outputs = np.append(outputs, output[~mask_lb].cpu().numpy())
#             targets = np.append(targets, label[~mask_lb].numpy())
#             indexes = np.append(indexes, index[~mask_lb].numpy())
#     outputs = outputs.reshape(targets.shape[0], -1)
#     print(outputs.shape[0])
#     c, _, _ = FINCH(outputs)
#     return c, targets, indexes

## snc
def generate_pl(model, test_loader):
    model.eval()
    outputs=np.array([[]])
    targets=np.array([])
    masks=np.array([])
    indexes=np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, mask, index) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            _, output = model(x)

            outputs = np.append(outputs, output.cpu().numpy())
            targets = np.append(targets, label.numpy())
            masks = np.append(masks, mask.numpy())
            indexes = np.append(indexes, index.numpy())
    outputs = outputs.reshape(targets.shape[0], -1)
    print(outputs.shape[0])
    c, _, _, _ = SNC(outputs, labeled=targets, mask=masks)
    return c, targets, indexes

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

    return purity

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SelfConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SelfConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode

    def forward(self, features, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def train(model, train_loader, train_loader_test, args):
    param_backbone = [param for name, param in model.named_parameters() if param.requires_grad and 'head' not in name]
    param_linear = [param for name, param in model.named_parameters() if param.requires_grad and 'head' in name]

    optimizer = SGD([
        {'params': param_backbone, "lr": args.lr},
        {'params': param_linear, 'lr': args.lr_linear}
    ], momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr*1e-2)

    criterion_sup = SupConLoss(temperature=0.07)
    criterion_pseudo = SupConLoss(temperature=0.12)
    criterion_self = SelfConLoss(temperature=1.0)
    
    for epoch in range(args.epochs):

        if epoch % 1 == 0:
            pl_all, target_all, index_all = generate_pl(model, train_loader_test)
            purity = cluster_purity(target_all, pl_all[:,args.partition])
            pl_all = torch.from_numpy(pl_all).to(device) # choose the partition

            print('Purity: {}'.format(purity))
            wandb.log({"purity": purity}, step=epoch)   

        loss_record = AverageMeter()
        loss1_record = AverageMeter()
        loss3_record = AverageMeter()
        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader),desc = "Epoch:{}".format(epoch))
        for batch_idx, ((x, x_bar), label, mask, index) in pbar:

            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            embed, _ = model(x) 
            embed_bar, _ = model(x_bar)

            features = torch.stack([embed, embed_bar], dim = 1)
            mask_lb = mask == 1
            if features[mask_lb].shape[0] == 0 or features[~mask_lb].shape[0] == 0:
                continue

            # xsorted = np.argsort(index_all)
            # ypos = np.searchsorted(index_all[xsorted], index[~mask_lb])
            # indices = xsorted[ypos]

            # pl = pl_all[indices, args.partition]

            pl = pl_all[index, args.partition]

            loss1 = criterion_sup(features[mask_lb], labels=label[mask_lb]) 
            loss2 = criterion_self(features)
            loss3 = criterion_pseudo(features, labels=pl)
            
            if args.warmup and epoch < 10:
                print("warmuping")
                loss = loss1 + loss2
            else:
                loss = loss1 + loss3
            loss_record.update(loss.item(), x.size(0))
            loss1_record.update(loss1.item(), x.size(0))
            loss3_record.update(loss3.item(), x.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss_sup, loss_unsup":"{:.3f} {:.3f}".format(loss1, loss3)})
            # outputs=np.append(outputs, embed.detach().cpu().numpy())

        scheduler.step()

        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        wandb.log({"train_loss": loss_record.avg}, step=epoch)
        wandb.log({"sup_loss": loss1_record.avg}, step=epoch)
        wandb.log({"unsup_loss": loss3_record.avg}, step=epoch)

        if epoch % 10 == 0:
            print('test on unlabeled classes')
            acc_unlab = test_unlab_new(model, train_loader_test, args.num_labeled_classes, args.num_unlabeled_classes, proj=False)
            wandb.log({"eval_acc_unlab": acc_unlab}, step=epoch)
        
        if epoch % 10 == 0:
            model_dir = args.model_dir + '/{:0>5d}.pth'.format(epoch) 
            torch.save(model.state_dict(), model_dir)
            print("model saved to {}.".format(model_dir))

def test_unlab_new(model, test_loader, num_old, num_new, proj):
    num_class = num_old + num_new
    model.eval()
    outputs=np.array([[]])
    targets=np.array([])
    masks = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, mask, index) in enumerate(tqdm(test_loader)):
            x = x.to(device)
            if proj:
                output, _ = model(x)
            else:
                _, output = model(x)

            outputs=np.append(outputs, output.cpu().numpy())
            targets=np.append(targets, label.numpy())
            masks=np.append(masks, mask.numpy())

    outputs = outputs.reshape(targets.shape[0], -1)

    prd, num_clust, req, d_all = SNC(outputs, req_clust=num_class, labeled=targets, mask=masks)
    
    mask_unlb = masks == 0
    trg_unlb = targets[mask_unlb]
    req_unlb = req[mask_unlb]

    mask_old = trg_unlb < num_old
    acc_all, acc_old, acc_new = split_cluster_acc_v2(trg_unlb, req_unlb, mask_old)

    print('Test acc_old {:.4f}, acc_new {:.4f}, acc {:.4f}'.format(acc_old, acc_new, acc_all))
    return acc_all

class ViT_Linear(nn.Module):
    def __init__(self, dim_out, vit_samll=False, dinov2=False):
        super(ViT_Linear, self).__init__()

        if vit_samll:
            self.backbone = vit_small()
            print("Using ViT-small!")
        elif dinov2:
            self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            print("Using Dinov2!")
        else:
            self.backbone = vit_base()
        
        # self.head = MLP(self.backbone.embed_dim, dim_out, num_layers=4)
        self.head = DINOHead(self.backbone.embed_dim, dim_out)

    def forward(self, x):
        y = self.backbone(x) # batch, dim
        embed = F.normalize(self.head(y), dim=1)
        # embed = self.head(y)
        return embed, y

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=1e-3) # 1e-1
    parser.add_argument('--lr_linear', type=float, default=1e-1)
    # parser.add_argument('--min_lr_rate', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
 
    parser.add_argument('--exp_root', type=str, default='./checkpoints/')
    parser.add_argument('--warmup_model_dir', type=str, default='./pretrain/dino_vitbase16_pretrain.pth')
    parser.add_argument('--model_name', type=str, default='cipr')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--lambd', default=0.35, type=float)
    parser.add_argument('--proj_dim', default=65536, type=float)
    parser.add_argument('--partition', default=1, type=int)
    parser.add_argument('--warmup', action='store_true')

    parser.add_argument('--vit_small', action='store_true')
    parser.add_argument('--dinov2', action='store_true')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, imagenet100, cub, car, herb')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print(args.cuda)
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}_{}'.format(args.model_name, args.dataset_name) 
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    dataset_txt = args.dataset_name.split('_')[0]

    wandb.init(project="CiPR_" + dataset_txt, name=args.model_name, config=args)

    device_ids = range(args.gpus)
    model = ViT_Linear(args.proj_dim, vit_samll=args.vit_small, dinov2=args.dinov2).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)


    if args.mode=='train':
        if args.vit_small:
            args.warmup_model_dir = "./pretrain/dino_deitsmall16_pretrain.pth"
        if not args.dinov2:
            state_dict = torch.load(args.warmup_model_dir)
            model.module.backbone.load_state_dict(state_dict)
        for name, param in model.module.backbone.named_parameters(): 
            if 'blocks.11' not in name:
                param.requires_grad = False
    
    # specify a dataset
    if args.dataset_name == 'cifar100':
        args.num_unlabeled_classes = 20
        args.num_labeled_classes = 80
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data'
        args.long_tail = False
        train_loader = CIFAR100LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, target_list=range(args.num_classes))
        train_loader_test = CIFAR100LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, target_list=range(args.num_classes))

    elif args.dataset_name == 'cifar10':
        args.num_unlabeled_classes = 5
        args.num_labeled_classes = 5
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data'
        args.long_tail = False
        train_loader = CIFAR10LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, target_list=range(args.num_classes))
        train_loader_test = CIFAR10LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, target_list=range(args.num_classes))

    elif args.dataset_name == 'cub':
        args.num_unlabeled_classes = 100
        args.num_labeled_classes = 100
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data/cub_200_2011'
        args.long_tail = False
        train_loader = CUBLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, num_lab_classes=args.num_labeled_classes)
        train_loader_test = CUBLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, num_lab_classes=args.num_labeled_classes)

    elif args.dataset_name == 'imagenet100':
        args.num_unlabeled_classes = 50
        args.num_labeled_classes = 50
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data/imagenet100/train'
        args.long_tail = False
        train_loader = IMGNet100LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, num_lab_classes=args.num_labeled_classes)
        train_loader_test = IMGNet100LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, num_lab_classes=args.num_labeled_classes)

    elif args.dataset_name == 'car':
        args.num_unlabeled_classes = 98
        args.num_labeled_classes = 98
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data/cars_train'
        trg_dir = '../data/cars_devkit/train_perfect_preds.txt'
        args.long_tail = False
        train_loader = CarLoaderMixGCD(data_dir=args.dataset_root, trg_dir=trg_dir, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, num_lab_classes=args.num_labeled_classes)
        train_loader_test = CarLoaderMixGCD(data_dir=args.dataset_root, trg_dir=trg_dir, batch_size=args.batch_size, aug=None, shuffle=False, num_lab_classes=args.num_labeled_classes)

    elif args.dataset_name == 'herb':
        args.num_unlabeled_classes = 342
        args.num_labeled_classes = 341
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data/herbarium19/small-train'
        args.long_tail = True
        train_loader = HerbLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, num_lab_classes=args.num_labeled_classes)
        train_loader_test = HerbLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, num_lab_classes=args.num_labeled_classes)

    elif args.dataset_name == 'air':
        args.num_unlabeled_classes = 50
        args.num_labeled_classes = 50
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data/aircrafts_trainval'
        args.long_tail = False
        train_loader = AirLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, num_lab_classes=args.num_labeled_classes)
        train_loader_test = AirLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, num_lab_classes=args.num_labeled_classes)

    # train or test
    if args.mode == 'train':
        train(model, train_loader, train_loader_test, args)
        model_dir = args.model_dir + '/final.pth'
        torch.save(model.state_dict(), model_dir)
        print("model saved to {}.".format(model_dir))
    else:
        model_dir = args.model_dir + '/final.pth'
        print("model loaded from {}.".format(model_dir))
        model.load_state_dict(torch.load(args.model_dir))
        _ = test_unlab_new(model, train_loader_test, args.num_labeled_classes, args.num_unlabeled_classes, False)