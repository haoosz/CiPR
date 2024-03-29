from email.policy import strict
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, AdamW, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans
from utils.utils import BCE, PairSame, minmaxscaler, norm_feat, PairEnum, cluster_acc, Identity, AverageMeter, seed_torch
from utils import ramps 
from utils.heads import MLP, DINOHead
from models.vision_transformer import vit_base
from data.cifarloader import CIFAR100LoaderMixGCD, CIFAR10LoaderMixGCD
from data.imgnetloader import IMGNet100LoaderMixGCD
from data.cubloader import CUBLoaderMixGCD
from data.carloader import CarLoaderMixGCD
from data.airloader import AirLoaderMixGCD
from data.herbloader import HerbLoaderMixGCD
from tqdm import tqdm
import numpy as np
import os
import wandb
import math
from torchvision.models import resnet50

def get_files(model, root, test_loader, proj):
    model.eval()
    outputs=np.array([[]])
    targets=np.array([])
    masks=np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, mask, index) in enumerate(tqdm(test_loader)):
            x, label = x.to(device), label.to(device)
            if proj:
                output, _ = model(x)
            else:
                _, output = model(x)

            outputs=np.append(outputs, output.cpu().numpy())
            targets=np.append(targets, label.cpu().numpy())
            masks = np.append(masks, mask.numpy())
    outputs = outputs.reshape(targets.shape[0], -1)
    print(outputs.shape)
    if not os.path.exists(root):
        os.makedirs(root)
    np.save(os.path.join(root,"outputs.npy"),outputs)
    np.save(os.path.join(root,"targets.npy"),targets)
    np.save(os.path.join(root,"masks.npy"), masks)

class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        norm = x.norm(dim = 1, keepdim = True).clamp(min = eps)
        return x / norm

class MLP(nn.Module):
    def __init__(self, dim, dim_out, num_layers, hidden_size = 256):
        super().__init__()

        layers = []
        dims = (dim, *((hidden_size,) * (num_layers - 1)))

        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = ind == (len(dims) - 1)

            layers.extend([
                nn.Linear(layer_dim_in, layer_dim_out),
                nn.GELU() if not is_last else nn.Identity()
            ])

        self.net = nn.Sequential(
            *layers,
            L2Norm(),
            nn.Linear(hidden_size, dim_out)
        )

    def forward(self, x):
        return self.net(x)

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

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--exp_root', type=str, default='./checkpoints/')
    parser.add_argument('--warmup_model_dir', type=str, default='./pretrain/dino_vitbase16_pretrain.pth')
    parser.add_argument('--model_name', type=str, default='cipr')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--proj_dim', default=65536, type=float)

    parser.add_argument('--vit_small', action='store_true')
    parser.add_argument('--dinov2', action='store_true')
    parser.add_argument('--dataset_name', type=str, default='herb')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(args.seed)

    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir= os.path.join(args.exp_root, 'run')
    args.model_dir = model_dir+'/{}_{}/final.pth'.format(args.model_name, args.dataset_name)
    print("Load model from {}".format(args.model_dir))

    dataset_txt = args.dataset_name.split('_')[0]

    wandb.init(project="CiPR_" + dataset_txt, name=args.model_name, config=args)

    device_ids = range(args.gpus)
    model = ViT_Linear(args.proj_dim, vit_samll=args.vit_small, dinov2=args.dinov2).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    

    if args.vit_small:
        args.warmup_model_dir = "./pretrain/dino_deitsmall16_pretrain.pth"
    if not args.dinov2:
        state_dict = torch.load(args.warmup_model_dir)
        model.module.backbone.load_state_dict(state_dict)
                
    if args.dataset_name == 'cifar100':
        args.num_unlabeled_classes = 20
        args.num_labeled_classes = 80
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data'
        train_loader = CIFAR100LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, target_list=range(args.num_classes))
        train_loader_test = CIFAR100LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, target_list=range(args.num_classes))

    elif args.dataset_name == 'cifar10':
        args.num_unlabeled_classes = 5
        args.num_labeled_classes = 5
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data'
        train_loader = CIFAR10LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True,target_list=range(args.num_classes))
        train_loader_test = CIFAR10LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, target_list=range(args.num_classes))

    elif args.dataset_name == 'cub':
        args.num_unlabeled_classes = 100
        args.num_labeled_classes = 100
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data/cub_200_2011'
        train_loader = CUBLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, num_lab_classes=args.num_labeled_classes)
        train_loader_test = CUBLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, num_lab_classes=args.num_labeled_classes)

    elif args.dataset_name == 'imgnet100':
        args.num_unlabeled_classes = 50
        args.num_labeled_classes = 50
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data/imagenet100/train'
        train_loader = IMGNet100LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, num_lab_classes=args.num_labeled_classes)
        train_loader_test = IMGNet100LoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, num_lab_classes=args.num_labeled_classes)

    elif args.dataset_name == 'car':
        args.num_unlabeled_classes = 98
        args.num_labeled_classes = 98
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data/cars_train'
        trg_dir = '../data/cars_devkit/train_perfect_preds.txt'
        train_loader = CarLoaderMixGCD(data_dir=args.dataset_root, trg_dir=trg_dir, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, num_lab_classes=args.num_labeled_classes)
        train_loader_test = CarLoaderMixGCD(data_dir=args.dataset_root, trg_dir=trg_dir, batch_size=args.batch_size, aug=None, shuffle=False, num_lab_classes=args.num_labeled_classes)
    
    elif args.dataset_name == 'herb':
            args.num_unlabeled_classes = 342
            args.num_labeled_classes = 341
            args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
            args.dataset_root= '../data/herbarium19/small-train'
            train_loader = HerbLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, num_lab_classes=args.num_labeled_classes)
            train_loader_test = HerbLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, num_lab_classes=args.num_labeled_classes)

    elif args.dataset_name == 'air':
        args.num_unlabeled_classes = 50
        args.num_labeled_classes = 50
        args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
        args.dataset_root= '../data/aircrafts_trainval'
        train_loader = AirLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug='twice', shuffle=False, sampler=True, num_lab_classes=args.num_labeled_classes)
        train_loader_test = AirLoaderMixGCD(root=args.dataset_root, batch_size=args.batch_size, aug=None, shuffle=False, num_lab_classes=args.num_labeled_classes)

    root = "./features/{}".format(args.dataset_name)
    model.load_state_dict(torch.load(args.model_dir), strict=True)
    get_files(model, root, train_loader_test, False)