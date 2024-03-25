import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .utils import TransformTwice
import torch.utils.data as data
import numpy as np
import torch
from data.randomaugment import RandAugment

def ssb_target(targets, split_path='./data/splits/scars_osr_splits.pkl'):
    splits = np.load(split_path, encoding='bytes', allow_pickle=True)
    new_class = splits['known_classes'] + splits['unknown_classes']['Easy'] + splits['unknown_classes']['Medium'] + splits['unknown_classes']['Hard']
    targets_new = []
    for t in targets:
        targets_new.append(new_class.index(t-1)) # convert scars label from 1-196 to 0-195 
    return targets_new

class CarDataset(Dataset):
    def __init__(self, data_dir, trg_dir, transforms=None, num_lab=98):
        self.img_list, self.trg_list = self.get_img_info(data_dir, trg_dir)
        self.transforms = transforms
        self.num_lab = num_lab

        self.targets = np.array(self.trg_list)
        sort_id = np.argsort(self.targets)
        sort_tgt = np.sort(self.targets)

        sort_id = sort_id[sort_tgt<num_lab]
        if len(sort_id) % 2 == 0:
            sort_id = np.reshape(sort_id, (-1, 2))
        else:
            sort_id = np.reshape(sort_id[:-1], (-1, 2))
        idx = sort_id[:,0]
        self.mask = np.zeros_like(self.targets)
        self.mask[idx] = 1

    def __getitem__(self, index):
        img_path, label, mask = self.img_list[index], self.targets[index], self.mask[index]
        image = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label, mask, index

    def __len__(self):
        return len(self.trg_list)
    
    @staticmethod
    def get_img_info(data_dir, trg_dir):
        img_list = []
        trg_list = []
        with open(trg_dir, 'r') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            img_path = os.path.join(data_dir, '{:0>5d}.jpg'.format(i+1))
            img_list.append(img_path)
            trg_list.append(int(lines[i].strip()))
        trg_list = ssb_target(trg_list)
        return img_list, trg_list

# rand-aug
def CarData_GCD(data_dir, trg_dir, aug=None, num_lab_classes=98):
    normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    if aug==None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            normalize,
        ])
    elif aug=='once':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
        transform.transforms.insert(0, RandAugment(2, 30, args=None))

    elif aug=='twice':
        transform1 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
        transform1.transforms.insert(0, RandAugment(2, 30, args=None))

        transform2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            normalize,
        ])
        transform2.transforms.insert(0, RandAugment(2, 30, args=None))

        transform = TransformTwice(transform1, transform2)
    dataset = CarDataset(data_dir=data_dir, trg_dir=trg_dir, transforms=transform, num_lab=num_lab_classes)
    return dataset

def CarLoaderMixGCD(data_dir, trg_dir, batch_size, num_workers=2, aug=None, shuffle=True, sampler=None, num_lab_classes=98):
    dataset = CarData_GCD(data_dir, trg_dir, aug, num_lab_classes)

    if sampler: 
        label_len = sum(dataset.mask==1)
        unlabelled_len = sum(dataset.mask==0)

        print(label_len)
        print(unlabelled_len)
        sample_weights = np.zeros_like(dataset.mask, dtype=np.float64)
        sample_weights[dataset.mask==1] = 1
        sample_weights[dataset.mask==0] = label_len / unlabelled_len
        # sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in dataset.mask]
        sample_weights = torch.DoubleTensor(sample_weights)
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(dataset))

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler, pin_memory=True)
    return loader
