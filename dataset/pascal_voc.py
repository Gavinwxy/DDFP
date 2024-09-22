import copy
import math
import os
import os.path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from dataset.transform import *
from copy import deepcopy

from .base import BaseDataset


class voc_dset(BaseDataset):
    def __init__(
        self, data_root, data_list, seed=0, n_sup=10582, split="val", img_size=513
    ):
        super(voc_dset, self).__init__(data_list)
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        
        random.seed(seed)
        if len(self.list_sample) >= n_sup and split != "val":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split != "val":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample
        

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
        img = self.img_loader(image_path, "RGB")
        mask = self.img_loader(label_path, "L")

        if self.split == 'val':
            ignore_value = 255
            img, mask = crop_center(img, mask, 513, ignore_value)
            img, mask = normalize(img, mask)
            return img, mask

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.split == 'unsup' else 255
        img, mask = crop(img, mask, self.img_size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.split == 'sup':
            return normalize(img, mask)
        
        img_w, img_s = deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s)
        img_s = transforms.RandomGrayscale(p=0.2)(img_s)
        img_s = blur(img_s, p=0.5)


        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s, ignore_mask = normalize(img_s, ignore_mask)
        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s, ignore_mask


    def __len__(self):
        return len(self.list_sample_new)


def build_vocloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 10582)
    # build transform
    dset = voc_dset(cfg["data_root"], cfg["data_list"], seed, n_sup)

    # build sampler
    sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_voc_semi_loader(all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get('train', {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = 10582 - cfg.get("n_sup", 10582)

    # build transform
    dset = voc_dset(cfg["data_root"], cfg["data_list"], seed, n_sup, 'sup', cfg['train_crop_size'])

    # build sampler for unlabeled set
    data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
    dset_unsup = voc_dset(cfg["data_root"], data_list_unsup, seed, n_sup, 'unsup', cfg['train_crop_size'])

    sample_sup = DistributedSampler(dset)
    loader_sup = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample_sup,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    sample_unsup = DistributedSampler(dset_unsup)
    loader_unsup = DataLoader(
        dset_unsup,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample_unsup,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    return loader_sup, loader_unsup
