# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#

'''
    Add timm prefetcher to evaluation dataloader.
'''


import os
import json
import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

# for prefetcher
from timm.data.transforms import ToNumpy
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.loader import fast_collate, PrefetchLoader, MultiEpochsDataLoader


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        split = ''
        if args.use_holdout:
            if is_train:
                split = 'sub-train' #'train-no-holdout'
            else:
                split = 'sub-val' #'holdout'
        else:
            if is_train:
                split = 'train'
            else:
                split = 'val'
        root = os.path.join(args.data_path, split)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    '''
        1. Add prefetcher transform for eval prefetch.
    '''
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    if args.use_prefetcher:
        t.append(ToNumpy())
    else:
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataloader(is_train, args, use_multi_epoch=False):
    '''
        Use prefetcher for evaluation dataset.
        
        Refernce: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/loader.py
    '''
    assert not is_train, "Currently support eval only"
    
    dataset, _ = build_dataset(is_train, args)
    sampler = None
    if args.distributed:
        if not is_train: 
            sampler = OrderedDistributedSampler(dataset)
            
    collate_fn = fast_collate if args.use_prefetcher else torch.utils.data.dataloader.default_collate
    
    dataloader_class = MultiEpochsDataLoader if use_multi_epoch else torch.utils.data.DataLoader
    loader = dataloader_class(dataset, 
        batch_size=args.val_bs, 
        shuffle=False, 
        num_workers=args.num_workers, 
        sampler=sampler, 
        collate_fn=collate_fn, 
        pin_memory=args.pin_mem, 
        drop_last=is_train)
    
    if args.use_prefetcher:
        #prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(loader)
        ''',
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )
        '''
    
    return loader