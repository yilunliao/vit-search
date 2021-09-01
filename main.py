# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import json
import os
import sys

from pathlib import Path

from timm.data import Mixup
from timm.data.loader import MultiEpochsDataLoader
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEmaV2

from datasets import build_dataset, build_dataloader
from engine import train_one_epoch, evaluate
from samplers import RASampler
import models
import utils

#from nets.create_model import my_create_model
import nets

# for super-network training
import supernet_config

# for finetuning from supernet's weights
from nets.net_utils import get_sub_state_dict

from logger import FileLogger
from utils import _load_teacher_model
from ast import literal_eval

from token_mixup import SwitchTokenMix

# for finetuning at higher resolution
from network_utils.finetune_state_dict import load_interpolated_state_dict


_MODELS_USE_NETWORK_DEF = ['flexible_vit_patch16_224', 'flexible_vit_patch16_224_supernet', 
                           'flexible_vit_patch16_192', 'flexible_vit_patch16_192_supernet', 
                           'flexible_vit_sr_patch14_224', 'flexible_vit_sr_patch14_224_supernet', 
                           'flexible_vit_sr_distill_patch14_224', 'flexible_vit_sr_distill_patch14_224_supernet', 
                           'flexible_vit_sr_patch14_224_patch_output', 
                           'flexible_vit_sr_patch14_224_patch_output_supernet', 
                           'flexible_vit_sr_patch14_280_patch_output', 'flexible_vit_sr_patch14_336_patch_output', 
                           'flexible_vit_sr_patch14_392_patch_output'
                           ]
_MODELS_FOR_SUPERNET = ['flexible_vit_patch16_224_supernet', 'flexible_vit_patch16_192_supernet', 
                        'flexible_vit_sr_patch14_224_supernet', 'flexible_vit_sr_distill_patch14_224_supernet', 
                        'flexible_vit_sr_patch14_224_patch_output_supernet']

_SEARCH_SPACE_CHOICES = ['tiny', 'tiny_deep', 'small', 'small_deep', 
                         'sr_tiny', 'sr_tiny_666', 'sr_tiny_mh', 'sr_small_mh', 'sr_small']

ModelEma = ModelEmaV2


def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--val-bs', default=64, type=int)

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--no-prefetcher', action='store_false', dest='use_prefetcher',
                        help='')
    parser.set_defaults(use_prefetcher=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # Misc
    parser.add_argument('--print-freq', default=100, type=int, 
                        help='Interval of iterations to print training/eval info.')

    # Knowledge distillation
    parser.add_argument('--teacher-ckpt-path', default=None, type=str, 
                        help='Path to teacher model checkpoint (.pth or .pth.tar).')

    # for flexible vit
    parser.add_argument('--network-def', default=None, type=str, 
                        help='Network def to construct network when --model is flexible_vit_patch16_224 or flexible_vit_patch16_224_supernet')
    
    # for super-network training
    parser.add_argument('--search-space', default=None, type=str, choices=_SEARCH_SPACE_CHOICES, 
                        help='Numbers of channels to keep when training super-networks.')
    parser.add_argument('--example-per-arch', default=None, type=int, 
                        help='Sub batch size when using multi-architecture super-network training.')
    parser.add_argument('--num-warmup-epochs', default=30, type=int, 
                        help='Number of warmup epochs when training super-networks.')
    parser.add_argument('--single-arch', action='store_true', dest='single_arch', default=False,
                        help='Use single-architecture or multi-architecture super-network training.')
    parser.add_argument('--hybrid-arch', action='store_true', dest='hybrid_arch', default=None,
                        help='When using multi-architecture super-network training, sample one width for embedding layer.')
    parser.add_argument('--use-holdout', action='store_true', dest='use_holdout', default=False,
                        help='Use sub-train and sub-eval set for super-network training.')
    
    # for finetuning from supernet's weights
    parser.add_argument('--resume-supernet-weights', default=None, type=str, help='Path to weights of super-network.')
    
    # for shifted patch token mixup
    parser.add_argument('--use-patch-mixup', action='store_true', dest='use_patch_mixup', default=False)
    parser.add_argument('--mixup-patch-len', default=4, type=int, help='Length of patches when using shifted patch token mixup.')
    parser.add_argument('--switch-prob', default=0.5, type=float)
    
    # for finetuning at higher resolution
    parser.add_argument('--finetune', default=None, type=str, 
                        help='Path to trained checkpoint.')
    
    return parser

    
def main(args):
    utils.init_distributed_mode(args)
    
    if not hasattr(args, 'gpu'):
        args.gpu = 0
    is_rank0 = (args.gpu == 0)
    _log = FileLogger(is_master=utils.is_main_process(), is_rank0=is_rank0, output_dir=args.output_dir)
    _log.info(args)
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed #+ utils.get_rank()
    #if args.single_arch:
    #    _log.info('Use the same random seed when using single-architecture super-network training')
    #else:
    seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    if not args.use_prefetcher:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    # torch.utils.data.DataLoader
    # MultiEpochDataLoader is faster but seems to change the random behavior
    data_loader_train = MultiEpochsDataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if args.use_prefetcher:
        data_loader_val = build_dataloader(is_train=False, args=args)
    else:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.val_bs,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False
        )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    patch_mixup_fn = None
    if args.use_patch_mixup:
        patch_mixup_fn = SwitchTokenMix(args.mixup_patch_len, 
            switch_prob=args.switch_prob, 
            num_classes=args.nb_classes, smoothing=args.smoothing)
        mixup_fn = None
        _log.info('Use Shifted Patch Token Mixup: {} and remove Mixup'.format(patch_mixup_fn))

    _log.info(f"Creating model: {args.model}")
    if args.network_def:
        network_def = literal_eval(args.network_def)
    else:
        network_def = None
    model_kwargs = {'model_name': args.model, 'pretrained': False, 
        'num_classes': args.nb_classes, 'drop_rate': args.drop, 
        'drop_path_rate': args.drop_path, 'drop_block_rate': args.drop_block}
    if args.model in _MODELS_USE_NETWORK_DEF:
        model_kwargs['network_def'] = network_def 
        
    # for super-network training
    if args.model in _MODELS_FOR_SUPERNET:
        model_kwargs['example_per_arch'] = args.example_per_arch
        model_kwargs['num_warmup_epochs'] = args.num_warmup_epochs
        model_kwargs['single_arch'] = args.single_arch
        
        if args.hybrid_arch is not None:
            model_kwargs['hybrid_arch'] = args.hybrid_arch
        
        config = getattr(supernet_config, args.search_space)
        model_kwargs['num_channels_to_keep'] = config.num_channels_to_keep
        _log.event(config.num_channels_to_keep)
        
    model = create_model(**model_kwargs)
    
    # TODO: finetuning
    if args.finetune:
        state_dict = load_interpolated_state_dict(model_state_dict=model.state_dict(), ckpt_path=args.finetune)
        model.load_state_dict(state_dict)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else None)

    #model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info('number of params: {}'.format(n_parameters))

    # Load teacher model
    teacher_model = None
    if args.teacher_ckpt_path is not None:
        teacher_model = _load_teacher_model(args.teacher_ckpt_path)
        teacher_model.to(device)
        #if args.distributed:
        #    teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu])
        
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0. or args.use_patch_mixup:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
        # when evaluating, use model_ema
        if args.eval and 'model_ema' in checkpoint.keys():
            model_without_ddp.load_state_dict(checkpoint['model_ema'])            

    if args.resume_supernet_weights:
        _log.info('Inherit weights from a super-network checkpoint: {}'.format(args.resume_supernet_weights))
        checkpoint = torch.load(args.resume_supernet_weights, map_location='cpu')
        subnet_state_dict = model_without_ddp.state_dict()
        subnet_state_dict = get_sub_state_dict(source_dict=checkpoint['model'], 
            sub_dict=subnet_state_dict)
        model_without_ddp.load_state_dict(subnet_state_dict)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args.print_freq, logger=_log)
        #print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        _log.info(f"Accuracy: {test_stats['acc1']:.2f}%")
        return

    _log.info(args)
    _log.info(model)
    if hasattr(model_without_ddp, 'no_weight_decay'):
        _log.info('Parameters without weight decay: {}'.format(model_without_ddp.no_weight_decay()))
        
    #_log.info("Start training")
        
    # configure architecture sampling
    arch_sample = None
    if args.model in _MODELS_FOR_SUPERNET:
        if args.single_arch:
            arch_sample = 'single'
            assert not args.hybrid_arch
        elif args.hybrid_arch:
            arch_sample = 'hybrid'
            assert not args.single_arch
        else:
            arch_sample = 'multi'
    _log.info('Architecture sample: {}'.format(arch_sample))
    
    # check testing at higher resolution
    if args.finetune:
        test_stats = evaluate(data_loader_val, model, device, args.print_freq, logger=_log)
        _log.info('Accuracy at higher resolution without finetuning: {:.2f}%'.format(test_stats['acc1']))
    
    start_time = time.time()
    max_accuracy = 0.0
    max_ema_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        
        lr_scheduler.step(epoch)
        
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        if hasattr(model_without_ddp, 'set_epoch'):
            model_without_ddp.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn, 
            args.print_freq, 
            teacher_model=teacher_model, 
            logger=_log, 
            arch_sample=arch_sample, 
            patch_mixup_fn=patch_mixup_fn)
        
        test_stats = evaluate(data_loader_val, model, device, args.print_freq, logger=_log)
        #print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.2f}%")
        _log.info(f"Accuracy: {test_stats['acc1']:.2f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        _log.info(f'Max accuracy: {max_accuracy:.2f}%\n')
        
        # for EMA
        if model_ema is not None:
            ema_test_stats = evaluate(data_loader_val, model_ema.module, device, args.print_freq, logger=_log)
            _log.info('EMA acc. = {:.2f}%'.format(ema_test_stats['acc1']))
            max_ema_acc = max(max_ema_acc, ema_test_stats['acc1'])
            _log.info('Max EMA acc. = {:.2f}%\n'.format(max_ema_acc))

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if model_ema is not None:
            log_stats['ema_test'] = ema_test_stats['acc1']

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            checkpoint_path = output_dir / 'checkpoint.pth.tar'
            save_checkpoint_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            }
                
            if model_ema is not None:
                save_checkpoint_dict['model_ema'] = get_state_dict(model_ema)
            torch.save(save_checkpoint_dict, checkpoint_path)
            
            if epoch % 10 == 9:
                torch.save(save_checkpoint_dict, output_dir / 'epoch@{}_checkpoint.pth.tar'.format(epoch))
            if test_stats['acc1'] == max_accuracy:
                torch.save(save_checkpoint_dict, output_dir / 'best_checkpoint.pth.tar')
            if model_ema is not None and ema_test_stats['acc1'] == max_ema_acc:
                torch.save(save_checkpoint_dict, output_dir / 'best_ema_checkpoint.pth.tar')
            
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    _log.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)