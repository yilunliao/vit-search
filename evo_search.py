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
import copy
import pickle

from pathlib import Path

from timm.models import create_model

from datasets import build_dataset, build_dataloader
from engine import train_one_epoch, evaluate
from samplers import RASampler
import models
import utils

#from nets.create_model import my_create_model
import nets

# for super-network training
import supernet_config

from logger import FileLogger
from utils import _load_teacher_model
from ast import literal_eval

# for computing MAC/FLOPs
from network_utils.compute_flop_mac import ComputationEstimator
# for evolutionary search
from search_utils.evolver import PopulationEvolver
from nets.net_utils import get_sub_state_dict

from functools import partial


_MODELS_USE_NETWORK_DEF = ['flexible_vit_patch16_224', 'flexible_vit_patch16_224_supernet', 
                           'flexible_vit_patch16_192', 'flexible_vit_patch16_192_supernet', 
                           'flexible_vit_sr_patch14_224', 'flexible_vit_sr_patch14_224_supernet', 
                           'flexible_vit_sr_distill_patch14_224', 'flexible_vit_sr_distill_patch14_224_supernet', 
                           'flexible_vit_sr_patch14_224_patch_output', 'flexible_vit_sr_patch14_224_patch_output_supernet']
#_MODELS_FOR_SUPERNET = ['flexible_vit_patch16_224_supernet', 'flexible_vit_patch16_192_supernet']

_SEARCH_SPACE_CHOICE = ['tiny', 'tiny_deep', 'small', 'small_deep', 'sr_tiny', 'sr_tiny_666', 
                        'sr_tiny_mh', 'sr_small_mh', 'sr_small']

# patch size for building ComputationEstimator
_USE_PATCH16 = []
_USE_PATCH14 = []
for model_name in _MODELS_USE_NETWORK_DEF:
    if 'patch14' in model_name:
        _USE_PATCH14.append(model_name)
    elif 'patch16' in model_name:
        _USE_PATCH16.append(model_name)


def get_args_parser():
    parser = argparse.ArgumentParser('ViT-NAS evolutionary search', add_help=False)
    parser.add_argument('--val-bs', default=64, type=int) # for re-use build_dataloader
    parser.add_argument('--model-path', default=None, type=str, 
                        help='Path to super-network checkpoint')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--model', default='flexible_vit_patch16_224', type=str, 
                        help='Model type to evaluate') 
    parser.add_argument('--model-ema', action='store_true', dest='model_ema',
                        help='')
    parser.set_defaults(model_ema=False)
    
    parser.add_argument('--num-gpu', default=1, type=int, 
                        help='Number of GPUS used to evaluate samples')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    
    parser.add_argument('--seed', default=0, type=int)
    #parser.add_argument('--resume', default='', help='resume from checkpoint')
    #parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                    help='start epoch')
    #parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--num_workers', default=8, type=int)
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

    # for flexible vit
    parser.add_argument('--network-def', default=None, type=str, 
                        help='Network def to construct network when --model is flexible_vit_patch16_224 or flexible_vit_patch16_224_supernet')
    
    # for searching
    parser.add_argument('--search-space', default=None, type=str, choices=_SEARCH_SPACE_CHOICE, 
                        help='Numbers of channels to keep when training super-networks.')
    parser.add_argument('--no-use-holdout', action='store_false', dest='use_holdout', default=True,
                        help='Use sub-train and sub-eval set for evolutionary search.')
    
    # search resource & constraint
    parser.add_argument('--constraint-type', default='MAC', type=str, choices=['MAC'])
    parser.add_argument('--constraint-value', default=None, type=float, 
                        help='Search sub-networks with complexity less than this value.')
    
    # evolutionary search hyper-parameters
    parser.add_argument('--init-popu-size', default=500, type=int, 
                        help='Initial population size, which determines how many sub-networks are randomly sampled at the first iteration.')
    parser.add_argument('--search-iter', default=20, type=int, 
                        help='Search iterations, with the first one being random sampling.')
    parser.add_argument('--parent-size', default=75, type=int, 
                        help='Number of top-performing sub-networks used to generate new sub-networks')
    parser.add_argument('--mutate-size', default=75, type=int, 
                        help='Number of sub-networks generated from mutation/crossover.')
    parser.add_argument('--mutate-prob', default=0.3, type=float, help='Mutation probability.')
    return parser


def pickle_save(obj, path):
    with open(path, 'wb') as file_id:
        pickle.dump(obj, file_id)
    

def write_results(obj, path, item_name_list=None):
    '''
        `obj`: is a list of Individual class
    '''
    
    if item_name_list is None:
        item_name_list = ['Idx', 'Acc', 'Network_def']
    else:
        assert len(item_name_list) == 3
    with open(path, 'w') as file_id:
        file_id.write('{}, {}, {}\n'.format(item_name_list[0], 
                                            item_name_list[1], 
                                            item_name_list[2]))
        for i in range(len(obj)):
            file_id.write('{}, {}, {}\n'.format(i, obj[i].score, obj[i].network_def))  


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
    #    seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = False
    cudnn.deterministic = True

    #dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    assert args.data_set == 'IMNET'
    args.nb_classes = 1000
    if not args.use_prefetcher:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.use_prefetcher:
        data_loader_val = build_dataloader(is_train=False, args=args, use_multi_epoch=True)
    else:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.val_bs,
            shuffle=False, num_workers=args.num_workers,
            pin_memory=args.pin_mem, drop_last=False
        )
    
    # initialize resource computation
    assert args.constraint_type == 'MAC', 'Currently only support MAC.'
    model_name = args.model
    distill = True if 'distill' in model_name else False
    if model_name in _USE_PATCH14:
        patch_size = 14
    elif model_name in _USE_PATCH16:
        patch_size = 16
    else:
        raise ValueError()
    compute_mac = ComputationEstimator(distill=distill, 
        input_resolution=args.input_size, patch_size=patch_size)
    _log.info('Computation Estimator: {}'.format(compute_mac))
    
    # search space
    if args.network_def:
        network_def = literal_eval(args.network_def)
    else:
        network_def = None
    config = getattr(supernet_config, args.search_space)
    num_channels_to_keep = config.num_channels_to_keep
    
    assert len(network_def) == len(num_channels_to_keep)
    
    # evolutionary search population
    popu_evolve = PopulationEvolver(largest_network_def=network_def, 
                                    num_channels_to_keep=num_channels_to_keep, 
                                    constraint=args.constraint_value, 
                                    compute_resource=compute_mac)
    
    # super-network weights
    checkpoint = torch.load(args.model_path, map_location='cpu')
    if args.model_ema:
        assert 'model_ema' in checkpoint.keys()
        supernet_state_dict = checkpoint['model_ema']
    else:
        supernet_state_dict = checkpoint['model'] 
    
    # common model arguments
    model_kwargs_template = {'model_name': args.model, 'pretrained': False, 
        'num_classes': args.nb_classes}
    assert args.model in _MODELS_USE_NETWORK_DEF
    
    # evolutionary search with distributed evaluation
    _dummy_log = FileLogger(is_master=False, is_rank0=False, output_dir=args.output_dir)
    _best_result_history = []
    for search_iter in range(args.search_iter):
        # generate sub-networks to evaluate
        if search_iter == 0:
            popu_evolve.random_sample(args.init_popu_size)
        else:
            popu_evolve.evolve_sample(parent_size=args.parent_size, 
                                      mutate_prob=args.mutate_prob, 
                                      mutate_size=args.mutate_size)
        # evaluate sub-networks
        for subnet_idx in range(len(popu_evolve.popu)):
            sub_network_def = popu_evolve.popu[subnet_idx].network_def
            _log.info('Iter: [{}][{}/{}]: {}'.format(search_iter, 
                subnet_idx, len(popu_evolve.popu), sub_network_def))
            
            model_kwargs = copy.deepcopy(model_kwargs_template)
            model_kwargs['network_def'] = sub_network_def
        
            #_log.info(f"Creating model: {args.model}")
            model = create_model(**model_kwargs)
            sub_state_dict = get_sub_state_dict(source_dict=supernet_state_dict, 
                                                sub_dict=model.state_dict())
            model.load_state_dict(sub_state_dict)
            #_log.info(model)
            
            # move to GPU & data parallel   
            model.to(device)
            #if args.num_gpu > 1:
            #    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
            if args.distributed:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                
            #n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            #_log.info('number of params: {}'.format(n_parameters))
    
            test_stats = evaluate(data_loader_val, model, device, args.print_freq, logger=_dummy_log)
            
            if 'dst_acc1' in test_stats.keys():
                popu_evolve.popu[subnet_idx].score = test_stats['dst_acc1']
                _log.info('Acc. = {:.2f}, Dst_acc. = {:.2f}, Jnt_acc. = {:.2f}\n'.format(test_stats['acc1'], test_stats['dst_acc1'], test_stats['jnt_acc1']))
            else:
                popu_evolve.popu[subnet_idx].score = test_stats['acc1']
                _log.info('Acc. = {:.2f}\n'.format(test_stats['acc1']))
            #_log.info(f"Accuracy: {test_stats['acc1']:.2f}%")
            #_log.info('\n')
        
        if utils.is_main_process() and args.output_dir:
            # create directory to save iter data
            Path(os.path.join(args.output_dir, 'iter@{}'.format(search_iter))).mkdir(parents=True, exist_ok=True)
            
            # save population 
            pickle_save(popu_evolve.popu, 
                path=os.path.join(args.output_dir, 'iter@{}'.format(search_iter), 'popu.pickle'))
            write_results(popu_evolve.popu, 
                path=os.path.join(args.output_dir, 'iter@{}'.format(search_iter), 'popu.txt'))        
            
        popu_evolve.update_history()
        popu_evolve.sort_history()
        _log.info('{}\n'.format(popu_evolve.history_popu[0]))
        _best_result_history.append(popu_evolve.history_popu[0])
        
        #print(popu_evolve.history_popu[0])
        if utils.is_main_process() and args.output_dir:
            # save history popu
            pickle_save(popu_evolve.history_popu, 
                        path=os.path.join(args.output_dir, 'iter@{}'.format(search_iter), 'history_popu.pickle'))
            # save top-`ags.parent_size` to a text file
            write_results(popu_evolve.history_popu[0:args.parent_size], 
                          path=os.path.join(args.output_dir, 'iter@{}'.format(search_iter), 'history_popu_top.txt'))
            # save best netwrok_def for each iteration
            write_results(_best_result_history, path=os.path.join(args.output_dir, 'summary.txt'), 
                          item_name_list=['Iter', 'Acc', 'Network_def'])
    
    for i in range(len(_best_result_history)):
        _log.info('Iter [{}]: Acc = {:.2f}, Network_def = {}'.format(i, _best_result_history[i].score, _best_result_history[i].network_def))
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ViT-NAS evolutionary search', parents=[get_args_parser()])
    args = parser.parse_args()  
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)