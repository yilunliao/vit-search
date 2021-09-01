# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torch.nn as nn

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils

#from contextlib import suppress # for amp


class KnowledgeDistillationLoss(nn.Module):
    
    def __init__(self, hard_distill=True, soft_temperature=3.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.hard_distill = hard_distill
        if self.hard_distill:
            self.nn_module = nn.CrossEntropyLoss()
            self.soft_temperature = None
        else:
            self.nn_module = nn.LogSoftmax()
            self.soft_temperature = float(soft_temperature)
    
    
    def forward(self, x, teacher_output):
        if self.hard_distill:
            teacher_output = torch.argmax(teacher_output, dim=1)
            loss = self.nn_module(x, teacher_output)
        else:
            t = self.soft_temperature
            teacher_output = nn.functional.softmax(teacher_output / t, dim=1)
            loss = torch.mean(torch.sum(- teacher_output * self.nn_module(x / t), 1)) * (t * t)
        return loss
    
    
    def extra_repr(self):
        info = 'hard_distill={}'.format(self.hard_distill)
        if not self.hard_distill:
            info = info + '\nsoft_temperature={}'.format(self.soft_temperature)
        info = info + super(KnowledgeDistillationLoss, self).extra_repr()
        return info
          

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    print_freq: int = 100, 
                    teacher_model=None, hard_distill=True, alpha: float = 0.5, 
                    logger=None, 
                    arch_sample=False, 
                    patch_mixup_fn=None):
    '''
        Add:
            `print_freq`: for displaying logging.
            `teacher_model`: for knowledge distillation.
            `hard_distillation`: use hard distillation r soft distillation.
            `alpha`: coefficient balancing `criterion` and KD loss.
            `logger`: output training log to .log files.
            `arch_sample`: specify how architectures are sampled during super-network training.
                `None` for standard training. "single" for single-architecture super-network training.
                "multi" for multi-architecture super-network training.
            `patch_mixup_fn`: for Shifted Patch Token Mixup.
    '''
    
    # TODO fix this for finetuning
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ", logger=logger)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_out = logger.info if logger else print
    
    header = 'Epoch: [{}]'.format(epoch)
    #print_freq = 10
    # this attribute is added by timm on one optimizer (adahessian)
    is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order    
    
    # knowledge distillation
    kd_criterion = None
    if teacher_model is not None:
        kd_criterion = KnowledgeDistillationLoss(hard_distill=hard_distill)
        teacher_model.eval()
        
    # for super-network training
    _sample_epoch_offset = 10000 # prevent similar single-architecture sampling 

    train_iter = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            assert patch_mixup_fn is None
        if patch_mixup_fn is not None:
            samples, targets, patch_targets, patch_output_type = patch_mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_output = teacher_model(samples)
            
            # for single architecture supernet training 
            if arch_sample is not None:
                rng = torch.random.get_rng_state()
                if arch_sample == 'single' or arch_sample == 'hybrid':
                    torch.manual_seed(epoch * _sample_epoch_offset + train_iter)
                elif arch_sample == 'multi':
                    # Use the original random state.
                    # However, this random state will be resumed to make sure single/multi-arch sampling
                    # see exactly the same data.
                    pass 
                else:
                    raise ValueError('arch_sample has invalid value {}.'.format(arch_sample))
            else:
                rng = None
                
            # For classification label only
            if patch_mixup_fn is None:
                outputs = model(samples)
                if isinstance(outputs, tuple):
                    output_cls = outputs[0]
                    output_dst = outputs[1]
                else:
                    output_cls = outputs
                    output_dst = outputs
                    
                loss = criterion(output_cls, targets)
                
                if teacher_model is not None:
                    kd_loss = kd_criterion(output_dst, teacher_output)
                    loss = loss * (1 - alpha) + kd_loss * alpha    
            
            else:
                # For both classification token and patch prediction
                # Not consider knowledge distillation here
                cls_pred, patch_pred = model(samples, patch_output_type=patch_output_type)
                loss = criterion(cls_pred, targets)
                
                #+ criterion(patch_pred, patch_targets) * alpha
                if patch_output_type == 'seq':
                    loss = loss + criterion(patch_pred, patch_targets) #* alpha
                elif patch_output_type == 'avg':
                    loss = loss + criterion(patch_pred, targets) #* alpha
                else:
                    raise ValueError()
                
            # resume random state
            if arch_sample is not None:
                torch.random.set_rng_state(rng)
            train_iter = train_iter + 1
            
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print_out("Loss is {}, stopping training".format(loss_value))
            print('Loss is {}. Stopping training.'.format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print_out("Averaged stats: {}".format(metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


#@torch.no_grad()
def evaluate(data_loader, model, device, print_freq=100, logger=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ", logger=logger)
    print_out = logger.info if logger else print
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    
    # for joint classifier softmax
    softmax_m = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, target in metric_logger.log_every(data_loader, print_freq, header):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
            
            if isinstance(output, tuple):
                output_cls = output[0]
                output_dst = output[1]
            else:
                output_cls = output
                output_dst = None
                
            loss = criterion(output_cls, target)
            acc1, acc5 = accuracy(output_cls, target, topk=(1, 5))
    
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            
            if output_dst is not None:
                dst_acc1, dst_acc5 = accuracy(output_dst, target, topk=(1, 5))
                metric_logger.meters['dst_acc1'].update(dst_acc1.item(), n=batch_size)
                metric_logger.meters['dst_acc5'].update(dst_acc5.item(), n=batch_size)
                
                output_jnt = softmax_m(output_cls) + softmax_m(output_dst)
                jnt_acc1, jnt_acc5 = accuracy(output_jnt, target, topk=(1, 5))
                metric_logger.meters['jnt_acc1'].update(jnt_acc1.item(), n=batch_size)
                metric_logger.meters['jnt_acc5'].update(jnt_acc5.item(), n=batch_size)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    #info_str = '* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #.format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss)
    info_str = 'Acc@1: {:.2f}, Acc@5: {:.2f}, loss: {:.2f}'.format(
        metric_logger.acc1.global_avg, 
        metric_logger.acc5.global_avg, 
        metric_logger.loss.global_avg
        )
    
    if 'dst_acc1' in metric_logger.meters.keys():
        info_str = info_str + ', Distill Acc@1: {:.2f}'.format(metric_logger.dst_acc1.global_avg)
        info_str = info_str + ', Distill Acc@5: {:.2f}'.format(metric_logger.dst_acc5.global_avg)
        info_str = info_str + ', Joint Acc@1: {:.2f}'.format(metric_logger.jnt_acc1.global_avg)
        info_str = info_str + ', Joint Acc@5: {:.2f}\n'.format(metric_logger.jnt_acc5.global_avg)
    else:
        info_str = info_str + '\n'
    print_out(info_str)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
