# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modified from RTD-Net (https://github.com/MCG-NJU/RTD-Action)

PointTAD Training and Inference functions.

"""

import json
import math
import sys
from typing import Iterable

import torch
from termcolor import colored

import util.misc as utils
from datasets.evaluate import Evaluator

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    args,
                    postprocessors=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
   
    metric_logger.add_meter(
        'class_error', utils.SmoothedValue(window_size=1,
                                            fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    max_norm = args.clip_max_norm

    for vid_name_list, locations, x, targets, num_frames, base \
        in metric_logger.log_every(data_loader, print_freq, header):

        x = x.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(x)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
                     if k in weight_dict)
        n_parameters = sum(p.numel() for p in model.parameters())
        losses += 0 * n_parameters
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg
            for k, meter in metric_logger.meters.items()}, loss_dict

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device, args):
    print(colored('evaluate', 'red'))
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'class_error', utils.SmoothedValue(window_size=1,
                                            fmt='{value:.2f}'))
    header = 'Test:'

    evaluator = Evaluator()

    video_pool = list(load_json(args.annotation_path).keys())
    video_pool.sort()
    video_dict = {i: video_pool[i] for i in range(len(video_pool))}
    for vid_name_list, locations, x, targets, num_frames, base in metric_logger.log_every(
            data_loader, 10, header):
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(x)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        results, dense_results = postprocessors['results'](outputs, num_frames, base)

        for target, output, dense_res, base_loc in zip(targets, results, dense_results, base):
            vid = video_dict[target['video_id'].item()]
            dense_gt = target['dense_gt']
            if args.dense_result:
                torch.save(dense_res, f'dense_results/{vid}_{base_loc}_dense')
            evaluator.update(vid, output, base_loc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    return evaluator, loss_dict
