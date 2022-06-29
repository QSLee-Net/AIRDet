# This file mainly comes from
# https://github.com/dvlab-research/SA-AutoAug/blob/master/FCOS/fcos_core/augmentations/scale_aware_aug.py
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


import copy
import torch
import torchvision
from airdet.augmentations.box_level_augs.box_level_augs import Box_augs
from airdet.augmentations.box_level_augs.color_augs import color_aug_func
from airdet.augmentations.box_level_augs.geometric_augs import geometric_aug_func
from airdet.utils import get_world_size


class SA_Aug(object):
    def __init__(self, cfg):
        sada_cfg = cfg.training.augmentation.autoaug
        autoaug_list = sada_cfg.autoaug_params
        num_policies = sada_cfg.num_subpolicies
        max_iters = (cfg.training.total_epochs -
                        cfg.training.no_aug_epochs) * cfg.training.iters_per_epoch
        scale_splits = sada_cfg.scale_splits
        box_prob = sada_cfg.box_prob

        self.use_box_level = sada_cfg.use_box_lvl
        self.use_box_color = sada_cfg.use_box_color
        self.use_box_geo = sada_cfg.use_box_geo
        self.dynamic_scale_split = sada_cfg.use_dynamic_scale

        img_aug_list = autoaug_list[:4]
        img_augs_dict = {'zoom_out':{'prob':img_aug_list[0]*0.05, 'level':img_aug_list[1]},
                         'zoom_in':{'prob':img_aug_list[2]*0.05, 'level':img_aug_list[3]}}

        box_aug_list = autoaug_list[4:]
        color_aug_types = list(color_aug_func.keys())
        geometric_aug_types = list(geometric_aug_func.keys())
        policies = []
        for i in range(num_policies):
            _start_pos = i * 6
            sub_policy = [(color_aug_types[box_aug_list[_start_pos+0]%len(color_aug_types)], box_aug_list[_start_pos+1]* 0.1, box_aug_list[_start_pos+2], ), # box_color policy
                          (geometric_aug_types[box_aug_list[_start_pos+3]%len(geometric_aug_types)], box_aug_list[_start_pos+4]* 0.1, box_aug_list[_start_pos+5])] # box_geometric policy
            policies.append(sub_policy)

        _start_pos = num_policies * 6
        scale_ratios = {'area': [box_aug_list[_start_pos+0], box_aug_list[_start_pos+1], box_aug_list[_start_pos+2]],
                        'prob': [box_aug_list[_start_pos+3], box_aug_list[_start_pos+4], box_aug_list[_start_pos+5]]}

        box_augs_dict = {'policies': policies, 'scale_ratios': scale_ratios}


        self.box_augs = Box_augs(box_augs_dict=box_augs_dict, max_iters=max_iters, scale_splits=scale_splits, box_prob=box_prob, dynamic_scale_split=self.dynamic_scale_split, use_color = self.use_box_color, use_geo=self.use_box_geo)

        self.max_iters = max_iters

        self.count = 0
        num_gpus = get_world_size()
        self.batch_size = cfg.training.images_per_batch // num_gpus
        self.num_workers = cfg.dataset.num_workers
        self.count = self.count + \
            (cfg.training.start_epoch * cfg.training.iters_per_epoch / self.num_workers * self.batch_size)
        if self.num_workers==0:
            self.num_workers += 1

    def __call__(self, tensor, target):
        iteration = self.count // self.batch_size * self.num_workers
        tensor = copy.deepcopy(tensor)
        target = copy.deepcopy(target)
        if self.use_box_level:
            tensor, target = self.box_augs(tensor, target, iteration=iteration)
        self.count += 1

        return tensor, target
