# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from . import transforms as T
from airdet.augmentations.scale_aware_aug import SA_Aug


def build_transforms(cfg, is_train=True):
    if is_train:
        if cfg.training.input_min_size_range[0] == -1:
            min_size = cfg.training.input_min_size
        else:
            assert len(cfg.training.input_min_size_range) == 2, \
                "input_min_size_range_train must have two elements (lower bound, upper bound)"
            min_size = list(range(
                cfg.training.input_min_size_range[0],
                cfg.training.input_min_size_range[1] + 1
            ))
        max_size = cfg.training.input_max_size
        flip_prob = cfg.training.augmentation.flip_prob  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.testing.input_min_size
        max_size = cfg.testing.input_max_size
        flip_prob = 0.0

    to_bgr255 = cfg.dataset.input_to_bgr255
    normalize_transform = T.Normalize(
        mean=cfg.dataset.input_pixel_mean, std=cfg.dataset.input_pixel_std, to_bgr255=to_bgr255
    )

    transform = [
            T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]

    if cfg.training.augmentation.use_autoaug and is_train:
        transform += [SA_Aug(cfg)]

    transform = T.Compose(transform)

    return transform


