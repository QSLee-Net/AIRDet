# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def get_size_ratio(self, image_size):
        target_size = random.choice(self.min_size)
        w, h = image_size
        t_w, t_h = target_size, target_size
        r = min(t_w / w, t_h / h)
        o_w, o_h = int(w * r), int(h * r)
        return (o_w, o_h)

    def __call__(self, image, target=None):
        h, w = image.shape[:2]
        size = self.get_size_ratio((w, h))

        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image, dtype=np.float32)
        if isinstance(target, list):
            target = [t.resize(size) for t in target]
        elif target is None:
            return image, target
        else:
            target = target.resize(size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[:, :, ::-1]
            image = np.ascontiguousarray(image, dtype=np.float32)
            if target is not None:
                target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return torch.from_numpy(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]]
        else:
            image = image
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        return image, target
