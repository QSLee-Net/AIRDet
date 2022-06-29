# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


import copy

from .gfocal_v2_tiny import GFocalHead_Tiny
from .yolox_head import YOLOXHead


def build_head(cfg):

    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "GFocalV2":
        return GFocalHead_Tiny(**head_cfg)
    elif name == "YOLOX":
        return YOLOXHead(**head_cfg)
