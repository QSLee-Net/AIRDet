# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


import copy

from .darknet import CSPDarknet
from .mobilenet import MobileNet
from .shufflenet import ShuffleNet

def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "CSPDarknet":
        return CSPDarknet(**backbone_cfg)
    elif name == "MobileNet":
        return MobileNet(**backbone_cfg)
    elif name == "ShuffleNet":
        return ShuffleNet(**backbone_cfg)