# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


import copy

from .darknet import CSPDarknet

def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "CSPDarknet":
        return CSPDarknet(**backbone_cfg)
