#!/usr/bin/env python
# coding=utf-8
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.
from .backbones.darknet import CSPDarknet
from .backbones.zennas import ZenNas
from .backbones import build_backbone
from .necks.pafpn import PAFPN
from .necks.giraffe_fpn import GiraffeNeck
from .necks import build_neck
from .heads.yolox_head import YOLOXHead
from .heads.gfocal_v2_tiny import GFocalHead_Tiny
from .heads import build_head
from .losses.losses import IOUloss
