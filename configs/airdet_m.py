#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from airdet.config import Config as MyConfig

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.model.backbone.dep_mul = 0.33
        self.model.backbone.wid_mul = 0.50

        self.model.neck.fpn_channels = [192, 256, 512]
        self.model.neck.out_fpn_channels = [192, 256, 512]
        self.model.neck.depth_multiplier = 5
        self.model.neck.width_multiplier = 1.0
        self.model.neck.feature_info = [dict(num_chs=128, reduction=8),
                                        dict(num_chs=256, reduction=16),
                                        dict(num_chs=512, reduction=32)]

        self.model.head.feat_channels = [128, 128, 128]
        self.model.head.in_channels = [192, 256, 512]
        self.model.head.conv_groups = 2


