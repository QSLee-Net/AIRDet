#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from airdet.config import Config as MyConfig
from airdet.config.base import airdet_Shuffle

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.miscs.eval_interval_epochs = 1

        self.model = airdet_Shuffle

        self.model.neck.feature_info = [dict(num_chs=116, reduction=8),
                                        dict(num_chs=232, reduction=16),
                                        dict(num_chs=464, reduction=32)]