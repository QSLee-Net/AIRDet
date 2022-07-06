#!/usr/bin/env python
# -*- encoding: utf-8 -*

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

from ..core.base_ops import conv_3x3_bn, ShuffleBottleneck


class ShuffleNet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
    ):
        super(ShuffleNet, self).__init__()
        
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if wid_mul == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif wid_mul == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif wid_mul == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif wid_mul == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise ValueError(
                """{} groups is not supported for
                       1x1 Grouped Convolutions""".format(num_groups))

        # stem
        self.stem  = conv_3x3_bn(3, self.stage_out_channels[1], 2)

        # stage 1    
        self.stage1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # stage 2
        self.stage2 = []
        input_channel_2 = self.stage_out_channels[1]
        output_channel_2 = self.stage_out_channels[2]
        for i in range(4):
            if i == 0:
                self.stage2.append(ShuffleBottleneck(input_channel_2, output_channel_2, 2, 2))
            else:
                self.stage2.append(ShuffleBottleneck(input_channel_2, output_channel_2, 1, 1))
            input_channel_2 = output_channel_2
        self.stage2 = nn.Sequential(*self.stage2)

        # stage 3
        input_channel_3 = self.stage_out_channels[2]
        output_channel_3 = self.stage_out_channels[3]
        self.stage3 = []
        for i in range(8):
            if i == 0:
                self.stage3.append(ShuffleBottleneck(input_channel_3, output_channel_3, 2, 2))
            else:
                self.stage3.append(ShuffleBottleneck(input_channel_3, output_channel_3, 1, 1))
            input_channel_3 = output_channel_3
        self.stage3 = nn.Sequential(*self.stage3)

        # stage 4
        input_channel_4 = self.stage_out_channels[3]
        output_channel_4 = self.stage_out_channels[4]
        self.stage4 = []
        for i in range(4):
            if i == 0:
                self.stage4.append(ShuffleBottleneck(input_channel_4, output_channel_4, 2, 2))
            else:
                self.stage4.append(ShuffleBottleneck(input_channel_4, output_channel_4, 1, 1))
            input_channel_4 = output_channel_4
        self.stage4 = nn.Sequential(*self.stage4)


    def init_weights(self, pretrain=None):
        if pretrain is None:
            return
        else:
            pretrained_dict = torch.load(pretrain, map_location='cpu')['state_dict']
            new_params = self.state_dict().copy()
            for k, v in pretrained_dict.items():
                ks = k.split('.')
                if ks[0] == 'fc' or ks[-1] == 'total_ops' or ks[-1] == 'total_params':
                    continue
                else:
                    new_params[k] = v

            self.load_state_dict(new_params)
            print(f" load pretrain backbone from {pretrain}")


    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.stage1(x)
        outputs["stage1"] = x
        x = self.stage2(x)
        outputs["stage2"] = x
        x = self.stage3(x)
        outputs["stage3"] = x
        x = self.stage4(x)
        outputs["stage4"] = x
        features_out = [outputs["stem"], outputs["stage1"], outputs["stage2"], outputs["stage3"], outputs["stage4"]]

        return features_out
