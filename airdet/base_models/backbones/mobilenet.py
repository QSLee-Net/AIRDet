#!/usr/bin/env python
# -*- encoding: utf-8 -*

import torch.nn as nn
import math
from ..core.base_ops import h_sigmoid, h_swish, conv_1x1_bn, conv_3x3_bn, _make_divisible, InvertedResidual


class MobileNet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
    ):
        super(MobileNet, self).__init__()
        # setting of inverted residual blocks

        base_channels = _make_divisible(16 * wid_mul, 8)

        # stem
        self.stem  = conv_3x3_bn(3, base_channels, 2)

        # stage 1 [in, hidden, out, k, s, se, hs]
        self.stage1 = nn.Sequential(
                InvertedResidual(base_channels      , base_channels      , base_channels      , 3, 2, True , False)
        )

        # stage 2
        self.stage2 = nn.Sequential(
                InvertedResidual(base_channels      , int(base_channels * 4.5), int(base_channels * 1.5), 3, 2, False, False),
                InvertedResidual(int(base_channels * 1.5), int(base_channels * 5.5), int(base_channels * 1.5) , 3, 1, False, False),
        )

        # stage 3
        self.stage3 = nn.Sequential(
                InvertedResidual(int(base_channels * 1.5), int(base_channels * 6)  , int(base_channels * 2.5), 5, 2, True, True),
                InvertedResidual(int(base_channels * 2.5), int(base_channels * 15) , int(base_channels * 2.5), 5, 1, True, True),
                InvertedResidual(int(base_channels * 2.5), int(base_channels * 15) , int(base_channels * 2.5), 5, 1, True, True),
                InvertedResidual(int(base_channels * 2.5), int(base_channels * 7.5), int(base_channels * 3  ), 5, 1, True, True),
                InvertedResidual(int(base_channels * 3)  , int(base_channels * 9)  , int(base_channels * 3. ), 5, 1, True, True),
        )

        # stage 4
        self.stage4 = nn.Sequential(
                InvertedResidual(int(base_channels * 3), int(base_channels * 18) , int(base_channels * 6), 5, 2, True, True),
                InvertedResidual(int(base_channels * 6), int(base_channels * 36) , int(base_channels * 6), 5, 1, True, True),
                InvertedResidual(int(base_channels * 6), int(base_channels * 36) , int(base_channels * 6), 5, 1, True, True),
        )


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