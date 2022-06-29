# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


from ..core.base_ops import BaseConv, Focus
import torch.nn as nn

class S2D_Chain(nn.Module):
    def __init__(self, channels=[128,256,512,1024,2048]):
        super(S2D_Chain, self).__init__()
        self.conv1 = BaseConv(3, 32, ksize=3, stride=2)
        self.conv2 = BaseConv(32, 64, ksize=3, stride=2)

        self.focus_8x = Focus(in_channels=64, out_channels=channels[0])
        self.focus_16x = Focus(in_channels=channels[0], out_channels=channels[1])
        self.focus_32x = Focus(in_channels=channels[1], out_channels=channels[2])
        self.focus_64x = Focus(in_channels=channels[2], out_channels=channels[3])
        self.focus_128x = Focus(in_channels=channels[3], out_channels=channels[4])

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pass

    def forward(self, x):
        outs = {}
        x_2x = self.conv1(x)
        outs['2x'] = x_2x
        x_4x = self.conv2(x_2x)
        outs['4x'] = x_4x
        x_8x = self.focus_8x(x_4x)
        outs['8x'] = x_8x
        x_16x = self.focus_16x(x_8x)
        outs['16x'] = x_16x
        x_32x = self.focus_32x(x_16x)
        outs['32x'] = x_32x
        x_64x = self.focus_64x(x_32x)
        outs['64x'] = x_64x
        x_128x = self.focus_128x(x_64x)
        outs['128x'] = x_128x

        features_out = [outs['2x'], outs['4x'], outs['8x'], outs['16x'], outs['32x'], outs['64x'], outs['128x']]

        return features_out

