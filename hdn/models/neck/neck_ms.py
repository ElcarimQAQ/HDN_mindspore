# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mindspore.nn as nn


class AdjustLayer(nn.Cell):
    def __init__(self, in_channels, out_channels, cut=True, cut_left=4, cut_num=7):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='vaild'),
            nn.BatchNorm2d(out_channels),
            )
        self.cut = cut
        self.cut_left = cut_left
        self.cut_num = cut_num

    def forward(self, x):
        x = self.downsample(x)
        if self.cut and x.size(3) < 20:
            l = self.cut_left
            r = l + self.cut_num
            x = x[:, :, l:r, l:r]

        return x


class AdjustAllLayer(nn.Cell):
    def __init__(self, in_channels, out_channels, cut=True, cut_left=4, cut_num=7):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0], cut, cut_left, cut_num)
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i], cut, cut_left, cut_num))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out
