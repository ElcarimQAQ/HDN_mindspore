from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mindspore as ms
import numpy as np
from mindspore import  nn, ops

from hdn.core.xcorr import xcorr_fast, xcorr_depthwise, xcorr_depthwise_circular
from hdn.models.head.ban import BAN


class DepthwiseXCorrCirc(nn.Cell):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorrCirc, self).__init__()
        self.conv_kernel = nn.SequentialCell(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, pad_mode='valid'),
                nn.BatchNorm2d(hidden),
                nn.ReLU(),
                )
        self.conv_search = nn.SequentialCell(
                nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, pad_mode='valid'),
                nn.BatchNorm2d(hidden),
                nn.ReLU(),
                )
        self.head = nn.SequentialCell(
                nn.Conv2d(hidden, hidden, kernel_size=1, pad_mode='valid'),
                nn.BatchNorm2d(hidden),
                nn.ReLU(),
                nn.Conv2d(hidden, out_channels, kernel_size=1, pad_mode='valid')
                )
        

    def construct(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise_circular(search, kernel)
        # feature_np = np.load('/home/lbyang/data/feature_np.npy')
        # feature = ms.Tensor(feature_np)
        out = self.head(feature) # Todo: 会有一定变化
        return out


class DepthwiseCircBAN(BAN):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2, weighted=False):
        super(DepthwiseCircBAN, self).__init__()
        self.cls = DepthwiseXCorrCirc(in_channels, out_channels, cls_out_channels)
        self.loc = DepthwiseXCorrCirc(in_channels, out_channels, 4)

    def construct(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        # cls = ms.Tensor(np.load('/home/lbyang/data/cls_np.npy'))
        # loc = ms.Tensor(np.load('/home/lbyang/data/loc_np.npy'))
        return cls, loc


class MultiCircBAN(BAN):
    def __init__(self, in_channels, cls_out_channels, weighted=False):
        super(MultiCircBAN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.insert_child_to_cell('box'+str(i+2), DepthwiseCircBAN(in_channels[i], in_channels[i], cls_out_channels))
        if self.weighted:
            self.cls_weight = ms.Parameter(ops.Ones()(len(in_channels), ms.float32))
            self.loc_weight = ms.Parameter(ops.Ones()(len(in_channels), ms.float32))
        self.loc_scale = ms.Parameter(ops.Ones()(len(in_channels), ms.float32))

    def construct(self, z_fs, x_fs):
        cls = []
        loc = []
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            box = getattr(self, 'box'+str(idx))
            c, l = box(z_f, x_f)
            cls.append(c)
            loc.append(l*self.loc_scale[idx-2])


        if self.weighted:
            cls_weight = ops.Softmax(0)(self.cls_weight)
            loc_weight = ops.Softmax(0)(self.loc_weight)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)
