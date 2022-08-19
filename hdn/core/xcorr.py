from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mindspore as ms
from  mindspore import ops, nn
from mindspore import numpy as np
import numpy


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = ops.Conv2D(px, pk)
        out.append(po)
    out = ops.Concat(0)(out)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = ops.Conv2D(px, pk, group=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.shape[0]
    channel = kernel.shape[1]
    x = x.view(1, batch*channel, x.shape[2], x.shape[3])
    kernel = kernel.view(batch*channel, 1, kernel.shape[2], kernel.shape[3])
    out = ops.Conv2D(kernel.shape[0],kernel.shape[2],group=batch*channel)(x, kernel)
    out = out.view(batch, channel, out.shape[2], out.shape[3])
    return out

def xcorr_depthwise_circular(x, kernel):
    """depthwise cross correlation with circular
        This corr is specular for logpolar coordinates
    """
    batch = kernel.shape[0]
    channel = kernel.shape[1]
    # padding the input data
    # x = F.pad(x, (0, 0, x.size(2)//2, x.size(2)//2), "circular")  # rotation is circular
    x = np.pad(x, pad_width=((0, 0), (0, 0), (x.shape[2]//2, x.shape[2]//2), (0, 0)), mode="wrap")

    # x = F.pad(x, (x.size(3)//2, x.size(3)//2, 0, 0), "replicate")  # polar coordinate lacks info, so use the nearby data
    x = ms.Tensor(numpy.pad(x.asnumpy(), pad_width=((0,0), (0,0), (0,0), (x.shape[3]//2, x.shape[3]//2)), mode= "edge"))
    x = x.view(1, batch*channel, x.shape[2], x.shape[3])
    kernel = kernel.view(batch*channel, 1, kernel.shape[2], kernel.shape[3])
    out = ops.Conv2D(kernel.shape[0],kernel.shape[2],group=batch*channel)(x, kernel)
    out = out.view(batch, channel, out.shape[2], out.shape[3]) # the size should be the same as input x
    return out
