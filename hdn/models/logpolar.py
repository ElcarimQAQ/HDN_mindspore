import cv2
import numpy as np
import math

import os
# import argparse


import matplotlib.pyplot as plt
from hdn.core.config import cfg
import mindspore as ms
from mindspore import nn, ops, context, Tensor, numpy
from mindspore.common.initializer import Normal, initializer,One
from hdn.utils.point import generate_points, generate_points_lp, lp_pick

# parser = argparse.ArgumentParser(description='siamese tracking')
# parser.add_argument('--cfg', type=str, default='config.yaml',
#                     help='configuration of tracking')
# args = parser.parse_args()

def getPolarImg(img, original = None):
    """
    some assumption that img W==H
    :param img: image
    :return: polar image
    """
    sz = img.shape
    # maxRadius = math.hypot(sz[0] / 2, sz[1] / 2)
    maxRadius = sz[1]/2
    m = sz[1] / math.log(maxRadius)
    o = tuple(np.round(original)) if original is not None else (sz[0] // 2, sz[1] // 2)
    result = cv2.logPolar(img, o, m, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR )
    #test
    # plt.imshow(result/255)
    # plt.show()
    # plt.imshow(img/255)
    # plt.show()
    # plt.close('all')
    return result

def getLinearPolarImg(img, original = None):
    """
    some assumption that img W==H
    :param img: image
    :return: polar image
    """
    sz = img.shape
    # maxRadius = math.hypot(sz[0] / 2, sz[1] / 2)
    maxRadius = sz[1]/2
    o = tuple(np.round(original)) if original is not None else (sz[0] // 2, sz[1] // 2)
    result = cv2.linearPolar(img, o, maxRadius, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR )
    # plt.imshow(result)
    # plt.show()
    # plt.imshow(img)
    # plt.show()
    return result



class STN_Polar(nn.Cell):
    """
    STN head
    """
    def __init__(self, image_sz):
        super(STN_Polar, self).__init__()
        self._orignal_sz = [image_sz//2, image_sz//2]  # sample center position

    def _prepare_grid(self, sz, delta):
        assert len(sz) == 2  # W, H
        x_ls = ops.LinSpace()(Tensor(0, ms.float32), Tensor(sz[0]-1, ms.float32), sz[0])
        y_ls = ops.LinSpace()(Tensor(0, ms.float32), Tensor(sz[1]-1, ms.float32), sz[1])

        # get log polar coordinates
        mag = math.log(sz[0]/2) / sz[0]
        # rho = (torch.exp(mag * x_ls) - 1.0) + delta[0]
        rho = (ops.Exp()(mag * x_ls) - 1.0)
        theta = y_ls * 2.0 * math.pi / sz[1] + delta[1]# add rotation
        y, x = ops.Meshgrid(indexing="ij")((theta, rho))
        cosy = ops.Cos()(y)
        siny = ops.Sin()(y)

        # construct final indices
        self.indices_x = ops.Mul()(x, cosy)
        self.indices_y = ops.Mul()(x, siny)

        # # test
        # y, x = torch.meshgrid([x_ls, y_ls])
        # self.indices_x = x.cuda()
        # self.indices_y = y.cuda()
    def _prepare_batch_grid(self, sz, delta, batch):
        assert len(sz) == 2  # W, H
        x_ls = ops.LinSpace()(Tensor(0, ms.float32), Tensor(sz[0] - 1, ms.float32), sz[0])
        y_ls = ops.LinSpace()(Tensor(0, ms.float32), Tensor(sz[1] - 1, ms.float32), sz[1])

        # get log polar coordinates
        mag = math.log(sz[0]/2) / sz[0]
        rho_batch = delta[0] + (ops.exp(mag * x_ls) - 1.0)
        theta_batch = delta[1] + y_ls * 2.0 * math.pi / sz[1]
        for rho, theta in rho_batch,theta_batch:
            y, x = ops.Meshgrid(indexing="ij")((theta, rho))
            cosy = ops.cos(y)
            siny = ops.sin(y)

            # construct final indices
            self.indices_x = ops.mul(x, cosy)
            self.indices_y = ops.mul(x, siny)



    def get_logpolar_grid(self, polar, sz):
        """
        This implementation is based on OpenCV source code to match the transformation.
        :param polar: N*2 N pairs of original of coordinates [-1.0, 1.0]
        :param sz: 4 the size of the output
        :return: N*W*H*2 the grid we generated
        """
        assert len(sz) == 4 # N, C, W, H
        batch = sz[0]
        # generate grid mesh
        x = self.indices_x # for multi-gpus
        y = self.indices_y
        expand_dims = ops.ExpandDims()
        indices_x = numpy.tile(x, (batch, 1, 1)) + expand_dims(expand_dims(polar[:, 0], 1), 1)
        indices_y = numpy.tile(y, (batch, 1, 1)) + expand_dims(expand_dims(polar[:, 1], 1), 1)
        # print('indices_x.shape',indices_x.shape)
        # print('indices_y.shape',indices_y.shape)
        # indices = ops.concat((indices_x.unsqueeze(3)/(sz[2]//2), indices_y.unsqueeze(3)/(sz[3]//2)), 3)
        indices = ops.Concat(3)((expand_dims(indices_x, 3)/(sz[2]//2), expand_dims(indices_y, 3)/(sz[3]//2)))
        return indices

    def construct(self, x, polar, delta=[0,0]):
        self._prepare_grid(self._orignal_sz, delta)
        grid = self.get_logpolar_grid(polar, x.shape)#[1 127 127 2]
        # self.test_polar_points(grid.cpu().squeeze(0).view(-1,2))
        x = ops.grid_sample(x, grid, interpolation_mode='bilinear', padding_mode='border')  #Todo，有问题

        # test plt log-polar img
        # x_lp_cpu = ops.Transpose()(x[0],(1,2,0)).asnumpy()
        # plt.imshow(x_lp_cpu/256)
        # fig = plt.figure()
        # fig,ax = plt.subplots(1,dpi=96)
        # ax.plot([polar[0], ], [polar[1], ], c='r', marker='x')
        # plt.show()
        # plt.close('all')
        return x, grid



class STN_LinearPolar(nn.Cell):
    """
    STN head
    """
    def __init__(self, image_sz):
        super(STN_LinearPolar, self).__init__()
        self._orignal_sz = [image_sz//2, image_sz//2]  # sample center position
        self._prepare_grid(self._orignal_sz)

    def _prepare_grid(self, sz):
        assert len(sz) == 2  # W, H
        x_ls = torch.linspace(0, sz[0]-1, sz[0])
        y_ls = torch.linspace(0, sz[1]-1, sz[1])

        # get linear polar coordinates
        maxR =sz[0]/2
        rho = maxR * x_ls / sz[0]
        theta = y_ls * 2.0 * math.pi / sz[1]
        y, x = torch.meshgrid([theta, rho])
        cosy = ops.cos(y)
        siny = ops.sin(y)

        # construct final indices
        self.indices_x = torch.mul(x, cosy)
        self.indices_y = torch.mul(x, siny)

        # # test
        # y, x = torch.meshgrid([x_ls, y_ls])
        # self.indices_x = x.cuda()
        # self.indices_y = y.cuda()


    def get_logpolar_grid(self, polar, sz):
        """
        This implementation is based on OpenCV source code to match the transformation.
        :param polar: N*2 N pairs of original of coordinates [-1.0, 1.0]
        :param sz: 4 the size of the output
        :return: N*W*H*2 the grid we generated
        """
        assert len(sz) == 4 # N, C, W, H
        batch = sz[0]
        # generate grid mesh
        x = self.indices_x.cuda() # for multi-gpus
        y = self.indices_y.cuda()
        indices_x = x.repeat([batch, 1, 1]) + polar[:, 0].unsqueeze(1).unsqueeze(1)
        indices_y = y.repeat([batch, 1, 1]) + polar[:, 1].unsqueeze(1).unsqueeze(1)
        indices = ops.concat((indices_x.unsqueeze(3)/(sz[2]//2), indices_y.unsqueeze(3)/(sz[3]//2)), 3)

        return indices

    def forward(self, x, polar):
        grid = self.get_logpolar_grid(polar, x.shape)
        # x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        x = F.grid_sample(x, grid, mode='bilinear', padding_mode='border')

        return x


class Polar_Pick(nn.Cell):
    """
    SiamFC head
    """
    def __init__(self):
        super(Polar_Pick, self).__init__()
        points = self.generate_points(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE)
        self.points = ms.Tensor.from_numpy(points)
        self.points_cuda = self.points



    def generate_points(self, stride, size):
        # print('stride',stride,'size',size)
        ori = - (size // 2) * stride # -96
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points


    def _getArgMax(self, r):
        sizes = r.shape
        batch = sizes[0]
        m = r.view(batch, -1).argmax(1).view(-1, 1)
        indices = ops.concat((m // sizes[2], m % sizes[2]), dim=1)
        indices = (indices - (sizes[2]-1)/2) / (sizes[2]-1)/2
        return indices

    def _getSoftArgMax(self, r):
        r = r.squeeze(1)
        sizes = r.shape
        assert len(sizes) == 3
        batch = sizes[0]
        sm = r.view(batch, -1).softmax(1).view(sizes)
        x_ls = torch.linspace(0, sizes[1] - 1, sizes[1])
        y_ls = torch.linspace(0, sizes[2] - 1, sizes[2])
        x, y = torch.meshgrid([x_ls, y_ls])
        indices_x = torch.mul(sm, x.unsqueeze(0).cuda()).sum([1, 2]) / (sizes[1] - 1)
        indices_y = torch.mul(sm, y.unsqueeze(0).cuda()).sum([1, 2]) / (sizes[2] - 1)
        indices = ops.concat((indices_x.view(-1, 1), indices_y.view(-1, 1)), 1)
        return indices

    def test_self_points(self):
        points = self.points
        points = ops.transpose(points (1,0))
        plt.scatter(points[0],points[1])

    #4 parameters loc
    def forward(self, cls, loc):
        # self.test_self_points()
        sizes = cls.shape
        batch = sizes[0]
        score = ops.transpose(cls.view(batch, cfg.BAN.KWARGS.cls_out_channels, -1), (0, 2, 1))
        best_idx = ops.Argmax(1)(score[:, :, 1])

        idx = best_idx.unsqueeze(1)
        idx = idx.unsqueeze(2)

        delta = ops.transpose(loc.view(batch, 4, -1), (0, 2, 1))
        # delta = loc.view(batch, 6, -1).permute(0, 2, 1)
        # delta = loc.view(batch, 2, -1).permute(0, 2, 1)

        dummy = idx.expand(batch, 1, delta.shape[2])
        point = self.points.cuda()
        point = point.expand(batch, point.shape[0], point.shape[1])

        delta = ops.gather_elements(delta, 1, dummy).squeeze(1)
        point = ops.gather_elements(point, 1, dummy[:,:,0:2]).squeeze(1)

        out = torch.zeros(batch, 2).cuda()
        out[:, 0] = (point[:, 0] - delta[:, 0] + point[:, 0] + delta[:, 2]) / 2
        out[:, 1] = (point[:, 1] - delta[:, 1] + point[:, 1] + delta[:, 3]) / 2
        return out

    #shorten the time.
    def get_polar_from_two_para_loc (self, cls, loc):
        sizes = cls.shape
        batch = sizes[0]
        score = ops.transpose(cls.view(batch, cfg.BAN.KWARGS.cls_out_channels, -1), (0, 2, 1))
        best_idx = ops.Argmax(1)(score[:, :, 1])

        idx = ops.expand_dims(best_idx, 1)
        idx = ops.expand_dims(idx, 2)
        delta = ops.transpose(loc.view(batch, 2, -1), (0, 2, 1))

        dummy = ops.broadcast_to(idx, (batch, 1, delta.shape[2]))
        point = self.points_cuda
        point = ops.broadcast_to(point, (batch, point.shape[0], point.shape[1]) )

        delta = ops.gather_elements(delta, 1, dummy).squeeze(1)
        point = ops.gather_elements(point, 1, dummy[:,:,0:2]).squeeze(1)

        out = ops.Zeros()((batch, 2), ms.float32)
        out[:, 0] = point[:, 0] - delta[:, 0]
        out[:, 1] = point[:, 1] - delta[:, 1]
        return out

if __name__ == '__main__':
    cfg.merge_from_file(args.cfg)

    img=cv2.imread("/home/lbyang/workspace/HDN_mindspore/testing_dataset/UCSB/bricks_dynamic_lighting/frame00001.jpg")

    cls = ms.Tensor(shape=(32, 2, 25, 25), dtype= ms.float32, init=One())
    loc = ms.Tensor(shape=(32, 2, 25, 25), dtype= ms.float32, init=One())

    context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(device_target='GPU')
    # Polar_Pick().get_polar_from_two_para_loc(cls, loc)
    scale, rot = lp_pick(cls, loc, cfg.BAN.KWARGS.cls_out_channels, cfg.POINT.STRIDE, cfg.POINT.STRIDE_LP,
                         cfg.TRAIN.OUTPUT_SIZE_LP, cfg.TRAIN.EXEMPLAR_SIZE)