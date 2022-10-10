#Copyright 2021, XinruiZhan
#Copyright 2022, LibangYang
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import math

from mindspore import ops
from hdn.core.config import cfg
from hdn.tracker.base_tracker import SiameseTracker
from hdn.utils.bbox import corner2center, cetner2poly, getRotMatrix, transformPoly,center2corner
from hdn.utils.point import Point, generate_points, generate_points_lp
from hdn.utils.transform import img_rot_around_center, img_rot_scale_around_center, img_shift, img_shift_crop_w_h, get_hamming_window
import cv2
import matplotlib.pyplot as plt
class hdnTracker(SiameseTracker):
    def __init__(self, model):
        super(hdnTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = generate_points(cfg.POINT.STRIDE, self.score_size)
        self.p = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.EXEMPLAR_SIZE // 2)
        self.points_lp = generate_points_lp(cfg.POINT.STRIDE_LP, cfg.POINT.STRIDE_LP, cfg.TRAIN.OUTPUT_SIZE_LP) #self.p.points.transpose((1, 2, 0)).reshape(-1, 2)
        self.model = model

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride # -96
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def generate_points_lp(self, stride_w, stride_h, size):
        # ori = - (size // 2) * stride  # -96
        ori_x = - (size // 2) * stride_w  # -96
        ori_y = - (size // 2) * stride_h  # -96
        x, y = np.meshgrid([ori_x + stride_w * dx for dx in np.arange(0, size)],
                           [ori_y + stride_h * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()
        return points

    def _convert_logpolar_simi(self, delta, point, peak_idx, idx=0):
        delta = ops.Transpose()(delta, (1, 2, 3, 0)).view(4, -1)
        delta = delta.asnumpy()
        # rotation
        delta[2, :] = point[:, 1] - delta[2, :] * cfg.POINT.STRIDE_LP
        delta[3, :] = point[:, 1] + delta[3, :] * cfg.POINT.STRIDE_LP

        delta[0, :] = point[:, 0] - delta[0, :] * cfg.POINT.STRIDE_LP
        delta[1, :] = point[:, 0] + delta[1, :] * cfg.POINT.STRIDE_LP
        scale = delta[0, :]
        rotation = delta[2, :]
        rotation = rotation * (2 * np.pi / cfg.TRAIN.EXEMPLAR_SIZE)
        mag = np.log(cfg.TRAIN.EXEMPLAR_SIZE / 2) / cfg.TRAIN.EXEMPLAR_SIZE
        delta[0, :] = np.exp(scale * mag)
        delta[1, :] = delta[0, :]
        delta[2, :] = rotation
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = ops.Transpose()(score, (1, 2, 3, 0)).view(-1)
            score = ops.Sigmoid()(score).asnumpy()
        else:
            score = ops.Transpose()(score, (1, 2, 3, 0)).view(self.cls_out_channels, -1)
            score = ops.Transpose()(score, (1, 0))
            score = ops.Softmax(1)(score)[:, 1].asnumpy()
        return score

    def get_window_scale_coef(self,region):
        region = region.reshape(8,-1)
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
             np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        return s

    def init(self, img, bbox, poly, first_point):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
            poly: (cx, cy, w, h, theta)
            first_point: (x1, y1) first point of gt
        """
        # self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
        #                             bbox[1]+(bbox[3]-1)/2])
        self.center_pos = np.array([poly[0],poly[1]])
        self.init_rot = poly[4]        # self.init_rot = 0
        self.rot = poly[4]            # self.rot = 0
        polygon = cetner2poly(poly[:4])
        tran = getRotMatrix(poly[0], poly[1], poly[4])
        polygon = transformPoly(polygon, tran)
        self.scale_coeff = self.get_window_scale_coef(polygon)
        fir_dis = (polygon - first_point) ** 2
        fir_dis = np.argmin(fir_dis[:,0] + fir_dis[:,1])
        self.poly_shift_l = fir_dis
        self.scale = 1
        self.lp_shift = [0,0]
        self.v = 0
        self.size = np.array([poly[2], poly[3]])
        self.align_size = np.array([bbox[2], bbox[3]])
        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        s_z = np.floor(s_z)
        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))
        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average, islog=1)
        self.model.template(z_crop)
        self.init_img = img
        self.init_crop_size = np.array([w_z, h_z])
        self.init_size = self.size
        self.init_s_z = s_z
        self.init_pos = np.array([poly[0],poly[1]])
        self.window_scale_factor = 1.0
        self.lost = True
        self.lost_count = 0
        self.last_lost = False






