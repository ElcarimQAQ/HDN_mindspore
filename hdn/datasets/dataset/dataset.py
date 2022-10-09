#Copyright 2022,Libing Yang
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from mindspore.dataset import  transforms

import orjson as json

import logging

import sys
import os
import math

import cv2
import numpy as np

from hdn.utils.bbox import center2corner, Center, corner2center, SimT
from hdn.datasets.point_target.point_target import PointTarget, PointTargetLP, PointTargetRot
from hdn.datasets.augmentation.homo_augmentation_e2e import Augmentation
from hdn.core.config import cfg
from hdn.models.logpolar import getPolarImg
import matplotlib.pyplot as plt
logger = logging.getLogger("global")
# matplotlib.use('agg')
# matplotlib.use('TkAgg')
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    # @profile
    def __init__(self, name, root, anno, frame_range, num_use, start_idx, if_unsup=False):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        print('cur_path',cur_path)
        self.name = name
        self.root = os.path.join(cur_path, '../../../', root)
        self.anno = os.path.join(cur_path, '../../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        self.if_unsup = if_unsup
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            # meta_data = json.load(f)
            meta_data = json.loads(f.read())
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int, filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]
        self.labels = meta_data
        self.num = len(meta_data)  #video_num
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox[0], dict):
                        if len(bbox[0]) == 4:
                            x1, y1, x2, y2 = bbox[0]
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox[0]
                        if w <= 1 or h <= 1: # 1 pixel too small to handle
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    #随机选择num_use帧
    def shuffle(self):
        print('self.start_idx',self.start_idx,'self.num',self.num, 'self.num_use:',self.num_use)#self.start_idx 0 self.num 24 self.num_use: 100000
        pickList = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(pickList)
            pick += pickList
        np.save("/data/HDN_var/subPickList.npy", np.array(pickList))
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num