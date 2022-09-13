import argparse

import mindspore as ms
from  transform import combine_affine_c0_v2, combine_affine_lt0
from hdn.core.config import cfg
from mindspore.common.initializer import Normal, initializer,One

parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
args = parser.parse_args()

if __name__ == '__main__':
    cfg.merge_from_file(args.cfg)
    polar = ms.Tensor(shape=(32,2), dtype=ms.float32, init=One())
    scale = ms.Tensor(shape=(32,), dtype=ms.float32, init=One())
    rot = ms.Tensor(shape=(32,), dtype=ms.float32, init=One())
    scale_h = True

    affine_m = combine_affine_c0_v2(cfg.TRACK.EXEMPLAR_SIZE / 2, cfg.TRACK.EXEMPLAR_SIZE / 2, polar, scale, rot, scale_h,
                         cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.EXEMPLAR_SIZE)
    affine_m_lt0 = combine_affine_lt0(cfg.TRACK.EXEMPLAR_SIZE / 2, cfg.TRACK.EXEMPLAR_SIZE / 2, polar, 1 / scale, -rot,
                                      cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.EXEMPLAR_SIZE)