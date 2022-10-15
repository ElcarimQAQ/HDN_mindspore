#Copyright 2022, Libing Yang
# A distribute version of training
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import mindspore as ms
from mindspore import context, set_seed, nn, Tensor
import mindspore.dataset as ds
from hdn.utils.lr_scheduler import build_lr_scheduler
from hdn.utils.log_helper import init_log, print_speed, add_file_handler
from hdn.utils.distributed import dist_init, DistModule, reduce_gradients, \
    average_reduce, get_rank, get_world_size
from hdn.utils.model_load import load_pretrain, restore_from
from hdn.utils.average_meter import AverageMeter
from hdn.utils.misc import describe, commit
from hdn.models.model_builder_e2e_unconstrained_v2 import ModelBuilder
# from hdn.datasets.dataset.semi_supervised_dataset import BANDataset
from hdn.datasets.dataset import get_dataset

from hdn.core.config import cfg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
import time
logger = logging.getLogger('global')

parser = argparse.ArgumentParser(description='siamese tracking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--local_rank', type=int, default=0,
                    help='compulsory for pytorch launcer')
args = parser.parse_args()

CUDA_VISIBLE_DEVICES=0




def seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    set_seed(seed)
    np.random.seed(123456)

def make_mesh(patch_w,patch_h):
    x_flat = np.arange(0,patch_w)
    x_flat = x_flat[np.newaxis,:]
    y_one = np.ones(patch_h)
    y_one = y_one[:,np.newaxis]
    x_mesh = np.matmul(y_one , x_flat)

    y_flat = np.arange(0,patch_h)
    y_flat = y_flat[:,np.newaxis]
    x_one = np.ones(patch_w)
    x_one = x_one[np.newaxis,:]
    y_mesh = np.matmul(y_flat,x_one)
    return x_mesh,y_mesh

def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    if cfg.BAN.BAN:
        print('cfg.DATASET.TYPE',cfg.DATASET.TYPE)
        train_dataset =get_dataset(cfg.DATASET.TYPE)
    logger.info("build dataset done")

    train_sampler = None
    print('num_worker',cfg.TRAIN.NUM_WORKERS)
    #we don't have enough memory
    train_dataset = ds.GeneratorDataset(train_dataset, column_names = [
                    'template',
                    "template_lp",
                    'search',
                    'template_poly',
                    'search_poly',
                    'label_cls',
                    'label_loc',
                    'label_cls_lp',
                    'label_loc_lp',
                    'scale_dist',
                    'label_cls_c',
                    'label_loc_c',
                    'window_map',
                    'template_hm',
                    'search_hm',
                    'template_window',
                    'search_window',
                    'if_pos',
                    'temp_cx',
                    'temp_cy',
                    'if_unsup'], num_parallel_workers = cfg.TRAIN.NUM_WORKERS, sampler= train_sampler)
    train_dataset = train_dataset.batch(cfg.TRAIN.BATCH_SIZE)
    return train_dataset

def build_opt_lr(model, current_epoch=0):
    model.set_train(True)
    for param in model.backbone.trainable_params():
        param.requires_grad = False
    for _, m in model.backbone.cells_and_names():
        if isinstance(m, nn.BatchNorm2d):
            m.set_train(False)
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:
        for layer in cfg.BACKBONE.TRAIN_LAYERS:
            for param in getattr(model.backbone, layer).get_parameters():
                if 'beta'in param.name or 'weight' in param.name or 'gamma'in param.name:
                    param.requires_grad = True
            for _, m in getattr(model.backbone, layer).cells_and_names():
                if isinstance(m, nn.BatchNorm2d):
                    m.set_train(True)

    trainable_params = []
    trainable_params += [{'params': model.backbone.trainable_params(), #Todo:没有
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    if cfg.ADJUST.ADJUST:
        if cfg.TRAIN.OBJ == 'LP':
            for param in model.neck.get_parameters():
                param.requires_grad = False
            trainable_params += [{'params': model.neck_lp.trainable_params(),
                                  'lr': cfg.TRAIN.BASE_LR}]
        elif cfg.TRAIN.OBJ == 'NM':
            for param in model.neck_lp.get_parameters():
                param.requires_grad = False
            trainable_params += [{'params': model.neck.trainable_params(),
                                  'lr': cfg.TRAIN.BASE_LR}]
        elif cfg.TRAIN.OBJ == 'SIM' or cfg.TRAIN.OBJ == 'ALL':
            trainable_params += [{'params': model.neck_lp.trainable_params(),
                                  'lr': cfg.TRAIN.BASE_LR}]
            trainable_params += [{'params': model.neck.trainable_params(),
                                  'lr': cfg.TRAIN.BASE_LR}]
        elif cfg.TRAIN.OBJ == 'HOMO':
            for param in model.neck.get_parameters():
                param.requires_grad = False
            for param in model.neck_lp.get_parameters():
                param.requires_grad = False
    # neck & head
    if cfg.TRAIN.OBJ == 'LP':
        trainable_params += [{'params': model.head_lp.trainable_params(),
                              'lr': cfg.TRAIN.BASE_LR}]
        for param in model.head.get_parameters():
            param.requires_grad = False
    elif cfg.TRAIN.OBJ == 'NM':
        trainable_params += [{'params': model.head.trainable_params(),
                              'lr': cfg.TRAIN.BASE_LR}]
        for param in model.head_lp.get_parameters():
            param.requires_grad = False
    elif cfg.TRAIN.OBJ == 'SIM':
        trainable_params += [{'params': model.head_lp.trainable_params(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.head.trainable_params(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.hm_net.trainable_params(),
                              'lr': cfg.TRAIN.HOMO_START_LR}]
    elif cfg.TRAIN.OBJ == 'ALL':
        trainable_params += [{'params': model.head_lp.trainable_params(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.head.trainable_params(),
                              'lr': cfg.TRAIN.BASE_LR}]
        trainable_params += [{'params': model.hm_net.trainable_params(),
                              'lr': cfg.TRAIN.HOMO_START_LR * cfg.TRAIN.HOMO_LR_RATIO}]

    lr_scheduler = nn.ExponentialDecayLR(cfg.TRAIN.BASE_LR, decay_rate=0.8, decay_steps= cfg.TRAIN.EPOCH)
    for i in range(cfg.TRAIN.EPOCH):
        step = ms.Tensor(i, ms.int32)
        result = lr_scheduler(step)
        print(f"step{i + 1}, lr:{result}")

    optimizer = nn.Adam(params=trainable_params, learning_rate=lr_scheduler, use_amsgrad=True)
    return optimizer, lr_scheduler

def train_ms(train_dataset, model, optimizer, lr_scheduler):
    average_meter = AverageMeter()

    def is_valid_number(x):
        return not (math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = 1
    # train_dataset.get_dataset_size()
    num_per_epoch = train_dataset.children[0].source_len // cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)

    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch
    print('start epoch', cfg.TRAIN.START_EPOCH)
    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR):
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)

    end = time.time()
    print('num_per_epoch', num_per_epoch)

    idx = 0
    for data in train_dataset.create_dict_iterator():
        # with open('/data/HDN_var/trainDictData.json') as fp:
        #     data = json.load(fp)
        # for key in data:
        #     data[key] = Tensor(data[key])

        if epoch != idx // num_per_epoch + start_epoch:  # 更新epoch
            epoch = idx // num_per_epoch + start_epoch

            ms.save_checkpoint(
                {'epoch': epoch,
                 'state_dict': model.parameters_dict(),
                 'optimizer': optimizer.state_dict()},  # 估计有问题
                cfg.TRAIN.SNAPSHOT_DIR + '/got_e2e_%s_e%d.pth' % (cfg.TRAIN.OBJ, epoch))

            if epoch == cfg.TRAIN.EPOCH:
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))
            # lr_scheduler(epoch)
            logger.info('epoch: {}'.format(epoch + 1))

        # if idx % num_per_epoch == 0 and idx != 0:
        #     for idx, pg in enumerate(optimizer.param_groups):
        #         logger.info('epoch {} lr {}'.format(epoch + 1, pg['lr']))
        
       
        data_time = average_reduce(time.time() - end)
        outputs = model(data)

        loss = outputs['total_loss']
        # if is_valid_number(loss):
            # loss.backward()
            # reduce_gradients(model)
            # clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)
            # optimizer.step()
        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)

        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.data.item())
        average_meter.update(**batch_info)

        end = time.time()
        idx = idx + 1


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # load cfg
    cfg.merge_from_file(args.cfg)
    if not os.path.exists(cfg.TRAIN.LOG_DIR):
        os.makedirs(cfg.TRAIN.LOG_DIR)
    init_log('global', logging.INFO)
    if cfg.TRAIN.LOG_DIR:
        add_file_handler('global',os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),logging.INFO)

    logger.info("Version Information: \n{}\n".format(commit()))
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    # create model
    model = ModelBuilder()

    # load pretrained backbone weights
    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        print('pretrained path', backbone_path)
        load_pretrain(model.backbone, backbone_path)

    # build dataset loader
    train_dataset = build_data_loader()

    start_epoch = cfg.TRAIN.START_EPOCH
    print('start_epoch',start_epoch)

    #build optimizer and lr_scheduler
    optimizer, lr_scheduler = build_opt_lr(model, cfg.TRAIN.START_EPOCH)
    # resume training
    RESUME_PATH = cfg.BASE.PROJ_PATH + cfg.TRAIN.RESUME
    if cfg.TRAIN.RESUME:
        logger.info("resume from {}".format(RESUME_PATH))
        assert os.path.isfile(RESUME_PATH), \
            '{} is not a valid file.'.format(RESUME_PATH)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, RESUME_PATH)
    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        print('if cfg.TRAIN.PRETRAINED')
        load_pretrain(model, cfg.TRAIN.PRETRAINED)

    logger.info(lr_scheduler)
    logger.info("model prepare done")
    cfg.TRAIN.START_EPOCH = start_epoch

    # start training
    train_net = nn.TrainOneStepCell(model, optimizer)
    # train_net.set_train(False)
    average_meter = AverageMeter()

    step = 0
    end = time.time()
    for epoch in range(cfg.TRAIN.EPOCH):
        for data in train_dataset.create_dict_iterator():
            # with open('/data/HDN_var/trainDictData.json') as fp:
            #     data = json.load(fp)
            # for key in data:
            #     data[key] = Tensor(data[key])
            loss = train_net(data)
            batch_time = time.time() - end
            batch_info = {}
            batch_info['batch_time'] = average_reduce(batch_time)
            # batch_info['data_time'] = average_reduce(data_time)

            # for k, v in sorted(outputs.items()):
            #     batch_info[k] = average_reduce(v.data.item())
            average_meter.update(**batch_info)

            end = time.time()
            print(loss)
            step = step + 1
    # train_ms(train_dataset, model, optimizer, lr_scheduler)


if __name__ == '__main__':
    seed(args.seed)
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', max_device_memory='3.5GB')
    main()
