from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import mindspore as ms
from mindspore import ops, Parameter,Tensor, nn


def get_cls_loss(pred, label, select):
    if len(select.shape) == 0 or select.shape == (0,):
        return 0
    pred = ops.gather(pred, select, 0)
    label = ops.cast(ops.gather(label, select, 0), ms.int32)
    return nn.NLLLoss()(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = ops.equal(label, 1).nonzero().squeeze()
    neg = ops.equal(label, 0).nonzero().squeeze()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def select_xr_focal_fuse_smooth_l1_loss_top_k(pred_cls, label_cls,delta_weight=0.1):
    """
    smooth_l1_loss, we only choose the top_K neg_loss as neg_loss cause too many neg points pull the loss down.
    :param pred_cls:
    :param label_cls:
    :param delta_weight:
    :return:
    """
    batch_size = label_cls.shape[0]
    label_cls = label_cls.reshape(-1)
    label_cls_new = label_cls.copy()
    pred_cls = pred_cls.view(-1,2)
    neg = ops.equal(label_cls, 0).nonzero().squeeze()
    pos = ops.gt(label_cls, 0).nonzero().squeeze()
    # cur_device = pred_cls.device
    zero_loss = Tensor(0.0)

    if len(pos.shape) == 0 or pos.shape == (0):
        pos_loss = zero_loss
    else:
        # pred_cls_pos = pred_cls[pos][:,1]
        pred_cls_pos = ops.gather(pred_cls, pos, 0)[:, 1]
        absolute_loss_pos = ops.abs(label_cls_new[pos] - pred_cls_pos)
        reg_loss_pos = absolute_loss_pos# use l1 loss
        pos_loss = reg_loss_pos.sum()/ (reg_loss_pos.shape[0]+1)

    if len(neg.shape) == 0 or neg.shape == (0, ):
        neg_loss = zero_loss  #problem here
    else:
        # pred_cls_neg = pred_cls[neg][:,1]
        pred_cls_neg = ops.gather(pred_cls, neg, 0)[:, 1]
        pred_cls_neg = ops.clip_by_value(pred_cls_neg, 0.000001, 0.9999999)
        reg_loss_neg = - ops.log(1 - pred_cls_neg)
        reg_loss_neg = ops.top_k(reg_loss_neg, batch_size*100, True)[0]  # TODO: topk暂时不能用
        neg_loss = reg_loss_neg.sum() / (reg_loss_neg.shape[0]+1)
    reg_loss = pos_loss + neg_loss
    return reg_loss


def select_xr_focal_fuse_smooth_l1_loss(pred_cls, label_cls,delta_weight=0.1):
    label_cls = label_cls.reshape(-1)
    label_cls_new = label_cls.copy()
    pred_cls = pred_cls.view(-1,2)
    neg = ops.equal(label_cls, 0).nonzero().squeeze()
    pos = ops.gt(label_cls, 0).nonzero().squeeze()
    zero_loss = Tensor(0.0)

    pos_loss = zero_loss
    neg_loss = zero_loss
    if len(pos.shape) == 0 or pos.shape == (0):
        pos_loss = zero_loss

    else:
        pred_cls_pos = ops.gather(pred_cls, pos, 0)[:, 1]
        absolute_loss_pos = ops.abs(label_cls_new[pos] - pred_cls_pos)
        square_loss_pos =  0.5 * ((label_cls_new[pos] - pred_cls_pos)) ** 2
        inds_pos = ops.le(absolute_loss_pos, 1)
        reg_loss_pos = ( inds_pos * square_loss_pos + (1 - inds_pos) * (absolute_loss_pos - 0.5))
        pos_loss = reg_loss_pos.sum()/ (reg_loss_pos.shape[0]+1)

    if len(neg.shape) == 0 or neg.shape == (0):
        neg_loss = zero_loss  #problem here
    else:
        pred_cls_neg = ops.gather(pred_cls, neg, 0)[:, 1]
        pred_cls_neg = ops.clip_by_value(pred_cls_neg, 0.000001, 0.9999999)
        absolute_loss_neg = ops.abs(label_cls_new[neg] - pred_cls_neg)
        reg_loss_neg = -0.5*absolute_loss_neg * ops.log(1 - pred_cls_neg)
        neg_loss = reg_loss_neg.sum() / (reg_loss_neg.shape[0]+1)
    reg_loss = pos_loss + neg_loss
    return reg_loss


def select_l1_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = ops.gt(label_cls, 0).nonzero().squeeze()

    pred_loc = pred_loc.transpose(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = ops.gather(pred_loc, pos, 0)

    label_loc = label_loc.transpose(0, 2, 3, 1).reshape(-1, 4)
    label_loc = ops.gather(label_loc, pos, 0)
    return kalyo_l1_loss(pred_loc, label_loc) #+ 0.5 * kalyo_l1_loss(pred_loc_add, label_loc_add)


def select_l1_loss_c(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    _, max_c = ops.ArgMaxWithValue()(label_cls)
    pos = ops.gt(label_cls, max_c - 0.2).nonzero().squeeze()
    zero_loss = Tensor(0.0)
    if len(pos.shape) == 0 or pos.shape == (0, ):
        loss_pos = zero_loss
        return loss_pos
    pred_loc = pred_loc.transpose(0, 2, 3, 1).reshape(-1, 2)
    pred_loc = ops.gather(pred_loc, pos, 0)
    label_loc = label_loc.transpose(0, 2, 3, 1).reshape(-1, 2)
    label_loc = ops.gather(label_loc, pos, 0)
    absolute_loss = ops.abs(pred_loc - label_loc)
    square_loss = 0.5 * (label_loc - pred_loc) * (label_loc - pred_loc)
    inds = ops.cast(ops.less(absolute_loss, 1), ms.float32)
    reg_loss = (inds * square_loss + (1 - inds) * (absolute_loss - 0.5))
    tsz = label_loc.shape[0] * label_loc.shape[1]+1
    reg_loss = reg_loss.sum()/tsz#weighted loss
    return reg_loss

#TODO:
def select_l1_loss_lp(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    label_cls_new = label_cls.clone()
    pos = label_cls_new.gt(0).nonzero().squeeze().cuda()
    pred_loc = pred_loc.transpose(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = ops.gather(pred_loc, pos, 0)
    label_loc = label_loc.transpose(0, 2, 3, 1).reshape(-1, 4)
    label_loc = ops.gather(label_loc, pos, 0)
    absolute_loss = ops.abs(pred_loc - label_loc)
    square_loss = 0.5 * ((label_loc - pred_loc)) ** 2
    inds = absolute_loss.lt(1).float()
    reg_loss = (inds * square_loss + (1 - inds) * (absolute_loss - 0.5))
    reg_loss = (reg_loss[:,1]).sum()/(pos.sum()) #weighted loss
    reg_loss = (reg_loss.sum())/(pos.sum()) #weighted loss
    tsz = label_loc.shape[0] * label_loc.shape[1]+1
    reg_loss = (reg_loss.sum())/tsz #weighted loss
    return reg_loss


def kalyo_l1_loss(output, target, norm=False):
    if len(output.shape) == 1:
        tsz = output.shape[0]+1
    else :
        tsz = output.shape[0] * output.shape[1]+1
    # w, h
    absolute_loss = ops.abs(target - output)
    square_loss = 0.5 * (target - output) * (target - output)
    if norm:
        absolute_loss = absolute_loss / (target[:, :2] + 1e-10)
        square_loss = square_loss / (target[:, :2] + 1e-10) * (target[:, :2] + 1e-10)
    inds = ops.less(absolute_loss, 1)
    reg_loss = (inds * square_loss + (1 - inds) * (absolute_loss - 0.5))
    return reg_loss.sum()/tsz

