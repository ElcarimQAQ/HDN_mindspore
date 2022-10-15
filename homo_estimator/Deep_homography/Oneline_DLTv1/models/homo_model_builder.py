from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from mindspore import ops, nn, Tensor
import torch.nn.functional as F
import imageio
from hdn.core.config import cfg
from homo_estimator.Deep_homography.Oneline_DLTv1.backbone import get_backbone
from homo_estimator.Deep_homography.Oneline_DLTv1.preprocess import get_pre
import mindspore as ms
from homo_estimator.Deep_homography.Oneline_DLTv1.utils import transform, DLT_solve
import matplotlib.pyplot as plt
from mindspore.scipy.linalg import inv
from hdn.models.triplet_loss import TripletMarginLoss as myTripletMarginLoss
from mindspore.nn.loss.loss import TripletMarginLoss

"""
The model_builder we use right now.
"""

criterion_l2 = nn.MSELoss()
# triplet_loss = nn.TripletMarginLoss(margin=1.0, p=1, reduce=False, size_average=False)#anchor p, n
myTriplet_loss = myTripletMarginLoss(margin=1.0, p=1)
triplet_loss = TripletMarginLoss(p=1)


def normMask(mask, strenth=0.5):
    """
    :return: to attention more region

    """
    batch_size, c_m, c_h, c_w = mask.shape
    max_value = mask.reshape(batch_size, -1).max(1)[0]
    # print('max_value.shape',max_value.shape)
    max_value = max_value.reshape(batch_size, 1, 1, 1)
    mask = mask / (max_value * strenth)
    mask = ops.clip_by_value(mask, 0, 1)

    return mask

class HomoModelBuilder(nn.Cell):
    def __init__(self, pretrained = False):
        super(HomoModelBuilder, self).__init__()

        # build head
        self.ShareFeature = get_pre('PreShareFeature')
        model_name = cfg.BACKBONE_HOMO.TYPE
        print('pretrained:',pretrained)
        self.backbone = get_backbone(model_name,
                                     pretrained, **cfg.BACKBONE_HOMO.KWARGS)
        self.avgpool = ops.AdaptiveAvgPool2D(1)
        print('self.avgpool',self.avgpool)

        if model_name == 'resnet18' or model_name == 'resnet34':
            self.fc = nn.Dense(512, 8)
        elif model_name == 'resnet50':
            self.fc = nn.Dense(2048, 8)


    def construct(self, data):
        org_imgs = data['org_imgs']
        input_tensors = data['input_tensors']
        h4p = data['h4p']
        patch_inds = data['patch_indices']
        # _device = 'cuda' if str(org_imgs.device)[:4] =='cuda' else 'cpu'
        # tmp_window = data['template_mask'] #[8,127,127]
        if 'search_windowx' in data: #acturally search_window
            sear_window = data['search_window'].squeeze(1) #[8,127,127]
        else:
            sear_window = ops.ones((input_tensors.shape[0], 127,127), ms.float32)
        if 'if_pos' in data:
            if_pos = data['if_pos']
        else:
            if_pos = ops.ones((input_tensors.shape[0],1, 127,127), ms.float32)
        if 'if_unsup' in data:
            if_unsup = data['if_unsup']
        else:
            if_unsup = ops.ones((input_tensors.shape[0],1, 127,127), ms.float32)
        batch_size, _, img_h, img_w = org_imgs.shape
        _, _, patch_size_h, patch_size_w = input_tensors.shape
        y_t = ms.numpy.arange(0, batch_size * img_w * img_h,img_w * img_h)
        # batch_inds_tensor = y_t.unsqueeze(1).expand(y_t.shape[0], patch_size_h * patch_size_w).reshape(-1)
        batch_indices_tensor = ops.reshape(ops.broadcast_to(ops.expand_dims(y_t, 1), (y_t.shape[0], patch_size_h * patch_size_w)), (-1,))
        w_h_scala = 63.5
        M_tensor = Tensor([[w_h_scala, 0., w_h_scala],
                                 [0., w_h_scala, w_h_scala],
                                 [0., 0., 1.]], ms.float32)

        M_tile = M_tensor.expand_dims(0).broadcast_to((batch_size, M_tensor.shape[-2], M_tensor.shape[-1]))
        # Inverse of M
        M_tensor_inv = inv(M_tensor)
        M_tile_inv = M_tensor_inv.expand_dims(0).broadcast_to((batch_size, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1]))

        #original feature
        patch_1 = self.ShareFeature(input_tensors[:, :1, ...])
        patch_2 = self.ShareFeature(input_tensors[:, 1:, ...])

        #feature normed
        patch_1_res = patch_1
        patch_2_res = patch_2

        x = ops.concat((patch_1_res, patch_2_res), 1)
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)#[bsz, 8]
        H_mat = DLT_solve(h4p, x).squeeze(1)  #H: search -> template
        # 'DLT_solve'
        pred_I2 = transform(patch_size_h, patch_size_w, M_tile_inv, H_mat, M_tile,
                            org_imgs[:, :1, ...], patch_inds, batch_indices_tensor)

        pred_I2_CnnFeature = self.ShareFeature(pred_I2)

        ## handle the negative samples loss
        neg_ids = ops.equal(if_pos, 0).nonzero().squeeze(1)
        #only unsupervised homo
        pos_ids = ops.equal(if_pos*if_unsup, 1).nonzero().squeeze(1)
        #add center mask
        mask_sear = ops.Cast()(ops.gt(sear_window, 0).expand_dims(1), ms.float32)
        #do not use mask at all
        patch_1_m = patch_1
        patch_2_m = patch_2
        pred_I2_CnnFeature_m = pred_I2_CnnFeature
        ## use neg samples, it seems loss doesn't descend
        if neg_ids.shape[0] != 0:
            if pos_ids.shape == (0,):
                tmp_pos = Tensor([])
                sear_pos = Tensor([])
                pred_pos = Tensor([])
            else:
                tmp_pos = patch_1_m[pos_ids]
                sear_pos = patch_2_m[pos_ids]
                pred_pos = pred_I2_CnnFeature_m[pos_ids]
            #only use the pos samples
            tmp_replace = tmp_pos
            sear_replace = sear_pos
            pred_replace = pred_pos
        else:
            mask_num = mask_sear.nonzero().shape[0]
            tmp_replace = patch_1_m
            sear_replace = patch_2_m
            pred_replace = pred_I2_CnnFeature_m
        margin = ms.Tensor(1.0)
        # feature_loss_mat = triplet_loss(sear_replace, pred_replace, tmp_replace, margin)
        feature_loss_mat = myTriplet_loss(sear_replace, pred_replace, tmp_replace)
        feature_loss = feature_loss_mat.sum() / pos_ids.shape[0] /(127*127)
        feature_loss = ops.expand_dims(feature_loss, 0)
        #neg loss
        # cur_device = feature_loss.device
        homo_neg_loss = Tensor(0.0)
        if neg_ids.shape[0] > 0:
            homo_neg_loss = ops.ReduceSum()(ops.norm(x[neg_ids,:], p=2, axis=1)) / neg_ids.shape[0]


        pred_I2_d = pred_I2[:1, ...]
        patch_2_res_d = patch_2_res[:1, ...]
        pred_I2_CnnFeature_d = pred_I2_CnnFeature[:1, ...]

        out_dict = {}

        out_dict.update(feature_loss=feature_loss, pred_I2_d=pred_I2_d, x=x, H_mat=H_mat, patch_2_res_d=patch_2_res_d,
                        pred_I2_CnnFeature_d=pred_I2_CnnFeature_d, homo_neg_loss=homo_neg_loss)

        return out_dict
