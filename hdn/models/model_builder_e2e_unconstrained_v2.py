'''
Designed for end-to-end homo-estimation.
unconstrained means we whether dataset give us label we can train the model.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import mindspore as ms
from mindspore import ops, nn, Tensor, numpy
import numpy as np
from hdn.core.config import cfg
from hdn.models.loss import select_cross_entropy_loss, \
    select_l1_loss, select_l1_loss_c, \
    select_l1_loss_lp, \
    select_xr_focal_fuse_smooth_l1_loss_top_k, kalyo_l1_loss
from hdn.models.loss import  select_xr_focal_fuse_smooth_l1_loss
from hdn.models.backbone import get_backbone
from hdn.models.head import get_ban_head
from hdn.models.neck import get_neck
from hdn.models.logpolar import STN_Polar, getPolarImg, Polar_Pick
import matplotlib.pyplot as plt
from hdn.utils.point import Point
from homo_estimator.Deep_homography.Oneline_DLTv1.models.homo_model_builder import HomoModelBuilder, normMask
from hdn.utils.transform import combine_affine_c0, combine_affine_lt0, combine_affine_c0_v2
from hdn.utils.point import generate_points, generate_points_lp, lp_pick, get_center
from homo_estimator.Deep_homography.Oneline_DLTv1.utils import DLT_solve
from homo_estimator.Deep_homography.Oneline_DLTv1.utils import transform as Homo_STN
from mindspore.scipy.linalg import inv
from mindspore.common.initializer import One
from hdn.models.triplet_loss import TripletMarginLoss as myTripletMarginLoss
from mindspore.nn.loss.loss import TripletMarginLoss

criterion_l2 = nn.MSELoss()

#todo: bug in mindspore 's triplet_loss
myTriplet_loss = myTripletMarginLoss(margin=1.0, p=1)
triplet_loss = TripletMarginLoss(p=1, reduction='none')

class ModelBuilder(nn.Cell):
    # @profile
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,**cfg.BACKBONE.KWARGS)

        self.logpolar_instance = STN_Polar(cfg.TRACK.INSTANCE_SIZE)
        self.getPolar = Polar_Pick()
        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)
            self.neck_lp = get_neck(cfg.ADJUST.TYPE,
                                    **cfg.ADJUST.KWARGS, cut=False)

        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                          cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE  # Todo
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.points = generate_points(cfg.POINT.STRIDE, self.score_size)
        self.p = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.EXEMPLAR_SIZE // 2)
        self.points_lp = generate_points_lp(cfg.POINT.STRIDE_LP, cfg.POINT.STRIDE_LP,
                                            cfg.TRAIN.OUTPUT_SIZE_LP)  # self.p.points.transpose((1, 2, 0)).reshape(-1, 2)

        # build ban head
        # print('cfg',cfg)
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)
            self.head_lp = get_ban_head('MultiCircBAN', **cfg.BAN.KWARGS)
        self.hm_net = HomoModelBuilder(pretrained=True) #TODO

    def _convert_score(self, score):
        score = score.view(score.shape[0], self.cls_out_channels, -1).transpose(0, 2, 1)  # [28, 625, 2] or [28, 169, 2]
        score = score[:,:,1]
        return score

    def _convert_c(self, delta, point):

        delta = delta.view(delta.shape[0], 2, -1) #(28, 2, 625)
        point = Tensor.from_numpy(point) #(625, 2)
        delta[:, 0, :] = point[:, 0] - delta[:, 0, :]*8
        delta[:, 1, :] = point[:, 1] - delta[:, 1, :]*8
        return delta


    def feature_extractor(self, x):
        xf = self.backbone(x)
        # show feature
        # plt.imshow(xf[-1].asnumpy()[0, 3, :, :])
        # plt.show()_convert_logpolar_simi
        return xf

    def template(self, z):
        # z shape(1, 3, 127, 127)
        z_lp = z[:, 3:6, :, :]
        z = z[:, 0:3, :, :]
        zf = self.feature_extractor(z)
        zf_lp = self.feature_extractor(z_lp)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            zf_lp = self.neck_lp(zf_lp) # please think about this part, not sure whether is proper.
        self.zf = zf
        self.zf_lp = zf_lp


    def update_template(self, z, rot):
        zf = self.feature_extractor(z)
        polar = ops.expand_dims(ops.Zeros()((2), ms.float32 ), 0) #assuming the target has been moved to the center in the img.
        z_lp,_ = self.logpolar_instance(z, polar, rot)
        zf_lp = self.feature_extractor(z_lp)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            zf_lp = self.neck_lp(zf_lp)
        self.zf = zf
        self.zf_lp = zf_lp

    # TODO
    def track_new(self, x, delta=[0,0]):
        # x shape [1, 3, 255, 255]
        xf = self.feature_extractor(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc_c = self.head(self.zf, xf)
        return {
            'cls': cls,
            'loc_c': loc_c,
        }


    def track_new_lp(self, x, delta=[0,0]):
        # x shape [1, 3, 255, 255]
        polar = ops.ExpandDims()(ops.Zeros()(2, ms.float32), 0)
        x_lp, grid = self.logpolar_instance(x, polar, delta)
        # x_lp_np = np.load('/home/lbyang/data/x_lp_np.npy')
        # grid_np = np.load('/home/lbyang/data/grid_np.npy')
        # x_lp = ms.Tensor(x_lp_np, ms.float32)
        # grid = ms.Tensor(grid_np, ms.float32)
        xf_lp = self.feature_extractor(x_lp)
        if cfg.ADJUST.ADJUST:
            xf_lp = self.neck_lp(xf_lp)
        cls_lp, loc_lp = self.head_lp(self.zf_lp, xf_lp)
        # cls_lp = ms.Tensor(np.load('/home/lbyang/data/cls_lp.npy'))
        # loc_lp = ms.Tensor(np.load('/home/lbyang/data/loc_lp.npy'))
        return {
            'x_lp': x_lp,
            'cls_lp': cls_lp,
            'loc_lp': loc_lp,
            'grid': grid,
        }


    def track_proj(self, data, tmp_mask):

        org_imgs = data['org_imgs']
        input_tensors = data['input_tensors']
        h4p = data['h4p']
        patch_inds = data['patch_indices']
        batch_size, _, img_h, img_w = org_imgs.shape
        _, _, patch_size_h, patch_size_w = input_tensors.shape
        y_t = ms.numpy.arange(0, batch_size * img_w * img_h, img_w * img_h)
        batch_inds_tensor = ops.broadcast_to(ops.ExpandDims()(y_t, 1), (y_t.shape[0], patch_size_h * patch_size_w)).reshape(-1)
        M_tensor = ms.Tensor([[img_w / 2.0, 0., img_w / 2.0],
                                 [0., img_h / 2.0, img_h / 2.0],
                                 [0., 0., 1.]])

        # Inverse of M
        #fixme
        patch_1 = self.hm_net.ShareFeature(input_tensors[:, :1, ...])
        patch_2 = self.hm_net.ShareFeature(input_tensors[:, 1:, ...])

        #without mask
        patch_1_res = patch_1
        patch_2_res = patch_2

        x = ops.concat((patch_1_res, patch_2_res), 1)
        x = self.hm_net.backbone(x) # TODO: 会有一点差别，误差不大
        # x = ms.Tensor(np.load('/home/lbyang/data/x.npy'))
        #fixme
        x = self.hm_net.avgpool(x)
        x = x.view(x.shape[0], -1)
        #fixme
        x = self.hm_net.fc(x)
        H_mat = DLT_solve(h4p, x).squeeze(1)  #
        w_h_scala = 63.5
        M_tensor = ms.Tensor([[w_h_scala, 0., w_h_scala], [0., w_h_scala, w_h_scala], [0., 0., 1.]])


        M_tile = ops.broadcast_to(ops.expand_dims(M_tensor, 0), (batch_size, M_tensor.shape[-2], M_tensor.shape[-1]))
        # Inverse of M
        M_tensor_inv = inv(M_tensor, overwrite_a=True, check_finite=False)
        M_tile_inv = ops.broadcast_to(ops.expand_dims(M_tensor_inv, 0), (batch_size, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1]))
        pred_I2 = Homo_STN(patch_size_h, patch_size_w, M_tile_inv, H_mat, M_tile,
                           org_imgs[:, :1, ...], patch_inds, batch_inds_tensor)
        pred_I2_CnnFeature = self.hm_net.ShareFeature(pred_I2)
        delta_mat = (patch_2 - pred_I2_CnnFeature).abs()[0][0]
        similarity_norm = ops.ReduceSum()(delta_mat) / (127*127) # wo mask
        delta_sim_mat = ops.abs(patch_2 - patch_1)[0][0]
        similarity_norm_simi = ops.ReduceSum()(delta_sim_mat) / (127*127) # wo mask
        return H_mat, similarity_norm, similarity_norm_simi


    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.transpose(0, 2, 3, 1)
            cls = ops.log_softmax(cls,3)
        return cls

    def softmax(self, cls):
        if cfg.BAN.BAN:
            #cls [28, 2, 25, 25]
            cls = cls.transpose(0, 2, 3, 1)
            cls = ops.Softmax(axis=3)(cls)
        return cls

    def stn_with_theta(self, x, theta, size, polar):
        theta = theta.view(-1, 2, 3)
        grid = ops.affine_grid(theta, size)
        x = ops.grid_sample(x, grid)
        return x

    def make_mesh(self, patch_w, patch_h):
        x_flat = np.arange(0, patch_w)
        x_flat = x_flat[np.newaxis, :]
        y_one = np.ones(patch_h)
        y_one = y_one[:, np.newaxis]
        x_mesh = np.matmul(y_one, x_flat)

        y_flat = np.arange(0, patch_h)
        y_flat = y_flat[:, np.newaxis]
        x_one = np.ones(patch_w)
        x_one = x_one[np.newaxis, :]
        y_mesh = np.matmul(y_flat, x_one)
        return x_mesh, y_mesh

    def getPatchFromFullimg(self, patch_size_h, patch_size_w, patchIndices, batch_indices_tensor, img_full):
        num_batch, num_channels, height, width = img_full.size()
        warped_images_flat = img_full.reshape(-1)
        patch_indices_flat = patchIndices.reshape(-1)
        pixel_indices = patch_indices_flat.long() + batch_indices_tensor
        mask_patch = ops.gather_elements(warped_images_flat, 0, pixel_indices)
        mask_patch = mask_patch.reshape([num_batch, 1, patch_size_h, patch_size_w])
        return mask_patch


    def normMask(self, mask, strenth=0.5):
        """
        :return: to attention more region
        """
        batch_size, c_m, c_h, c_w = mask.size()
        max_value = mask.reshape(batch_size, -1).max(1)[0]
        max_value = max_value.reshape(batch_size, 1, 1, 1)
        mask = mask / (max_value * strenth)
        mask = ops.clip_by_value(mask, 0, 1)
        return mask
    # @profile
    def get_homo_data(self,warped_template, warped_search, if_pos, if_unsup, template_window=None, search_window=None):
        _WIDTH, _HEIGHT = 127, 127
        _patch_w, _patch_h = 127, 127
        _rho = 0
        _x_mesh, _y_mesh = self.make_mesh(_patch_w, _patch_h)
        template_mean = ops.ReduceMean(True)(warped_template, 1)
        search_mean = ops.ReduceMean(True)(warped_search, 1)
        org_imges = ops.concat((template_mean, search_mean), 1)
        # input_tesnors = org_imges.clone()
        h, w = org_imges.shape[2], org_imges.shape[3]
        batch_size = org_imges.shape[0]
        x, y = 0, 0
        y_t_flat = np.reshape(_y_mesh, [-1])
        x_t_flat = np.reshape(_x_mesh, [-1])
        patch_indices = ms.numpy.tile(ops.expand_dims(Tensor.from_numpy((y_t_flat + y) * w + (x_t_flat + x)), 0), (batch_size, 1))
        top_left_point = (x, y)
        bottom_left_point = (x, y + _patch_h)
        bottom_right_point = (_patch_w + x, _patch_h + y)
        top_right_point = (x + _patch_w, y)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]
        four_points = np.reshape(four_points, (-1))
        h4p = ops.Cast()(ms.numpy.tile(ops.expand_dims(Tensor.from_numpy(four_points), 0), (batch_size, 1)), ms.float32)
        input_tensors = org_imges[:, :, y: y + _patch_h, x: x + _patch_w]
        I = org_imges[:, 0, ...]
        I = I[:, np.newaxis, ...]
        I2_ori_img = org_imges[:, 1, ...]
        I2_ori_img = I2_ori_img[:, np.newaxis, ...]
        I1 = input_tensors[:, 0, ...]
        I1 = I1[:, np.newaxis, ...]
        I2 = input_tensors[:, 1, ...]
        I2 = I2[:, np.newaxis, ...]

        data = {}
        data['org_imgs'] = org_imges
        data['input_tensors'] = input_tensors
        data['h4p'] = h4p
        data['patch_indices'] = patch_indices
        data['template_window'] = template_window
        data['search_window'] = search_window
        data['if_pos'] = if_pos
        data['if_unsup'] = if_unsup
        return data


    def get_simi_data(self,template, search, warped_search):
        _WIDTH, _HEIGHT = 127, 127
        _patch_w, _patch_h = 127, 127
        _rho = 0
        _x_mesh, _y_mesh = self.make_mesh(_patch_w, _patch_h)
        template_mean = ops.ReduceMean(True)(template, 1)
        search_mean = ops.ReduceMean(True)(search, 1)
        warped_search_mean = ops.ReduceMean(True)(warped_search, 1)
        return template_mean, search_mean, warped_search_mean

    def construct(self, data):
        """ only used in training
        """
        ##get data from dataset
        template = data['template']
        template_lp = data['template_lp']
        search = data['search']
        label_cls = data['label_cls']
        label_loc_c = data['label_loc_c']
        label_cls_lp = data['label_cls_lp']
        label_loc_lp = data['label_loc_lp']
        template_poly = data['template_poly']
        search_poly = data['search_poly']
        search_hm = data['search_hm']
        template_hm = data['template_hm']
        template_window = ops.cast(data['template_window'], ms.float32)
        search_window = ops.cast(data['search_window'], ms.float32)
        if_pos =  ops.cast(data['if_pos'], ms.float32)
        if_neg = ops.equal(if_pos, 0)
        if_unsup = ops.cast(data['if_unsup'], ms.float32)
        if_sup = ops.equal(if_unsup, 0)
        temp_cx = ops.cast(data['temp_cx'], ms.float32)
        temp_cy = ops.cast(data['temp_cy'], ms.float32)
        batch_sz = cfg.TRAIN.BATCH_SIZE

        # get feature
        zf = self.feature_extractor(template)
        zf_lp = self.feature_extractor(template_lp)
        xf = self.feature_extractor(search)

        ##cut feature
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        ##normal head
        cls, loc = self.head(zf, xf)

        ## log-polar branch
        #fixme e2e or separate
        # polar = self.getPolar(tmp, label_loc)#sample according to the label(original method)
        polar = self.getPolar.get_polar_from_two_para_loc(cls, loc*cfg.POINT.STRIDE)
        # # polar = self.getPolar(cls, loc*cfg.POINT.STRIDE) #sample according to the cls_map. we div the stride before.
        x_lp, _ = self.logpolar_instance(search, polar)#[8, 3, 127, 127]
        xf_lp = self.feature_extractor(x_lp)

        #neck_lp
        if cfg.ADJUST.ADJUST:
            zf_lp = self.neck_lp(zf_lp)
            xf_lp = self.neck_lp(xf_lp)

        #head_lp
        cls_lp, loc_lp = self.head_lp(zf_lp, xf_lp)#cls_lp :[batch_sz,2,13,13],loc_lp:[batch_sz,4,13,13]

        scale, rot = lp_pick(cls_lp, loc_lp, cfg.BAN.KWARGS.cls_out_channels, cfg.POINT.STRIDE, cfg.POINT.STRIDE_LP, cfg.TRAIN.OUTPUT_SIZE_LP, cfg.TRAIN.EXEMPLAR_SIZE)

        ##homo-estimator
        if cfg.TRAIN.OBJ == 'ALL' or cfg.TRAIN.OBJ == 'HOMO':

            scale_h = True
            scale_ones = ops.ones((batch_sz), ms.float32)
            rot_zero = ops.Zeros()((batch_sz), ms.float32)
            polar_zero = ops.Zeros()((batch_sz, 2), ms.float32)

            #FIXME use SIM-estimator
            affine_m = combine_affine_c0_v2(cfg.TRACK.EXEMPLAR_SIZE/2, cfg.TRACK.EXEMPLAR_SIZE/2, polar, scale, rot, scale_h, cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.EXEMPLAR_SIZE)#warp the search image to 127*127 (self, nm_shift, scale, rot, scale_h, in_sz, out_sz):
            affine_m_lt0 = combine_affine_lt0(cfg.TRACK.EXEMPLAR_SIZE/2, cfg.TRACK.EXEMPLAR_SIZE/2, polar, 1/scale, -rot, cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.EXEMPLAR_SIZE)# warp the poly in search to poly in 127*127
            affine_m_c_aug = combine_affine_c0_v2(temp_cx, temp_cy, polar_zero, scale_ones, rot_zero, scale_h, cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.EXEMPLAR_SIZE)# warp the search accroding to the ,considering the template location
            affine_m_lt0_aug = combine_affine_lt0(temp_cx, temp_cy, polar_zero, scale_ones, rot_zero, cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.EXEMPLAR_SIZE)
            grid_size = (cfg.TRAIN.BATCH_SIZE, 1, cfg.TRAIN.EXEMPLAR_SIZE, cfg.TRAIN.EXEMPLAR_SIZE)

            warped_search_ori = self.stn_with_theta(search_hm, affine_m, grid_size, polar)#directly warped the search_hm
            search_hm_simi_ori = self.stn_with_theta(search_hm, affine_m_c_aug, grid_size, polar) #to 127*127
            poly_ones_tmp_var = ms.Tensor(shape=(cfg.TRAIN.BATCH_SIZE,1,4), dtype=ms.float32, init=One())
            affine_m_ones_tmp_var = ms.numpy.tile(ms.Tensor([0,0,1], dtype=ms.float32), (cfg.TRAIN.BATCH_SIZE, 1, 1))
            search_poly_hmg = ops.concat((search_poly.transpose(0,2,1), poly_ones_tmp_var),1)
            search_lt0_hm = ops.concat((affine_m_lt0, affine_m_ones_tmp_var),1)
            search_aug_lt0_hm = ops.concat((affine_m_lt0_aug, affine_m_ones_tmp_var),1)#only resize the search poly
            aug_sear_points = ops.BatchMatMul()(search_aug_lt0_hm, search_poly_hmg)
            pred_points = ops.BatchMatMul()(search_lt0_hm, search_poly_hmg)#
            template_hm_simi, search_hm_simi, wapred_search_simi = self.get_simi_data(template_hm, search_hm_simi_ori, warped_search_ori)
            search_window = ops.expand_dims(search_window, 1)
            search_window = self.stn_with_theta(search_window, affine_m, grid_size, polar)

            #ShareFeature
            tmp_f = self.hm_net.ShareFeature(template_hm_simi)
            sear_f = self.hm_net.ShareFeature(search_hm_simi)

            #without mask
            masked_tmp_f = tmp_f
            masked_sear_f = sear_f
            pre_sear_f = self.hm_net.ShareFeature(wapred_search_simi)

            #negative sample
            unsup_pos_ids = ops.equal(if_pos*if_unsup, 1).nonzero().squeeze(1)
            unsup_ids = ops.equal(if_unsup, 1).nonzero().squeeze(1)

            if unsup_pos_ids.shape[0] != 0:
                masked_tmp_f = masked_tmp_f[unsup_pos_ids]
                masked_sear_f = masked_sear_f[unsup_pos_ids]
                pre_sear_f = pre_sear_f[unsup_pos_ids]
                margin = ms.Tensor(1.0)

                # sim_loss_mat = triplet_loss(masked_tmp_f, pre_sear_f, masked_sear_f, margin)# cause we warp back the search
                sim_loss_mat = myTriplet_loss(masked_tmp_f, pre_sear_f, masked_sear_f)
                #fixme
                sim_loss = sim_loss_mat.sum() / (127*127) / unsup_pos_ids.shape[0]
                corner_pts_simi = pred_points.transpose(0,2,1)[:, :, :-1] #
                corner_pts_simi_pos = corner_pts_simi[unsup_pos_ids]
                template_poly_simi_pos = template_poly[unsup_pos_ids]
                corner_error_simi = ops.ReduceSum()(ops.abs(corner_pts_simi_pos - template_poly_simi_pos)) / unsup_pos_ids.shape[0] / 4

            #todo normalization
            if cfg.TRAIN.MODEL_TYPE == 'E2E':
                warped_search = warped_search_ori
                warped_template = template_hm
            homo_data = self.get_homo_data(warped_template, warped_search, if_pos, if_unsup, template_window, search_window)
            #fixme use entire homo-net
            batch_out = self.hm_net(homo_data)
            H_mat = batch_out['H_mat'] #search-> template
            if len(unsup_ids) > 0:
                loss_feature = batch_out['feature_loss'].mean()

        sup_ids = ops.equal(if_unsup, 0).nonzero().squeeze(1) #so far the neg_ids are designed for simi-estimator,
        if len(sup_ids) > 0 :
            cls_sup = cls[sup_ids]
            label_cls_sup = label_cls[sup_ids]
            label_cls_lp_sup = label_cls_lp[sup_ids]
            label_loc_c_sup = label_loc_c[sup_ids]
            label_loc_lp_sup = label_loc_lp[sup_ids]
            cls_lp_sup = cls_lp[sup_ids]
            loc_sup = loc[sup_ids]
            loc_lp_sup = loc_lp[sup_ids]
            cls_sup = self.softmax(cls_sup)#[n, 25, 25, 2]
            cls_loss_sup = select_xr_focal_fuse_smooth_l1_loss_top_k(cls_sup, label_cls_sup)
            loc_loss_sup = select_l1_loss_c(loc_sup, label_loc_c_sup, label_cls_sup)

            if cfg.TRAIN.WEIGHTED_MAP_LP:
                cls_lp_sup = self.softmax(cls_lp_sup)
                cls_loss_lp_sup = select_xr_focal_fuse_smooth_l1_loss(cls_lp_sup, label_cls_lp_sup)
                loc_loss_lp_sup = select_l1_loss_lp(loc_lp_sup, label_loc_lp_sup, label_cls_lp_sup)
            else:
                cls_lp_sup = self.log_softmax(cls_lp_sup)
                cls_loss_lp_sup = select_cross_entropy_loss(cls_lp_sup, label_cls_lp_sup)
                loc_loss_lp_sup = select_l1_loss(loc_lp_sup, label_loc_lp_sup, label_cls_lp_sup)


        sup_pos_ids = ops.equal(if_sup*if_pos, 1).nonzero().squeeze(1)
        sup_neg_ids = ops.equal(if_sup*ops.cast(if_neg, ms.float32), 1).nonzero().squeeze(1)
        if len(sup_pos_ids) > 0 :
            ##corner error for supervised data(the poly)
            corner_pts = ops.BatchMatMul()(H_mat, pred_points).transpose(0,2,1)
            corner_pts = corner_pts / (ops.expand_dims(corner_pts[:,:,2], 2)) #normalize
            corner_pts = corner_pts[:, :, :-1]
            corner_pts_pos = corner_pts[sup_pos_ids]
            template_poly_pos = template_poly[sup_pos_ids]
            corner_error = ops.ReduceSum()(ops.abs(corner_pts_pos - template_poly_pos)) / sup_pos_ids.shape[0] / 4
            corner_pts_aug = aug_sear_points.transpose(0,2,1)[:, :, :-1]
            corner_pts_aug_pos = corner_pts_aug[sup_pos_ids]
            corner_error_aug = ops.ReduceSum()(ops.abs(corner_pts_aug_pos - template_poly_pos)) / sup_pos_ids.shape[0] / 4


        #supervised_loss
        if len(sup_pos_ids) > 0 :
            #corner error for supervised data(the crop)
            #H_gt
            homo_points = pred_points.transpose(0,2,1)
            homo_points = (homo_points / (ops.expand_dims(homo_points[:,:,2], 2)))[:,:,:-1]
            H_gt = DLT_solve(template_poly.reshape(-1,8), (homo_points - template_poly).reshape(-1,8)).squeeze(1)[sup_pos_ids]  #H: search -> template
            #construct a 127*127 search points.
            search_corner = ops.cast(numpy.tile(ops.expand_dims(ms.Tensor([[0,0], [0,127], [127,127], [127,0]]), 0), (H_gt.shape[0], 1, 1)), ms.float32)
            poly_ones_tmp_var = ops.ones((H_gt.shape[0],1,4), ms.float32)
            search_corner_hmg = ops.concat((search_corner.transpose(0,2,1), poly_ones_tmp_var),1)

            # #warp the search corner accroding to H_gt
            search_corner_warped = ops.BatchMatMul()(H_gt, search_corner_hmg).transpose(0,2,1)
            search_corner_warped = search_corner_warped / (ops.expand_dims(search_corner_warped[:,:,2], 2)) #normalize
            search_corner_warped = search_corner_warped[:, :, :-1]

            # #delta error.
            search_corner_delta = (search_corner_warped - search_corner).reshape(-1,8)
            pred_delta = batch_out['x'][sup_pos_ids]
            corner_loss_pos =  kalyo_l1_loss(pred_delta, search_corner_delta)


        neg_ids = ops.equal(if_neg, 1).nonzero().squeeze(1)
        if len(neg_ids):
            pred_neg_delta = batch_out['x'][neg_ids]
            neg_zeros = ops.zeros_like(pred_neg_delta)
            corner_loss_neg = kalyo_l1_loss(pred_neg_delta, neg_zeros)


        outputs = {}
        if cfg.TRAIN.OBJ == 'ALL':
            outputs['cls_loss_sup'] =  ms.Tensor(0.0)
            outputs['loc_loss_sup'] = ms.Tensor(0.0)
            outputs['cls_loss_lp_sup'] = ms.Tensor(0.0)
            outputs['loc_loss_lp_sup'] = ms.Tensor(0.0)
            outputs['sim_cent_loss'] = ms.Tensor(0.0)
            outputs['sim_loss'] =  ms.Tensor(0.0)
            outputs['cor_err_aug'] = ms.Tensor(0.0)
            outputs['cor_err_sim'] = ms.Tensor(0.0)
            outputs['cor_err'] = ms.Tensor(0.0)
            outputs['cor_loss'] = ms.Tensor(0.0)
            outputs['cor_pos_loss'] = ms.Tensor(0.0)
            outputs['cor_neg_loss'] = ms.Tensor(0.0)
            outputs['homo_unsup_loss'] = ms.Tensor(0.0)

            if len(sup_ids) > 0 :
                outputs['cls_loss_sup'] = cls_loss_sup
                outputs['loc_loss_sup'] = loc_loss_sup
                outputs['cls_loss_lp_sup'] = cls_loss_lp_sup
                outputs['loc_loss_lp_sup'] = loc_loss_lp_sup
                outputs['sim_cent_loss'] = cfg.TRAIN.CLS_WEIGHT * (cls_loss_lp_sup + cls_loss_sup) + \
                                           cfg.TRAIN.LOC_WEIGHT * (loc_loss_sup + loc_loss_lp_sup)#supervisedv  similarity component
            if len(sup_pos_ids) > 0:
                outputs['cor_err_aug'] = corner_error_aug #supervised
                outputs['cor_err'] = corner_error #supervised
                outputs['cor_loss'] += corner_loss_pos
                outputs['cor_pos_loss'] = corner_loss_pos
            if len(sup_neg_ids) > 0:
                outputs['cor_loss'] += corner_loss_neg
                outputs['cor_neg_loss'] = corner_loss_neg
            if len(unsup_ids) > 0:
                outputs['homo_unsup_loss'] = loss_feature
            if len(unsup_pos_ids) > 0:
                outputs['sim_loss']  = sim_loss #unsupervised
                outputs['cor_err_sim'] = corner_error_simi #supervised
            # outputs['total_loss'] = outputs['sim_cent_loss']*100 + outputs['homo_unsup_loss'] + outputs['cor_neg_loss'] + outputs['cor_err']/4
            outputs['total_loss'] =  outputs['homo_unsup_loss'] + outputs['cor_neg_loss'] + outputs['cor_err']/4
        return outputs['total_loss']