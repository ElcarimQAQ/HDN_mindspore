import mindspore as ms
import numpy as np
import cv2
import subprocess
import psutil

from mindspore import ops, Tensor, numpy
from mindspore.scipy.linalg import inv

def DLT_solve(src_p, off_set):
    # src_p: shape=(bs, n, 4, 2)
    # off_set: shape=(bs, n, 4, 2)
    # can be used to compute mesh points (multi-H)
    bs = src_p.shape[0]
    divide = int(np.sqrt(len(src_p[0])/2)-1)
    row_num = (divide+1)*2

    for i in range(divide):
        for j in range(divide):
            try:
                h4p = src_p[:,[2*j+row_num*i, 2*j+row_num*i+1,
                    2*(j+1)+row_num*i, 2*(j+1)+row_num*i+1, 
                    2*(j+1)+row_num*i+row_num, 2*(j+1)+row_num*i+row_num+1,
                    2*j+row_num*i+row_num, 2*j+row_num*i+row_num+1]].reshape(bs, 1, 4, 2)
            except:
                print("h4p reshape error")
            
            pred_h4p = off_set[:,[2*j+row_num*i, 2*j+row_num*i+1, 
                    2*(j+1)+row_num*i, 2*(j+1)+row_num*i+1, 
                    2*(j+1)+row_num*i+row_num, 2*(j+1)+row_num*i+row_num+1,
                    2*j+row_num*i+row_num, 2*j+row_num*i+row_num+1]].reshape(bs, 1, 4, 2)

            if i+j==0:
                src_ps = h4p
                off_sets = pred_h4p
            else:
                src_ps = ops.concat((src_ps, h4p), axis = 1)
                off_sets = ops.concat((off_sets, pred_h4p), axis = 1)

    bs, n, h, w = src_ps.shape

    N = bs*n

    src_ps = src_ps.reshape(N, h, w)
    off_sets = off_sets.reshape(N, h, w)

    dst_p = src_ps + off_sets

    ones = ops.ones((N, 4, 1), ms.float32)
    xy1 = ops.concat((src_ps, ones), 2)
    zeros = ops.zeros_like(xy1)

    xyu, xyd = ops.concat((xy1, zeros), 2), ops.concat((zeros, xy1), 2)
    M1 = ops.concat((xyu, xyd), 2).reshape(N, -1, 6)
    M2 = ops.matmul(
        dst_p.reshape(-1, 2, 1), 
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = ops.concat((M1, -M2), 2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = ops.expand_dims(inv(A[0,:,:]),0) # Todo:有一定偏差
    h8 = ops.matmul(Ainv, b).reshape(N, 8)
 
    H = ops.concat((h8, ones[:,0,:]), 1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)
    return H

 
def transformer(U, theta, out_size, **kwargs):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """

    def _repeat(x, n_repeats):

        rep = ops.expand_dims(ops.ones((n_repeats, ), ms.float32), 0)
        rep = ops.cast(rep, ms.float32)
        x = ops.cast(x, ms.float32)

        x = ops.cast(ops.matmul(x.reshape([-1,1]), rep), ms.int32)
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size, scale_h):

        num_batch, num_channels , height, width = im.shape

        height_f = height
        width_f = width
        out_height, out_width = out_size[0], out_size[1]

        zero = 0
        max_y = height - 1
        max_x = width - 1
        if scale_h:

            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = ops.cast(ops.floor(x), ms.int32)
        x1 = x0 + 1
        y0 = ops.cast(ops.floor(y), ms.int32)
        y1 = y0 + 1

        x0 = ops.clip_by_value(x0, zero, max_x)
        x1 = ops.clip_by_value(x1, zero, max_x)
        y0 = ops.clip_by_value(y0, zero, max_y)
        y1 = ops.clip_by_value(y1, zero, max_y)
        dim2 = Tensor.from_numpy( np.array(width) )
        dim1 = Tensor.from_numpy( np.array(width * height) )

        base = _repeat(numpy.arange(0,num_batch) * dim1, out_height * out_width)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # channels dim
        im = ops.transpose(im, (0,2,3,1))
        im_flat = ops.cast(im.reshape([-1, num_channels]), ms.float32)

        idx_a = ops.cast(ops.expand_dims(idx_a, -1), ms.int64)
        idx_a = ops.broadcast_to(idx_a, (height * width * num_batch,num_channels))
        Ia = ops.gather_elements(im_flat, 0, idx_a)

        idx_b = ops.cast(ops.expand_dims(idx_b, -1), ms.int64)
        idx_b = ops.broadcast_to(idx_b, (height * width * num_batch, num_channels))
        Ib = ops.gather_elements(im_flat, 0, idx_b)

        idx_c = ops.cast(ops.expand_dims(idx_c, -1), ms.int64)
        idx_c = ops.broadcast_to(idx_c, (height * width * num_batch, num_channels))
        Ic = ops.gather_elements(im_flat, 0, idx_c)

        idx_d = ops.cast(ops.expand_dims(idx_d, -1), ms.int64)
        idx_d = ops.broadcast_to(idx_d, (height * width * num_batch, num_channels))
        Id = ops.gather_elements(im_flat, 0, idx_d)

        x0_f = ops.cast(x0, ms.float32)
        x1_f = ops.cast(x1, ms.float32)
        y0_f = ops.cast(y0, ms.float32)
        y1_f = ops.cast(y1, ms.float32)

        wa = ops.expand_dims(((x1_f - x) * (y1_f - y)), 1)
        wb = ops.expand_dims(((x1_f - x) * (y - y0_f)), 1)
        wc = ops.expand_dims(((x - x0_f) * (y1_f - y)), 1)
        wd = ops.expand_dims(((x - x0_f) * (y - y0_f)), 1)
        output = wa*Ia+wb*Ib+wc*Ic+wd*Id

        return output

    def _meshgrid(height, width, scale_h):

        if scale_h:
            x_t = ops.matmul(ops.ones((height, 1), ms.float32),
                               ops.transpose(ops.expand_dims(ops.linspace(ms.Tensor(-1.0), ms.Tensor(1.0), width), 1),(1, 0)))
            y_t = ops.matmul(ops.expand_dims(ops.linspace(ms.Tensor(-1.0), ms.Tensor(1.0), height), 1),
                               ops.ones((1, width), ms.float32))#grid:[-1, 1]
        else:
            x_t = ops.matmul(ops.ones((height, 1), ms.float32),
                               ops.transpose(ops.expand_dims(ops.linspace(ms.Tensor(0.0), ms.Tensor(ops.broadcast_to(width, ms.float32)), width), 1), (1, 0)))
            y_t = ops.matmul(ops.expand_dims(ops.linspace(ms.Tensor(0.0), ms.Tensor(ops.broadcast_to(height, ms.float32)), height), 1),
                               ops.ones((1, width), ms.float32))#grid:[0, width(or height)]


        x_t_flat = ops.cast(x_t.reshape((1, -1)), ms.float32)
        y_t_flat = ops.cast(y_t.reshape((1, -1)), ms.float32)

        ones = ops.ones_like(x_t_flat)
        grid = ops.concat([x_t_flat, y_t_flat, ones], 0)

        return grid

    def _transform(theta, input_dim, out_size, scale_h):
        num_batch, num_channels , height, width = input_dim.shape
        #  Changed
        theta = ops.cast(theta.reshape([-1, 3, 3]), ms.float32)

        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, scale_h)
        grid = ops.expand_dims(grid, 0).reshape([1,-1])
        shape = grid.shape
        grid = ops.broadcast_to(grid, (num_batch,shape[1]))
        grid = grid.reshape([num_batch, 3, -1])

        T_g = ops.matmul(theta, grid)
        x_s = T_g[:,0,:]
        y_s = T_g[:,1,:]
        t_s = T_g[:,2,:]

        t_s_flat = t_s.reshape([-1])

        # smaller
        small = 1e-7
        smallers = 1e-6*(1.0 - ops.cast(ops.ge(t_s_flat.abs(), small), ms.float32))

        t_s_flat = t_s_flat + smallers
        condition = ops.ReduceSum()((ops.cast(ops.gt(t_s_flat.abs(), small), ms.float32)))
        # Ty changed
        x_s_flat = x_s.reshape([-1]) / t_s_flat
        y_s_flat = y_s.reshape([-1]) / t_s_flat

        input_transformed = _interpolate( input_dim, x_s_flat, y_s_flat,out_size,scale_h)

        output = input_transformed.reshape([num_batch, out_height, out_width, num_channels ])
        return output, condition

    img_w = U.shape[2]
    img_h = U.shape[1]

    scale_h = True
    output, condition = _transform(theta, U, out_size, scale_h)
    return output, condition


def transform(patch_size_h,patch_size_w,M_tile_inv,H_mat,M_tile,I1,patch_indices,batch_indices_tensor):
    # Transform H_mat since we scale image indices in transformer
    batch_size, num_channels, img_h, img_w = I1.shape
    H_mat = ops.matmul(ops.matmul(M_tile_inv, H_mat), M_tile)

    # Transform image 1 (large image) to image 2
    out_size = (img_h, img_w)
    warped_images, _ = transformer(I1, H_mat, out_size)
    warped_images_flat = warped_images.reshape([-1,num_channels])
    patch_indices_flat = patch_indices.reshape([-1])
    pixel_indices = ops.cast(patch_indices_flat, ms.int64) + batch_indices_tensor
    pixel_indices = ops.cast(ops.expand_dims(pixel_indices, 1), ms.int64)
    pixel_indices = ops.broadcast_to(pixel_indices, (patch_size_h*patch_size_w*batch_size, num_channels))
    pred_I2_flat = ops.gather_elements(warped_images_flat, 0, pixel_indices)
    pred_I2 = pred_I2_flat.reshape([batch_size, patch_size_h, patch_size_w, num_channels])
    return ops.transpose(pred_I2, (0,3,1,2))




