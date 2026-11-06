from functools import partial
from ...utils.spconv_utils import replace_feature, spconv
import torch.nn as nn
import numpy as np
import torch
import os
from pathlib import Path
from pcdet.datasets.augmentor.X_transform import X_TRANS

import math

from pcgc.data_utils import sort_spare_tensor, write_ply_ascii_geo, read_ply_ascii_geo
from pcgc.gpcc import gpcc_encode, gpcc_decode
from pcgc.codec_config import feature_shapes, validate_shape
from pcgc.attribute_codec import encode_quantized, decode_quantized
from transformer.helpers import get_clones
from transformer.transformer import TransformerEncoderLayer


def index2points(indices, pts_range=[0, -40, -3, 70.4, 40, 1], voxel_size=[0.05, 0.05, 0.05], stride=8):
    """
    convert 3D voxel indices to a set of grid points.
    """

    voxel_size = np.array(voxel_size) * stride
    min_x = pts_range[0] + voxel_size[0] / 2
    min_y = pts_range[1] + voxel_size[1] / 2
    min_z = pts_range[2] + voxel_size[2] / 2

    new_indices = indices.clone().float()
    indices_float = indices.clone().float()
    new_indices[:, 1] = indices_float[:, 3] * voxel_size[0] + min_x
    new_indices[:, 2] = indices_float[:, 2] * voxel_size[1] + min_y
    new_indices[:, 3] = indices_float[:, 1] * voxel_size[2] + min_z

    return new_indices


def index2uv(indices, batch_size, calib, stride, x_trans_train, trans_param):
    """
    convert the 3D voxel indices to image pixel indices.
    """

    new_uv = indices.new(size=(indices.shape[0], 3))
    depth = indices.new(size=(indices.shape[0], 1)).float()
    for b_i in range(batch_size):
        cur_in = indices[indices[:, 0] == b_i]
        cur_pts = index2points(cur_in, stride=stride)
        if trans_param is not None:
            transed = x_trans_train.backward_with_param({'points': cur_pts[:, 1:4],
                                                         'transform_param': trans_param[b_i]})
            cur_pts = transed['points']
        else:
            cur_pts = cur_pts[:, 1:4]

        pts_rect = calib[b_i].lidar_to_rect_cuda(cur_pts[:, 0:3])
        pts_img, pts_rect_depth = calib[b_i].rect_to_img_cuda(pts_rect)

        pts_img = pts_img.int()

        new_uv[indices[:, 0] == b_i, 1:3] = pts_img

        depth[indices[:, 0] == b_i, 0] = pts_rect_depth[:]
    new_uv[:, 0] = indices[:, 0]
    new_uv[:, 1] = torch.clamp(new_uv[:, 1], min=0, max=1400 - 1) // stride
    new_uv[:, 2] = torch.clamp(new_uv[:, 2], min=0, max=600 - 1) // stride

    return new_uv, depth


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m


def post_act_block2d(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                     conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m


def layer_voxel_discard(sparse_t, rat=0.15):
    """
    discard the voxels based on the given rate.
    """

    if rat == 0:
        return

    len = sparse_t.features.shape[0]
    randoms = np.random.permutation(len)
    randoms = torch.from_numpy(randoms[0:int(len * (1 - rat))]).to(sparse_t.features.device)

    sparse_t = replace_feature(sparse_t, sparse_t.features[randoms])
    sparse_t.indices = sparse_t.indices[randoms]


class NRConvBlock(nn.Module):
    """
    convolve the voxel features in both 3D and 2D space.
    """

    def __init__(self, input_c=16, output_c=16, stride=1, padding=1, indice_key='vir1', conv_depth=False):
        super(NRConvBlock, self).__init__()
        self.stride = stride
        block = post_act_block
        block2d = post_act_block2d
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_depth = conv_depth

        if self.stride > 1:
            self.down_layer = block(input_c,
                                    output_c,
                                    3,
                                    norm_fn=norm_fn,
                                    stride=stride,
                                    padding=padding,
                                    indice_key=('sp' + indice_key),
                                    conv_type='spconv')
        c1 = input_c

        if self.stride > 1:
            c1 = output_c
        if self.conv_depth:
            c1 += 4

        c2 = output_c

        self.d3_conv1 = block(c1,
                              c2 // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm1' + indice_key))
        self.d2_conv1 = block2d(c2 // 2,
                                c2 // 2,
                                3,
                                norm_fn=norm_fn,
                                padding=1,
                                indice_key=('subm3' + indice_key))

        self.d3_conv2 = block(c2 // 2,
                              c2 // 2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm2' + indice_key))
        self.d2_conv2 = block2d(c2 // 2,
                                c2 // 2,
                                3,
                                norm_fn=norm_fn,
                                padding=1,
                                indice_key=('subm4' + indice_key))

    def forward(self, sp_tensor, batch_size, calib, stride, x_trans_train, trans_param):

        if self.stride > 1:
            sp_tensor = self.down_layer(sp_tensor)

        d3_feat1 = self.d3_conv1(sp_tensor)
        d3_feat2 = self.d3_conv2(d3_feat1)

        uv_coords, depth = index2uv(d3_feat2.indices, batch_size, calib, stride, x_trans_train, trans_param)

        d2_sp_tensor1 = spconv.SparseConvTensor(
            features=d3_feat2.features,
            indices=uv_coords.int(),
            spatial_shape=[1600, 600],
            batch_size=batch_size
        )

        d2_feat1 = self.d2_conv1(d2_sp_tensor1)
        d2_feat2 = self.d2_conv2(d2_feat1)

        d3_feat3 = replace_feature(d3_feat2, torch.cat([d3_feat2.features, d2_feat2.features], -1))

        return d3_feat3


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
                     torch.ones((src_range[0].shape[0], 3), device=src_range[0].device), ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff) + dst_range[0][:, None, :]
    return prop_xyz


def array2vector(array, step):
    """ravel 2D array with multi-channel to one 1D vector by sum each channel with different step.
    """
    array, step = array.long().cpu(), step.long().cpu()

    vector = sum([array[:, i] * (step ** i) for i in range(array.shape[-1])])

    return vector


def sort_indices(indices):
    """
        Sort points in sparse tensor according to their coordinates.
    """
    indices_sort = np.argsort(array2vector(indices.cpu(), indices.cpu().max() + 1))
    indices_new = indices[indices_sort]

    return indices_new


def compute_mask(xyz_1, xyz_2, radius, threshold=False):
    with torch.no_grad():
        xyz_1_size = xyz_1.size(1)
        xyz_2_size = xyz_2.size(1)
        xyz_1_expanded = xyz_1.unsqueeze(1).expand(1, xyz_2_size, xyz_1_size, 3)
        xyz_2_expanded = xyz_2.unsqueeze(2).expand(1, xyz_2_size, xyz_1_size, 3)
        dist = torch.sqrt(torch.sum(torch.square(xyz_1_expanded - xyz_2_expanded), dim=-1))
        dist = dist.transpose(1, 2)
        if threshold == False:
            mask = dist >= radius
        else:
            mask = dist <= radius
    return mask, dist


def compute_big_dist(xyz_1, xyz_2, radius, computation_th):

    num_sub_tensors = len(xyz_2) // computation_th
    all_ind = torch.tensor([], dtype=torch.long).cuda()

    for i in range(num_sub_tensors):
        sub_tensors = xyz_2[(i * computation_th): ((i + 1) * computation_th)]
        xyz_1_size = xyz_1.size(0)
        xyz_2_size = sub_tensors.size(0)
        xyz_1_expanded = xyz_1.unsqueeze(0).expand(xyz_2_size, xyz_1_size, 3)
        xyz_2_expanded = sub_tensors.unsqueeze(1).expand(xyz_2_size, xyz_1_size, 3)
        dist = torch.sqrt(torch.sum(torch.square(xyz_1_expanded - xyz_2_expanded), dim=-1))
        ind = torch.where(dist < radius)[1]
        all_ind = torch.cat((all_ind, ind), dim=0)

    if (num_sub_tensors * computation_th) != len(xyz_2):
        remaining_tensor = xyz_2[(num_sub_tensors * computation_th):]
        xyz_1_size = xyz_1.size(0)
        xyz_2_size = remaining_tensor.size(0)
        xyz_1_expanded = xyz_1.unsqueeze(0).expand(xyz_2_size, xyz_1_size, 3)
        xyz_2_expanded = remaining_tensor.unsqueeze(1).expand(xyz_2_size, xyz_1_size, 3)
        dist = torch.sqrt(torch.sum(torch.square(xyz_1_expanded - xyz_2_expanded), dim=-1))
        ind = torch.where(dist < radius)[1]
        all_ind = torch.cat((all_ind, ind), dim=0)

    return all_ind


def compute_big_equal(xyz_1, xyz_2, computation_th):

    num_sub_tensors = len(xyz_2) // computation_th
    all_ind = torch.tensor([], dtype=torch.long).cuda()

    for i in range(num_sub_tensors):
        sub_tensors = xyz_2[(i * computation_th): ((i + 1) * computation_th)]
        equal_points = torch.eq(sub_tensors.unsqueeze(1), xyz_1.unsqueeze(0)).all(dim=-1)
        ind = torch.nonzero(equal_points, as_tuple=False)[:, 1:].squeeze()
        if ind.shape == torch.Size([]):
            ind = ind.unsqueeze(0)
            all_ind = torch.cat((all_ind, ind), dim=0)
        else:
            all_ind = torch.cat((all_ind, ind), dim=0)

    if (num_sub_tensors * computation_th) != len(xyz_2):
        remaining_tensor = xyz_2[(num_sub_tensors * computation_th):]
        equal_points = torch.eq(remaining_tensor.unsqueeze(1), xyz_1.unsqueeze(0)).all(dim=-1)
        ind = torch.nonzero(equal_points, as_tuple=False)[:, 1:].squeeze()
        if ind.shape == torch.Size([]):
            ind = ind.unsqueeze(0)
            all_ind = torch.cat((all_ind, ind), dim=0)
        else:
            all_ind = torch.cat((all_ind, ind), dim=0)

    return all_ind


def sin_xyz_embedding(sp_tensor, num_channels, temperature, scale):

    feature = sp_tensor.features
    xyz = sp_tensor.indices

    z_idx = xyz[:, 1]
    y_idx = xyz[:, 2]
    x_idx = xyz[:, 3]
    xyz_tensor = torch.stack([x_idx, y_idx, z_idx], dim=0).transpose(0, 1)

    range_min, _ = torch.min(xyz_tensor, dim=0, keepdim=True)
    range_max, _ = torch.max(xyz_tensor, dim=0, keepdim=True)
    input_range = [
        range_min,
        range_max,
    ]

    xyz_tensor = torch.unsqueeze(xyz_tensor, dim=0).cuda()
    xyz = xyz_tensor.clone()

    xyz = shift_scale_points(xyz, src_range=input_range)
    ndim = num_channels // xyz.shape[2]
    if ndim % 2 != 0:
        ndim -= 1

    rems = num_channels - (ndim * xyz.shape[2])
    assert (ndim % 2 == 0), f"Cannot handle odd sized ndim={ndim} where num_channels={num_channels} and xyz={xyz.shape}"
    final_embeds = []
    prev_dim = 0

    for d in range(xyz.shape[2]):
        cdim = ndim
        if rems > 0:

            cdim += 2
            rems -= 2

        if cdim != prev_dim:
            dim_t = torch.arange(cdim, dtype=torch.float32, device=xyz.device)
            dim_t = temperature ** (2 * (dim_t // 2) / cdim)

        raw_pos = xyz[:, :, d]
        if 1:
            raw_pos *= scale
        pos = raw_pos[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        final_embeds.append(pos)
        prev_dim = cdim
    position_embedding = torch.cat(final_embeds, dim=2).permute(0, 2, 1)

    feature = feature.unsqueeze(0)
    feature = feature.transpose(1, 2)
    src = torch.add(position_embedding, feature)
    src = src.transpose(0, 2).transpose(1, 2)

    return src, xyz_tensor, position_embedding


def cube_up_sampling(coordinates, surrounding_param, dense_param):
    expanded_points = []
    surrounding_param = int(surrounding_param * dense_param)
    for z in range(-surrounding_param, surrounding_param + 1):
        for y in range(-surrounding_param, surrounding_param + 1):
            for x in range(-surrounding_param, surrounding_param + 1):
                expanded_points.append(coordinates + (torch.tensor([0, z, y, x]) * (1 / dense_param)).cuda())
    expanded_points = torch.cat(expanded_points, dim=0)
    return expanded_points

# Select on both channels and keep the union, followed by radius compensation.

def slope_threshold(slope_channel_1, slope_channel_2,
                    step_channel_1, step_channel_2,
                    range_b_channel_1_top, range_b_channel_2_top,
                    range_b_channel_1_bottom, range_b_channel_2_bottom,
                    draw_curve_channel_1, draw_curve_channel_2,
                    radius_threshold_channel_1, radius_threshold_channel_2,
                    input_sp_tensor, print_threshold_results=True):

    all_feature = input_sp_tensor.features
    num_before_threshold = all_feature.shape[0]

    def find_first_intersection(y, k, step_size, range_b_top, range_b_bottom, draw_curve):
        def find_intersection(x_data, y_data, line_function):
            indices = torch.where(
                (y_data[:-1] <= line_function(x_data[:-1])) & (y_data[1:] >= line_function(x_data[1:])))
            if len(indices[0]) > 0:
                i = indices[0][0].item()
                x_intersection = x_data[i].item()
                y_intersection = line_function(x_intersection)
                return x_intersection, y_intersection
            return None

        y = y.cuda()
        half_length = len(y) // 2

        y[:half_length] = y[half_length]
        x = torch.arange(1, y.shape[0] + 1, dtype=torch.float32).cuda()
        line_function = lambda x: k * x + b_value
        intersection_flag = False

        for b_value in np.arange(-range_b_bottom, range_b_top, step_size):
            intersection_point = find_intersection(x, y, line_function)
            if intersection_point:

                intersection_flag = True
                return int(intersection_point[0])
                break

        if intersection_flag == False:
            return int(y.shape[0] - 1)

    all_feature_channel_1 = all_feature[:, :1]
    sorted_features_channel_1, sorted_indices_channel_1 = torch.sort(all_feature_channel_1, dim=0)
    indices_channel_1_num = find_first_intersection(sorted_features_channel_1.squeeze(dim=1), slope_channel_1,
                                                    step_channel_1, range_b_channel_1_top, range_b_channel_1_bottom,
                                                    draw_curve_channel_1)
    indices_channel_1 = sorted_indices_channel_1[indices_channel_1_num:, 0]

    all_feature_channel_2 = all_feature[:, 1:2]
    sorted_features_channel_2, sorted_indices_channel_2 = torch.sort(all_feature_channel_2, dim=0)
    indices_channel_2_num = find_first_intersection(sorted_features_channel_2.squeeze(dim=1), slope_channel_2,
                                                    step_channel_2, range_b_channel_2_top, range_b_channel_2_bottom,
                                                    draw_curve_channel_2)
    indices_channel_2 = sorted_indices_channel_2[indices_channel_2_num:, 0]

    'Keep neighboring points within the specified radius to compensate for sparse threshold-selected indices.'
    if radius_threshold_channel_1 | radius_threshold_channel_2 != 0:

        indices_all = input_sp_tensor.indices
        z_origin_idx_all = indices_all[:, 1]
        y_origin_idx_all = indices_all[:, 2]
        x_origin_idx_all = indices_all[:, 3]
        indices_xyz_all = torch.stack([x_origin_idx_all, y_origin_idx_all, z_origin_idx_all], dim=0).transpose(0, 1)
        indices_xyz_all = indices_xyz_all.unsqueeze(0)
        indices_xyz_all = indices_xyz_all.to(dtype=torch.float32)

        if radius_threshold_channel_1 != 0:

            origin_indices_channel_1 = input_sp_tensor.indices[indices_channel_1]
            z_origin_idx_channel_1 = origin_indices_channel_1[:, 1]
            y_origin_idx_channel_1 = origin_indices_channel_1[:, 2]
            x_origin_idx_channel_1 = origin_indices_channel_1[:, 3]
            origin_indices_xyz_channel_1 = torch.stack(
                [x_origin_idx_channel_1, y_origin_idx_channel_1, z_origin_idx_channel_1], dim=0).transpose(0, 1)
            origin_indices_xyz_channel_1 = origin_indices_xyz_channel_1.unsqueeze(0)
            origin_indices_xyz_channel_1 = origin_indices_xyz_channel_1.to(dtype=torch.float32)
            mask_channel_1, _ = compute_mask(origin_indices_xyz_channel_1, indices_xyz_all,
                                             radius_threshold_channel_1, threshold=True)
            mask_channel_1 = mask_channel_1.squeeze(0)
            indices_channel_1 = torch.tensor([]).cuda()
            
            for row in range(mask_channel_1.size(0)):
                indices_channel_1 = torch.cat((indices_channel_1, torch.nonzero(mask_channel_1[row, :] == True)),
                                              dim=0)
            
            indices_channel_1 = torch.unique(indices_channel_1, return_inverse=False)
            indices_channel_1 = indices_channel_1.to(dtype=torch.int32)

        if radius_threshold_channel_2 != 0:

            origin_indices_channel_2 = input_sp_tensor.indices[indices_channel_2]
            z_origin_idx_channel_2 = origin_indices_channel_2[:, 1]
            y_origin_idx_channel_2 = origin_indices_channel_2[:, 2]
            x_origin_idx_channel_2 = origin_indices_channel_2[:, 3]
            origin_indices_xyz_channel_2 = torch.stack(
                [x_origin_idx_channel_2, y_origin_idx_channel_2, z_origin_idx_channel_2], dim=0).transpose(0, 1)
            origin_indices_xyz_channel_2 = origin_indices_xyz_channel_2.unsqueeze(0)
            origin_indices_xyz_channel_2 = origin_indices_xyz_channel_2.to(dtype=torch.float32)
            mask_channel_2, _ = compute_mask(origin_indices_xyz_channel_2, indices_xyz_all,
                                             radius_threshold_channel_2, threshold=True)
            mask_channel_2 = mask_channel_2.squeeze(0)
            indices_channel_2 = torch.tensor([]).cuda()
            
            for row in range(mask_channel_2.size(0)):
                indices_channel_2 = torch.cat((indices_channel_2, torch.nonzero(mask_channel_2[row, :] == True)),
                                              dim=0)
            
            indices_channel_2 = torch.unique(indices_channel_2, return_inverse=False)
            indices_channel_2 = indices_channel_2.to(dtype=torch.int32)

    save_all_indices = torch.tensor(list(set(indices_channel_1.tolist()) | set(indices_channel_2.tolist())))

    new_indices = input_sp_tensor.indices[save_all_indices]
    input_sp_tensor.indices = new_indices
    feature_old = input_sp_tensor.features
    feature_new = feature_old[save_all_indices]
    input_sp_tensor = input_sp_tensor.replace_feature(feature_new)

    return input_sp_tensor

# Enhance sparse features using the existing masked attention.

class MaskedTransformer(nn.Module):
    """
        features enhancement using transformer.
    """

    def __init__(self, enc_nlayers=4, enc_dim=4, enc_ffn_dim=32, enc_nhead=4, enc_dropout=0.1, enc_activation="relu",
                 norm=None, mask_radius=[3, 4, 5, 6]):
        super(MaskedTransformer, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(d_model=enc_dim, nhead=enc_nhead, dim_feedforward=enc_ffn_dim,
                                                     dropout=enc_dropout, activation=enc_activation)
        self.layers = get_clones(self.encoder_layer, enc_nlayers)
        self.masking_radius = [math.pow(x, 2) for x in mask_radius]
        self.norm = norm

    def forward(self, transformer_input, xyz_tensor):

        transformer_output = transformer_input

        for idx, layer in enumerate(self.layers):
            mask = None
            if self.masking_radius[idx] > 0:
                mask, xyz_dist = compute_mask(xyz_tensor, xyz_tensor, self.masking_radius[idx])

                bsz, n, n = mask.shape
                nhead = layer.nhead
                mask = mask.unsqueeze(1)
                mask = mask.repeat(1, nhead, 1, 1)
                mask = mask.view(bsz * nhead, n, n)

            transformer_output = layer(transformer_output, src_mask=mask, src_key_padding_mask=None, pos=None)

        return transformer_output

# Keep the existing sparse pooling and coordinate conventions.

class MaxPooling(spconv.SparseModule):
    '''
    feature maxpooling
    '''

    def __init__(self, kernel_size=(3, 3, 3), padding=0, stride=3, print_maxpooling_results=False, max_output_num=0):
        super().__init__()
        self.maxpooling = spconv.SparseMaxPool3d(kernel_size=kernel_size, padding=padding, stride=stride)
        self.print_maxpooling_results = print_maxpooling_results
        self.stride = stride
        self.max_output_num = max_output_num

    def forward(self, input_sp_tensor):
        original_spatial_shape = input_sp_tensor.spatial_shape
        output_sp_tensor = self.maxpooling(input_sp_tensor)
        output_sp_tensor.spatial_shape = original_spatial_shape

        output_sp_tensor.indices = output_sp_tensor.indices * self.stride
        if self.max_output_num != 0:
            save_num = len(output_sp_tensor.indices)
            if save_num >= self.max_output_num:
                
                save_num = self.max_output_num
                original_indices = output_sp_tensor.indices
                save_indices = torch.randperm(len(original_indices))
                save_indices = save_indices[:save_num]

                output_sp_tensor.indices = output_sp_tensor.indices[save_indices]
                feature_old = output_sp_tensor.features
                feature_new = feature_old[save_indices]
                output_sp_tensor = output_sp_tensor.replace_feature(feature_new)

        return output_sp_tensor

# Legacy codec: G-PCC geometry and quantized CABAC attributes.

class EntropyEncDec(nn.Module):
    """
        features & indices entropy encoder.
    """
    def __init__(self, quantization_parameter=1, interval=1.0, step_size=1.0, _lambda=0.0, save_path="",
                 print_enc_time=False, del_ply=False, spatial_shape=None, legacy_shape_header=False):
        super().__init__()
        self.quantization_parameter = quantization_parameter
        self.spatial_shape = validate_shape(spatial_shape) if spatial_shape is not None else None
        self.legacy_shape_header = legacy_shape_header
        self.decode_device = None
        self.interval = interval
        self.step_size = step_size
        self._lambda = _lambda
        self.save_path = save_path
        self.print_enc_time = print_enc_time
        self.del_ply = del_ply

    def encode(self, input_sp_tensor, eval_name_str):
        if self.legacy_shape_header:
            raise ValueError('legacy_shape_header is decode-only; new streams have no spatial prefix.')

        if input_sp_tensor.batch_size != 1 or (input_sp_tensor.indices[:, 0] != 0).any():
            raise ValueError('Legacy entropy format supports only a single frame, batch index 0.')
        if input_sp_tensor.features.numel() == 0:
            raise ValueError('Legacy entropy format cannot encode an empty feature cloud.')
        if self.spatial_shape is None:
            raise ValueError('Configure the fixed decoder spatial_shape before encoding.')
        if validate_shape(input_sp_tensor.spatial_shape) != self.spatial_shape:
            raise ValueError('Input spatial shape differs from the configured decoder grid.')
        if not torch.isfinite(input_sp_tensor.features).all():
            raise ValueError('Features must be finite.')
        if self.quantization_parameter <= 0:
            raise ValueError('quantization_parameter must be positive.')

        self.decode_device = input_sp_tensor.features.device

        file_name = self.save_path
        ply_file_name = self.save_path + "input_ply/" + eval_name_str + "_input.ply"

        y_order = sort_spare_tensor(input_sp_tensor)

        float_feature = y_order.features

        int_feature = float_feature * self.quantization_parameter
        int_feature = torch.round(int_feature).cpu().numpy()

        stream_bytes = encode_quantized(int_feature, self.interval, self.step_size, self._lambda)

        directory = os.path.dirname(file_name +"_F/" +eval_name_str + '_F.bin')
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_name +"_F/" +eval_name_str + '_F.bin', 'wb') as f_out:
            f_out.write(stream_bytes)

        y_order_coords = y_order.indices[:, 1:].int().detach().cpu().numpy()

        directory = os.path.dirname(ply_file_name)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        write_ply_ascii_geo(filedir=ply_file_name, coords=y_order_coords)

        directory = os.path.dirname(file_name + "_C/" + eval_name_str + '_C.bin')
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        gpcc_encode(ply_file_name, file_name +"_C/" +eval_name_str + '_C.bin')

    def decode(self, eval_name_str):
        if self.spatial_shape is None:
            raise ValueError("Decode requires the fixed spatial_shape from model configuration.")

        file_name = self.save_path
        ply_file_name = self.save_path +"output_ply/" + eval_name_str + "_output.ply"

        directory = os.path.dirname(ply_file_name)
        
        if not os.path.exists(directory):
            os.makedirs(directory)

        gpcc_decode(file_name  +"_C/" +eval_name_str +  '_C.bin', ply_file_name)

        y_decoder_indices = read_ply_ascii_geo(ply_file_name)
        if len(y_decoder_indices) == 0:
            raise ValueError("Legacy entropy format cannot decode empty geometry.")

        y_order_decoder_indices = sort_indices(torch.tensor(y_decoder_indices, device=self.decode_device or torch.device('cuda')))

        with open(file_name +"_F/" +eval_name_str +  '_F.bin', 'rb') as fin:
            stream = np.frombuffer(fin.read(), dtype=np.uint8)

        if self.spatial_shape is None:
            raise ValueError('Decode requires the fixed spatial_shape from the model configuration.')
        spatial_shape = self.spatial_shape
        decoded_coords = y_order_decoder_indices.cpu().numpy()
        if (decoded_coords < 0).any() or (decoded_coords >= np.array(spatial_shape)).any():
            raise ValueError('Decoded coordinates are outside the configured spatial shape.')
        feats = decode_quantized(stream.tobytes(), spatial_shape, self.legacy_shape_header)

        y_order_decoder_features = torch.tensor(feats, dtype=torch.float32, device=self.decode_device or torch.device('cuda'))

        batch_idx = torch.zeros(y_order_decoder_indices.shape[0], device=y_order_decoder_indices.device).unsqueeze(0).transpose(1, 0)

        if y_order_decoder_features.shape[0] != y_order_decoder_indices.shape[0]:
            raise ValueError("Decoded geometry/attribute point counts differ.")

        y_order_decoder_features = y_order_decoder_features / self.quantization_parameter

        output_indices = torch.cat((batch_idx, y_order_decoder_indices), dim=1)

        output_sp_tensor = spconv.SparseConvTensor(features=y_order_decoder_features, indices=output_indices.int(),
                                                   spatial_shape=spatial_shape, batch_size=1)

        if self.del_ply == True:
            ply_file_name_input = self.save_path + eval_name_str + "/" + eval_name_str + "_input.ply"
            ply_file_name_output = self.save_path + eval_name_str + "/" + eval_name_str + "_output.ply"
            os.system('rm ' + ply_file_name_input)
            os.system('rm ' + ply_file_name_output)

        return output_sp_tensor


class VirConvL8x(nn.Module):

    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):

        super().__init__()
        self.model_cfg = model_cfg
        self.x_trans_train = X_TRANS()

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features = model_cfg.OUT_FEATURES
        self.layer_discard_rate = model_cfg.LAYER_DISCARD_RATE

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        codec_shapes = feature_shapes(self.sparse_shape)

        self.vir_conv1 = NRConvBlock(input_channels, num_filters[0], stride=1, indice_key='vir1')
        self.vir_conv2 = NRConvBlock(num_filters[0], num_filters[1], stride=2, indice_key='vir2')
        self.vir_conv3 = NRConvBlock(num_filters[1], num_filters[2], stride=2, indice_key='vir3')
        self.vir_conv4 = NRConvBlock(num_filters[2], num_filters[3], stride=2, padding=(0, 1, 1), indice_key='vir4')

        '''Channel DownSampling'''
        channel_input_down = 64
        channel_inter_down = 16
        channel_output_down = 4
        channel_output_down_2 = 2

        '''Channel UpSampling'''
        channel_input_up_2 = 2
        channel_input_up = 4
        channel_inter_up = 16
        channel_output_up = 64

        '''MaxPooling'''
        kernel_size = (3, 3, 3)
        padding = 0
        stride = 3
        print_maxpooling_results = True
        max_output_num = 0

        '''XYZ Embedding'''
        self.num_channels = 4
        self.temperature = 10000
        self.scale = 0 * math.pi
        self.guass_scale = 1
        self.type_of_embedding = "sin"

        '''Masked Transformer'''
        enc_dim = 4
        enc_ffn_dim = 32
        enc_nhead = 4
        enc_dropout = 0
        enc_activation = "relu"
        enc_nlayers = 4
        norm = None
        mask_radius = [3, 4, 5, 6]
        self.type_of_transformer = "masked"

        '''Slope Threshold'''

        kkk = 0.02
        self.slope_channel_1 = kkk
        self.slope_channel_2 = kkk
        self.step_channel_1 = 0.1
        self.step_channel_2 = 0.1
        self.range_b_channel_1_bottom = 100
        self.range_b_channel_2_bottom = 100
        self.range_b_channel_1_top = 1000
        self.range_b_channel_2_top = 1000

        rrr = 9
        self.radius_threshold_channel_1 = rrr
        self.radius_threshold_channel_2 = rrr
        self.draw_curve_channel_1 = False
        self.draw_curve_channel_2 = False
        self.print_threshold_results = True

        '''Entropy Encoder & EntrSopy Decoder'''
        quantization_parameter = 1
        interval_feat = 1.0
        step_bev = 1.0
        step_roi3 = 0.1
        step_roi4 = 0.1
        _lambda_feat = 0.0

        save_path_bev = str(Path(os.environ.get("FC3DOD_STREAM_ROOT", str(Path(__file__).resolve().parents[3] / "output/streams"))) / "affc" / f"r={rrr}" / f"k={kkk}" / "bev") + "/"
        save_path_roi3 = str(Path(os.environ.get("FC3DOD_STREAM_ROOT", str(Path(__file__).resolve().parents[3] / "output/streams"))) / "affc" / f"r={rrr}" / f"k={kkk}" / "roi3") + "/"
        save_path_roi4 = str(Path(os.environ.get("FC3DOD_STREAM_ROOT", str(Path(__file__).resolve().parents[3] / "output/streams"))) / "affc" / f"r={rrr}" / f"k={kkk}" / "roi4") + "/"
        print_enc_time = True
        del_ply = False

        self.vir_conv5 = NRConvBlock(channel_input_down, channel_inter_down, stride=1, indice_key='vir5')
        self.vir_conv6 = NRConvBlock(channel_inter_down, channel_output_down, stride=1, indice_key='vir6')

        self.maxpooling_layer = MaxPooling(kernel_size=kernel_size, padding=padding, stride=stride,
                                           print_maxpooling_results=print_maxpooling_results,
                                           max_output_num=max_output_num)

        self.mask_transformer_enc = MaskedTransformer(enc_nlayers=enc_nlayers, enc_dim=enc_dim,
                                                      enc_ffn_dim=enc_ffn_dim, enc_nhead=enc_nhead,
                                                      enc_dropout=enc_dropout, enc_activation=enc_activation,
                                                      norm=norm, mask_radius=mask_radius)

        self.entropy_bev = EntropyEncDec(quantization_parameter=quantization_parameter, interval=interval_feat,
                                         step_size=step_bev, _lambda=_lambda_feat, save_path=save_path_bev,
                                         print_enc_time=print_enc_time, del_ply=del_ply, spatial_shape=codec_shapes["F4"])

        self.entropy_roi3 = EntropyEncDec(quantization_parameter=quantization_parameter, interval=interval_feat,
                                          step_size=step_roi3, _lambda=_lambda_feat, save_path=save_path_roi3,
                                          print_enc_time=print_enc_time, del_ply=del_ply, spatial_shape=codec_shapes["F3"])

        self.entropy_roi4 = EntropyEncDec(quantization_parameter=quantization_parameter, interval=interval_feat,
                                          step_size=step_roi4, _lambda=_lambda_feat, save_path=save_path_roi4,
                                          print_enc_time=print_enc_time, del_ply=del_ply, spatial_shape=codec_shapes["F4"])

        self.vir_conv7 = NRConvBlock(channel_input_up, channel_inter_up, stride=1, indice_key='vir7')
        self.vir_conv8 = NRConvBlock(channel_inter_up, channel_output_up, stride=1, indice_key='vir8')

        self.vir_conv13 = NRConvBlock(channel_input_down, channel_inter_down, stride=1, indice_key='vir13')
        self.vir_conv14 = NRConvBlock(channel_inter_down, channel_output_down_2, stride=1, indice_key='vir14')

        self.vir_conv15 = NRConvBlock(channel_input_up_2, channel_inter_up, stride=1, indice_key='vir15')
        self.vir_conv16 = NRConvBlock(channel_inter_up, channel_output_up, stride=1, indice_key='vir16')

        self.vir_conv9 = NRConvBlock(channel_input_down, channel_inter_down, stride=1, indice_key='vir9')
        self.vir_conv10 = NRConvBlock(channel_inter_down, channel_output_down_2, stride=1, indice_key='vir10')

        self.vir_conv11 = NRConvBlock(channel_input_up_2, channel_inter_up, stride=1, indice_key='vir11')
        self.vir_conv12 = NRConvBlock(channel_inter_up, channel_output_up, stride=1, indice_key='vir12')

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(

            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )

        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features

    def decompose_tensor(self, tensor, i, batch_size):
        """
        decompose a sparse tensor by the given transformation index.
        """
        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict, eval_name_str):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """

        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        batch_size = batch_dict['batch_size']
        if batch_size != 1:
            raise ValueError('The released feature-compression backbones support batch_size=1.')

        for i in range(rot_num):
            if i == 0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)
            newvoxel_features, newvoxel_coords = batch_dict['voxel_features' + rot_num_id], batch_dict[
                'voxel_coords' + rot_num_id]

            newvoxel_features[:, 4:7] = 0

            newinput_sp_tensor = spconv.SparseConvTensor(
                features=newvoxel_features,
                indices=newvoxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )
            if self.training:
                layer_voxel_discard(newinput_sp_tensor, 0)

            calib = batch_dict['calib']
            batch_size = batch_dict['batch_size']
            if 'aug_param' in batch_dict:
                trans_param = batch_dict['aug_param']
            else:
                trans_param = None
            if 'transform_param' in batch_dict:
                trans_param = batch_dict['transform_param'][:, i, :]

            newx_conv1 = self.vir_conv1(newinput_sp_tensor, batch_size, calib, 1, self.x_trans_train, trans_param)

            if self.training:
                layer_voxel_discard(newx_conv1, self.layer_discard_rate)

            newx_conv2 = self.vir_conv2(newx_conv1, batch_size, calib, 2, self.x_trans_train, trans_param)

            if self.training:
                layer_voxel_discard(newx_conv2, self.layer_discard_rate)

            newx_conv3 = self.vir_conv3(newx_conv2, batch_size, calib, 4, self.x_trans_train, trans_param)

            if self.training:
                layer_voxel_discard(newx_conv3, self.layer_discard_rate)

            newx_conv4 = self.vir_conv4(newx_conv3, batch_size, calib, 8, self.x_trans_train, trans_param)

            newx_conv4_maxpooling = self.maxpooling_layer(newx_conv4)

            newx_conv5 = self.vir_conv5(newx_conv4_maxpooling, batch_size, calib, 8, self.x_trans_train, trans_param)
            newx_conv6_threshold = self.vir_conv6(newx_conv5, batch_size, calib, 8, self.x_trans_train, trans_param)

            src, xyz_tensor, pos = sin_xyz_embedding(newx_conv6_threshold, self.num_channels, self.temperature,
                                                     self.scale)

            transformer_output = self.mask_transformer_enc(src, xyz_tensor)

            shape_variable = transformer_output.shape
            transformer_output = transformer_output.reshape(shape_variable[0], shape_variable[2])
            newx_conv6_threshold = newx_conv6_threshold.replace_feature(transformer_output)

            newx_conv6 = slope_threshold(self.slope_channel_1, self.slope_channel_2,
                                         self.step_channel_1, self.step_channel_2,
                                         self.range_b_channel_1_top, self.range_b_channel_2_top,
                                         self.range_b_channel_1_bottom, self.range_b_channel_2_bottom,
                                         self.draw_curve_channel_1, self.draw_curve_channel_2,
                                         self.radius_threshold_channel_1, self.radius_threshold_channel_2,
                                         newx_conv6_threshold, self.print_threshold_results)

            newx_conv9 = self.vir_conv9(newx_conv4, batch_size, calib, 8, self.x_trans_train, trans_param)
            newx_conv10 = self.vir_conv10(newx_conv9, batch_size, calib, 8, self.x_trans_train, trans_param)

            newx_conv13 = self.vir_conv13(newx_conv3, batch_size, calib, 8, self.x_trans_train, trans_param)
            newx_conv14 = self.vir_conv14(newx_conv13, batch_size, calib, 8, self.x_trans_train, trans_param)

            geo_TH_6 = newx_conv6.indices.clone()
            geo_TH_cube_6 = cube_up_sampling(geo_TH_6, 1, 1)
            geo_TH_cube_6 = geo_TH_cube_6[:, 1:]
            geo_TH_cube_6 = torch.unique(geo_TH_cube_6, dim=0, return_inverse=False)

            geo_newx_conv10 = newx_conv10.indices[:, 1:].int()

            equal_indices_10 = compute_big_equal(geo_newx_conv10, geo_TH_cube_6, 3000)

            newx_conv10.indices = newx_conv10.indices[equal_indices_10]
            newx_conv10 = newx_conv10.replace_feature(newx_conv10.features[equal_indices_10])

            newx_conv6_ind = newx_conv6.indices.clone()
            newx_conv6_feat = newx_conv6.features.clone()
            newx_conv6_sps = newx_conv6.spatial_shape
            newx_conv3_sps = newx_conv14.spatial_shape

            newx_conv6.spatial_shape = newx_conv3_sps
            ratio_z = newx_conv3_sps[0] / newx_conv6_sps[0]
            ratio_y = newx_conv3_sps[1] / newx_conv6_sps[1]
            ratio_x = newx_conv3_sps[2] / newx_conv6_sps[2]
            newx_conv6.indices[:, 1] = torch.round(newx_conv6.indices[:, 1] * ratio_z)
            newx_conv6.indices[:, 2] = torch.round(newx_conv6.indices[:, 2] * ratio_y)
            newx_conv6.indices[:, 3] = torch.round(newx_conv6.indices[:, 3] * ratio_x)

            geo_TH_6 = newx_conv6.indices[:, 1:].clone()

            geo_newx_conv14 = newx_conv14.indices[:, 1:].int()

            min_indices_14 = compute_big_dist(geo_newx_conv14, geo_TH_6, 6, 3000)
            min_indices_14 = torch.unique(min_indices_14, dim=0, return_inverse=False)

            newx_conv14.indices = newx_conv14.indices[min_indices_14]
            newx_conv14 = newx_conv14.replace_feature(newx_conv14.features[min_indices_14])

            newx_conv6.indices = newx_conv6_ind
            newx_conv6 = newx_conv6.replace_feature(newx_conv6_feat)
            newx_conv6.spatial_shape = newx_conv6_sps

            if self.training:
                newx_conv6_dec = newx_conv6
                newx_conv10_dec = newx_conv10
                newx_conv14_dec = newx_conv14
            else:
                self.entropy_bev.encode(newx_conv6, eval_name_str)

                self.entropy_roi4.encode(newx_conv10, eval_name_str)

                self.entropy_roi3.encode(newx_conv14, eval_name_str)

                newx_conv6_dec = self.entropy_bev.decode(eval_name_str)

                newx_conv10_dec = self.entropy_roi4.decode(eval_name_str)

                newx_conv14_dec = self.entropy_roi3.decode(eval_name_str)
                newx_conv14_dec.spatial_shape = newx_conv14.spatial_shape

            newx_conv11 = self.vir_conv11(newx_conv10_dec, batch_size, calib, 8, self.x_trans_train, trans_param)
            newx_conv12 = self.vir_conv12(newx_conv11, batch_size, calib, 8, self.x_trans_train, trans_param)

            newx_conv15 = self.vir_conv15(newx_conv14_dec, batch_size, calib, 8, self.x_trans_train, trans_param)
            newx_conv16 = self.vir_conv16(newx_conv15, batch_size, calib, 8, self.x_trans_train, trans_param)

            newx_conv7 = self.vir_conv7(newx_conv6_dec, batch_size, calib, 8, self.x_trans_train, trans_param)
            newx_conv8 = self.vir_conv8(newx_conv7, batch_size, calib, 8, self.x_trans_train, trans_param)

            out = self.conv_out(newx_conv8)

            Empt_conv1_feat = torch.zeros(newx_conv1.features.shape)
            Empt_conv1_coor = torch.zeros(newx_conv1.indices.shape)
            Empt_conv1_sps = newx_conv1.spatial_shape
            Empt_conv1 = spconv.SparseConvTensor(features=Empt_conv1_feat, indices=Empt_conv1_coor.int(),
                                                 spatial_shape=Empt_conv1_sps, batch_size=1)

            Empt_conv2_feat = torch.zeros(newx_conv2.features.shape)
            Empt_conv2_coor = torch.zeros(newx_conv2.indices.shape)
            Empt_conv2_sps = newx_conv2.spatial_shape
            Empt_conv2 = spconv.SparseConvTensor(features=Empt_conv2_feat, indices=Empt_conv2_coor.int(),
                                                 spatial_shape=Empt_conv2_sps, batch_size=1)

            batch_dict.update({
                'encoded_spconv_tensor' + rot_num_id: out,
                'encoded_spconv_tensor_stride' + rot_num_id: 8,
            })
            batch_dict.update({
                'encoded_spconv_tensor_stride' + rot_num_id: 8
            })
            batch_dict.update({
                'multi_scale_3d_features' + rot_num_id: {
                    'x_conv1': Empt_conv1,
                    'x_conv2': Empt_conv2,
                    'x_conv3': newx_conv16,
                    'x_conv4': newx_conv12,
                },
                'multi_scale_3d_strides' + rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })

        return batch_dict

