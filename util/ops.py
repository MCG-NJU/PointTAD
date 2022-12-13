"""
Helper functions for Temporal Action Detection.

"""
import torch
import torch.nn.functional as F
import numpy as np

def bilinear_sampling(value, sampling_locations):
    # values: N, T, N_heads, Dim 
    # sampling_locations: N, N_query, N_heads, N_level, N_points, 2
    N_, T, n_heads, D_ = value.shape
    # features torch.Size([4096, 512, 1, 100])
    # grid torch.Size([4096, 1, 32, 2])
    # grid element tensor([0., 0.], device='cuda:0')
    _, Lq_, n_heads, L_, P_, _ = sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1
    lid_ = 0
    H_ = 1
    W_ = T
    # N_, H_*W_, n_heads, D_ -> N_, H_*W_, n_heads*D_ -> N_, n_heads*D_, H_*W_ -> N_*n_heads, D_, H_, W_
    value_l_ = value.permute(0,3,2,1).reshape(N_*n_heads, 1, D_, H_, W_).repeat(1,Lq_,1,1,1)
    value_l_ = value_l_.flatten(0,1)
    # N_, Lq_, n_heads, P_, 2 -> N_, n_heads, Lq_, P_, 2 -> N_*n_heads, Lq_, P_, 2
    sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
    sampling_grid_l_ = sampling_grid_l_.flatten(0,1).unsqueeze(-3)
    #print(sampling_locations.shape, sampling_grid_l_.shape, value_l_.shape)
    # N_*n_heads, D_, Lq_, P_
    # return 
    sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                        mode='bilinear', padding_mode='zeros', align_corners=True)
    # (N_, Lq_, n_heads, L_, P_) -> (N_, n_heads, Lq_, L_, P_) -> (N_, n_heads, 1, Lq_, L_*P_)
    output = sampling_value_l_
    return output.contiguous()

def convert(points):
    # input : (N, nr_segments, num_querypoints) or (N * nr_segments, num_querypoints)
    # output: (N, nr_segments, 2) or (N * nr_segments, 2)
    if len(points.shape) == 3:
        N, nr_segments, num_querypoints = points.shape
        pred_segments = points.new_zeros((N, nr_segments, 2))
        pred_segments[:,:,0] = torch.min(points[:,:,:-num_querypoints//3], dim=-1)[0]
        pred_segments[:,:,1] = torch.max(points[:,:,:-num_querypoints//3], dim=-1)[0]
        return pred_segments
    elif len(points.shape) == 2:
        N_nr_segments, num_querypoints = points.shape
        pred_segments = points.new_zeros((N_nr_segments, 2))
        pred_segments[:,0] = torch.min(points[:,:-num_querypoints//3], dim=-1)[0]
        pred_segments[:,1] = torch.max(points[:,:-num_querypoints//3], dim=-1)[0]
        return pred_segments
    else:
        print("Wrong Input in Convert!")
        return 

# rewrite for temporal localization setting
def prop_cl_to_se(x):
    c, l = x.unbind(-1)
    b = [(c - 0.5 * l), (c + 0.5 * l)]
    return torch.stack(b, dim=-1).clamp(0, 1)


def prop_se_to_cl(x):
    s, e = x.unbind(-1)
    b = [(s + e) / 2, (e - s)]
    return torch.stack(b, dim=-1)


def prop_relative_to_absolute(x, base, window_size, interval):
    s, e = x.unbind(-1)
    num_samples = s.shape[1]
    base = base.unsqueeze(1).repeat(1, num_samples).cuda()
    b = [s * window_size * interval + base, e * window_size * interval + base]
    return torch.stack(b, dim=-1)


def segment_tiou(seg_a, seg_b):
    # gt: [N, 2], detections: [M, 2]
    N = seg_a.shape[0]
    M = seg_b.shape[0]

    tiou = torch.zeros((N, M)).to(seg_a.device)
    for i in range(N):
        inter_max_xy = torch.min(seg_a[i, 1], seg_b[:, 1])
        inter_min_xy = torch.max(seg_a[i, 0], seg_b[:, 0])

        inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)

        # calculate union
        union = (seg_b[:, 1] - seg_b[:, 0]) + (seg_a[i, 1] -
                                               seg_a[i, 0]) - inter

        tiou[i, :] = inter / (union+1e-8)

    return tiou  # (N, M)
