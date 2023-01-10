#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
PointTAD Decoding Layer.

Modified from Sparse R-CNN Transformer class:
    * add parallel adaptive mixing in ParallelMixing()
    * query point representation to replace query segments
"""

import torch
from torch import nn
import numpy as np
import copy
import math
from typing import List
from util.ops import prop_cl_to_se, prop_se_to_cl, convert, bilinear_sampling
import torch.nn.functional as F

class PointtadDecoderHead(nn.Module):

    def __init__(self, num_classes, window_size, hidden_dim, dim_feedforward, num_querypoints, num_layers, nheads, dropout):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        dim_feedforward = dim_feedforward
        activation = 'relu'
        rcnn_head = PointtadDecoderLayer(hidden_dim, num_classes, num_querypoints, window_size, dim_feedforward, nheads, dropout, activation)        
        self.head_series = _get_clones(rcnn_head, num_layers)
        self.return_intermediate = True
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, features, init_proposal_points, init_features):
        
        inter_class_logits = []
        inter_pseudo_segments = []

        bs = features.shape[0]
        proposal_points = init_proposal_points 
        init_features = init_features[None].repeat(1, bs, 1)
        proposal_features = init_features.clone()
        for rcnn_head in self.head_series:
            class_logits, proposal_points, proposal_features = rcnn_head(features, proposal_points, proposal_features)
            pseudo_segments = convert(proposal_points)

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pseudo_segments.append(prop_se_to_cl(pseudo_segments))

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pseudo_segments), proposal_features.view(bs, -1, self.hidden_dim)

        return class_logits[None], prop_se_to_cl(pseudo_segments)[None], proposal_features.view(bs, -1, self.hidden_dim)


class PointtadDecoderLayer(nn.Module):

    def __init__(self, d_model, num_classes, num_querypoints, window_size, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu"):
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size
        self.num_querypoints = num_querypoints

        # Dynamic Decoding Modules.
        self.point_deformation = PointDeformation(d_model, window_size, num_querypoints)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_parallel_mix = ParallelMixing(d_model, window_size, num_querypoints)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # cls.
        num_cls = 1
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = 3
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.points_delta = nn.Linear(d_model, num_querypoints)

    def forward(self, features, proposal_points, pro_features):
        """
        :param features: (N, T, C)
        :param proposal_points: (N, nr_segments, num_querypoints)
        :param pro_features: (N, nr_segments, d_model)
        """

        N, nr_segments = proposal_points.shape[:2]
        d_model = self.d_model

        # Point-Level Local Deformation.
        point_features = self.point_deformation(pro_features, features, proposal_points)

        # Inter-proposal MHSA.
        pro_features = pro_features.view(N, nr_segments, self.d_model).permute(1, 0, 2)
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)
        
        # Instance-level Dynamic Convolution.
        pro_features = pro_features.view(nr_segments, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_segments, self.d_model)
        cls_features = self.inst_parallel_mix(pro_features, point_features)
        pro_features = pro_features + self.dropout2(cls_features)
        act_features = self.norm2(pro_features)

        act_features2 = self.linear2(self.dropout(self.activation(self.linear1(act_features))))
        act_features = act_features + self.dropout3(act_features2)
        act_features = self.norm3(act_features)
        
        fc_feature = act_features.transpose(0, 1).reshape(N * nr_segments, -1)
        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)

        class_logits = self.class_logits(cls_feature)
        points_deltas = self.points_delta(reg_feature)
        pred_points = self.apply_deltas(points_deltas, proposal_points.view(-1, self.num_querypoints)).view(N*nr_segments, -1)

        
        return class_logits.view(N, nr_segments, -1), pred_points.view(N, nr_segments, -1), act_features
    

    def apply_deltas(self, deltas, points):
        """
        Apply update `deltas` (dt) to `pred_points`.

        Args:
            deltas (Tensor): point deltas of shape (N, num_querypoints).
            pred_points (Tensor): updated query points of shape (N, num_querypoints)
        """
        points = points.to(deltas.dtype)

        segments_se = convert(points)
        length = segments_se[:,1] - segments_se[:,0]
        length = length.unsqueeze(-1).repeat(1,self.num_querypoints)

        dpoints = deltas * length * 0.5
        pred_points = points + dpoints
        
        return pred_points 

class PointDeformation(nn.Module):
    def __init__(self, d_model, window_size, num_querypoints):
        super().__init__()

        self.d_model = d_model
        self.window_size = window_size
        self.num_querypoints = num_querypoints
        self.num_subpoints = 4
        self.n_boundary_points = 4
        self.n_center_points = self.num_querypoints - self.n_boundary_points

        self.boundary_offsets = nn.Linear(d_model, self.num_subpoints)
        self.center_offsets = nn.Linear(d_model, self.num_subpoints)
        self.boundary_weights = nn.Linear(d_model, self.num_subpoints)
        self.center_weights = nn.Linear(d_model, self.num_subpoints)

        self._reset_parameters()

    def _reset_parameters(self):

        nn.init.constant_(self.boundary_weights.weight.data, 0.)
        nn.init.constant_(self.center_weights.weight.data, 0.)
        nn.init.constant_(self.boundary_weights.bias.data, 0.)
        nn.init.constant_(self.center_weights.bias.data, 0.)

        nn.init.constant_(self.boundary_offsets.weight.data, 0.)
        nn.init.constant_(self.center_offsets.weight.data, 0.)

        thetas = torch.arange(1, dtype=torch.float32) * (self.num_subpoints * math.pi / 1)
        grid_init = thetas.cos()[:, None]

        grid_init = grid_init.view(1, 1, 1, 1).repeat(
            1, 1, self.num_subpoints, 1)
        for i in range(self.num_subpoints):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.boundary_offsets.bias = nn.Parameter(grid_init.view(-1))
            self.center_offsets.bias = nn.Parameter(grid_init.view(-1))
        
    def forward(self, pro_features, features, sampled_points):

        N, nr_segments = sampled_points.shape[:2]

        bdy_offsets = self.boundary_offsets(pro_features).view(N, nr_segments, self.num_subpoints)
        ctr_offsets = self.center_offsets(pro_features).view(N, nr_segments, self.num_subpoints)

        boundary_weights = self.boundary_weights(pro_features).view(N*nr_segments, self.num_subpoints).softmax(-1)
        center_weights = self.center_weights(pro_features).view(N*nr_segments, self.num_subpoints).softmax(-1)
        boundary_weights = boundary_weights.transpose(1,0).unsqueeze(0).unsqueeze(-1).repeat(self.n_boundary_points,1,1,self.d_model)
        center_weights = center_weights.transpose(1,0).unsqueeze(0).unsqueeze(-1).repeat(self.n_center_points,1,1,self.d_model)
        point_weights = torch.cat([boundary_weights, center_weights], dim=0)

        boundary_points = sampled_points[:,:,:self.n_boundary_points].unsqueeze(-1).repeat(1,1,1,self.num_subpoints) + \
            bdy_offsets.unsqueeze(-2).repeat(1,1,self.n_boundary_points,1) / self.window_size
        center_points = sampled_points[:,:,self.n_boundary_points:].unsqueeze(-1).repeat(1,1,1,self.num_subpoints) + \
            ctr_offsets.unsqueeze(-2).repeat(1,1,self.n_center_points,1) / self.window_size
        boundary_points = boundary_points.flatten(2)
        center_points = center_points.flatten(2)
        sampled_points = torch.clamp(torch.cat([boundary_points, center_points],dim=-1), max=1., min=0.)

        grid = sampled_points.new_zeros((N,nr_segments,self.num_querypoints*self.num_subpoints,2))
        grid[:,:,:,0] = sampled_points

        features = bilinear_sampling(features.unsqueeze(2), grid.unsqueeze(2).unsqueeze(3))
        features = features.permute(3,0,2,1).flatten(1,2)
        features = features.view(self.num_querypoints, self.num_subpoints, -1, self.d_model)
        features = torch.sum(features * point_weights, dim=1)

        return features

class ParallelMixing(nn.Module):
    def __init__(self, hidden_dim, window_size, num_querypoints):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.num_querypoints = num_querypoints

        self.dim_dynamic = 64
        self.num_dynamic = 2
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.num_temp_dynamic = 1
        self.num_temp_params = self.num_querypoints * self.num_querypoints
        self.dynamic_temp_layer = nn.Linear(self.hidden_dim, self.num_temp_dynamic * self.num_temp_params)
        self.norm4 = nn.LayerNorm(self.num_querypoints)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        
        num_output = self.hidden_dim * self.num_querypoints
        self.out_layer = nn.Linear(num_output*2, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_segments, self.d_model)
        roi_features: (num_querypoints, N * nr_segments, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2) 

        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)
        temp_parameters = self.dynamic_temp_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)
        tparam = temp_parameters.view(-1, self.num_querypoints, self.num_querypoints)

        diff_features = features.permute(0,2,1)
        diff_features = torch.bmm(diff_features, tparam)
        diff_features = self.norm4(diff_features)
        diff_features = self.activation(diff_features)
        diff_features = diff_features.permute(0,2,1)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = torch.cat((features, diff_features),dim=-1)
        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def build_decoder(args):
    return PointtadDecoderHead(num_classes=args.num_classes, 
                window_size=args.window_size, 
                hidden_dim=args.hidden_dim, 
                dim_feedforward=args.dim_feedforward,
                num_querypoints=args.num_querypoints, 
                num_layers=args.dec_layers, 
                nheads=args.nheads, 
                dropout=args.dropout)