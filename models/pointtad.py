# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modified from RTD-Net (https://github.com/MCG-NJU/RTD-Action)

PointTAD Model and Criterion Class.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from util import ops, misc
from .matcher import build_matcher
from .decoder import build_decoder
from .backbone import I3D_BackBone

class PointTAD(nn.Module):
    """PointTAD for multi-label TAD.

    Args:
        backbone (obj): Object of backbone.
        decoder (obj): Object of decoder layer.
        hidden_dim (int): Number of the hidden dimension in decoder layers.
        num_classes (int): Number of action classes.
        window_size (int): Number of input features within a video clip.
        num_queries (int): Number of action queries, the maximal number of proposals
        num_querypoints (int): Number of learnable query points.
        aux_loss (bool): True if auxiliary decoding losses
            (loss at each decoder layer) are to be used. Default: False.
    """
    def __init__(self,
                backbone,
                decoder,
                hidden_dim,
                num_classes,
                window_size,
                num_queries,
                num_querypoints,
                aux_loss=False):

        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.window_size = window_size

        # Build Backbone
        self.backbone = backbone
        self.avg_pool_5c = nn.AvgPool3d(kernel_size=[1, 6, 6], stride=(1, 1, 1))
        self.avg_pool_4f = nn.AvgPool3d(kernel_size=[1, 12, 12], stride=(1, 1, 1))
        self.input_proj_5c = nn.Conv2d(backbone.outdim_5c, hidden_dim, kernel_size=1)
        self.input_proj_4f = nn.Conv2d(backbone.outdim_4f, hidden_dim, kernel_size=1)

        # Build Queries.
        self.init_proposal_features = nn.Embedding(self.num_queries, self.hidden_dim)
        self.init_proposal_points = nn.Embedding(self.num_queries, num_querypoints)
        nn.init.constant_(self.init_proposal_points.weight, 0.5)
        
        # Build Decoder
        self.decoder = decoder
        
        self.linear_pred = nn.Sequential(
            nn.Conv1d(hidden_dim,
                      hidden_dim//2,
                      kernel_size=3,
                      padding=1),           
            nn.Dropout(),
            nn.Conv1d(hidden_dim//2,
                      num_classes,
                      kernel_size=1))
        self.aux_loss = aux_loss

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """Forward process of PointTAD.

        Args:
            x (torch.Tensor): RGB frames of samples.
                Shape：(batch_size, 3, T, H, W).

        Returns:
            out (dict): A dict with the following elements:
                'pred_logits': the classification logits (including no-object) for all queries.
                    Shape: (batch_size, num_queries, (num_classes + 1)).
                'pred_segments': The normalized segments coordinates for all queries, represented as
                    (center_x, center_y, height, width). These values are normalized in [0, 1],
                    relative to the size of each individual image (disregarding possible padding).
                    See PostProcess for information on how to retrieve the unnormalized segments .
                    Shape: (batch_size, num_queries, 2).
                'logits': Dense segmentation scores of action predictions, which measure the action likeliness at each frame.
                    Shape: (batch_size, T, 1).
                'aux_outputs': Optional, only returned when auxilary losses are activated. It is a list of
                    dictionaries containing the two above keys for each decoder layer.
        """
        
        feat_dict = self.backbone(x)
        bs = feat_dict['Mixed_5c'].shape[0]
        samples = self.avg_pool_5c(feat_dict['Mixed_5c']).view(bs, -1, self.window_size).permute(0,2,1) 
        samples_4f = self.avg_pool_4f(feat_dict['Mixed_4f']).view(bs, -1, self.window_size).permute(0,2,1)

        features = self.input_proj_5c(samples.flatten(0,1).unsqueeze(-1).unsqueeze(-1)).view(bs,-1,self.hidden_dim)
        features += self.input_proj_4f(samples_4f.flatten(0,1).unsqueeze(-1).unsqueeze(-1)).view(bs,-1,self.hidden_dim)
        logits = self.linear_pred(features.permute(0,2,1)).permute(0,2,1)

        proposal_points = self.init_proposal_points.weight.clone()
        proposal_points = proposal_points.unsqueeze(0).repeat(bs, 1, 1)

        outputs_class, outputs_coord, hs = self.decoder(features, proposal_points, self.init_proposal_features.weight)
        
        out = {'pred_logits': outputs_class[-1], 'pred_segments': outputs_coord[-1], 'logits':logits} 

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class,outputs_coord, logits)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, logits):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
            'pred_logits': a,
            'pred_segments': b,
            'logits': d
        } for a, b, d in zip(outputs_class[:-1], outputs_coord[:-1], logits)]

class SetCriterion(nn.Module):
    """This class computes the loss for PointTAD.

    The process happens in two steps:
    1) We compute hungarian assignment between
        ground truth segments and the outputs of the model
    2) We supervise each pair of matched
        ground-truth / prediction (supervise class and segments)

    Args:
        num_classes (int): Number of action categories,
            omitting the special no-action category.
        window_size (int): Number of input features,
            we use it to normalize predicted pseudo segments.
        matcher (obj): Module able to compute a matching
            between targets and proposals.
        weight_dict (dict): Dict containing as key the names of the losses
            and as values their relative weight.
        eos_coef (float): Relative classification weight
            applied to the no-object category
        losses (list): List of all the losses to be applied.
            See get_loss for list of available losses.
    """
    def __init__(self, num_classes, window_size, matcher, weight_dict, eos_coef, losses):

        super().__init__()
        self.num_classes = num_classes
        self.window_size = window_size
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_segments, log=True):
        """Classification loss (NLL) targets dicts must contain the key
        'labels' containing a tensor of dim [nb_target_segments].

        Args:
            outputs (dict): Dict of PointTAD outputs.
            targets (list): A list of size batch_size. Each element is a dict composed of:
                'labels': Labels of groundtruth instances (0: action).
                'segments': Relative temporal ratio of groundtruth instances.
                'video_id': ID of the video sample.
            indices (list): A list of size batch_size.
                Each element is composed of two tensors,
                the first index_i is the indices of the selected predictions (in order),
                the second index_j is the indices of the corresponding selected targets (in order).
                For each batch element,
                it holds: len(index_i) = len(index_j) = min(num_queries, num_target_segments)

        Returns:
            losses (dict): Dict of losses.
        """
        assert 'pred_logits' in outputs

        src_logits = outputs['pred_logits']
        bs = len(src_logits)
        if indices is None:
            loss_ce = 0 * src_logits.sum()
            losses = {'loss_ce': loss_ce}
            if log:
                losses['class_error'] = loss_ce
            return losses

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # print(src_logits.shape, idx)
            losses['class_error'] = 100 - misc.accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_segments):
        """Compute the cardinality error, ie the absolute error in the number
        of predicted non-empty segments This is not really a loss, it is intended
        for logging purposes only.

        It doesn't propagate gradients
        """
        if indices is None:
            losses = {'cardinality_error': torch.tensor(0.).to(outputs['pred_logits'].device)}
            return losses

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets],
                                      device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_segments(self, outputs, targets, indices, num_segments):
        """Compute the losses related to the bounding segments, the L1 regression
        loss and the GIoU loss targets dicts must contain the key "segments"
        containing a tensor of dim [nb_target_segments, 2] The target segments are
        expected in format (start, end), normalized by the image
        size.

        Args:
            outputs (dict): Dict of PointTAD outputs.
            targets (list): A list of size batch_size. Each element is a dict composed of:
                'labels': Labels of groundtruth instances (0: action).
                'segments': Relative temporal ratio of groundtruth instances.
                'video_id': ID of the video sample.
            indices (list): A list of size batch_size.
                Each element is composed of two tensors,
                the first index_i is the indices of the selected predictions (in order),
                the second index_j is the indices of the corresponding selected targets (in order).
                For each batch element,
                it holds: len(index_i) = len(index_j) = min(num_queries, num_target_segments)
            num_segments (int): Number of positive samples.

        Returns:
            losses (dict): Dict of losses.
        """
        assert 'pred_segments' in outputs
        if indices is None:
            loss_seg = 0 * outputs['pred_segments'].sum()
            return {'loss_seg': loss_seg, 'loss_giou': loss_seg}

        idx = self._get_src_permutation_idx(indices)
        src_segments = outputs['pred_segments'][idx]
        target_segments = torch.cat(
            [t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_seg = F.l1_loss(src_segments,
                              ops.prop_se_to_cl(target_segments),
                              reduction='none')

        losses = {}
        losses['loss_seg'] = loss_seg.sum() / num_segments

        loss_giou = 1 - torch.diag(
            ops.segment_tiou(ops.prop_cl_to_se(src_segments),
                                         target_segments))
        losses['loss_giou'] = loss_giou.sum() / num_segments
        return losses

    def loss_dense(self, outputs, targets, indices, num_segments):
        assert 'logits' in outputs
        groundtruth = []
        bs, ws, c = outputs['logits'].shape
        for tgt in targets:
            groundtruth.append(tgt['dense_gt'])
        groundtruth = torch.stack(groundtruth, dim=0)
        assert outputs['logits'].shape == groundtruth.shape
        loss_f = F.binary_cross_entropy_with_logits(outputs['logits'], groundtruth, reduction='mean')
        return {'loss_dense':loss_f}

    def _get_src_permutation_idx(self, indices):
        '''
        Args:
            indices (list): A list of size batch_size.
                Each element is composed of two tensors,
                the first index_i is the indices of the selected predictions (in order),
                the second index_j is the indices of the corresponding selected targets (in order).
                For each batch element,
                it holds: len(index_i) = len(index_j) = min(num_queries, num_target_segments)

        Returns:
            A tuple composed of two tensors:
                the first is batch idx,
                the second is sample_idx.
        '''
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_segments, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'segments': self.loss_segments,
            'dense': self.loss_dense
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_segments, **kwargs)

    def forward(self, outputs, targets):
        """Loss computation.

        Args:
            outputs (dict): Dict of PointTAD outputs, which are tensors.
            targets (dict): List of dicts, such that len(targets) == batch_size.
                The expected keys in each dict depends on the losses applied.

        Returns:
            losses (dict): Dict of losses.
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        } 

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target segments accross all nodes, for normalization purposes
        num_segments = sum(len(t['labels']) for t in targets)
        num_segments = torch.as_tensor([num_segments],
                                    dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if misc.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_segments)
        num_segments = torch.clamp(num_segments / misc.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_segments))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs and 'iou' not in self.losses:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks' or loss == 'dense':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_segments, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the
    THUMOS14 dataset evaluation tool."""
    def __init__(self, args):
        super().__init__()
        self.window_size = args.window_size
        self.interval = args.interval
        self.num_proposals = args.num_queries
        self.num_classes = args.num_classes

    @torch.no_grad()
    def forward(self, outputs, num_frames, base):
        """ Perform the computation
        Parameters:
            outputs (dict): Dict of PointTAD outputs.
            num_frames (torch.Tensor): Number of frames in samples.
                Shape: (batch_size, )
            base (torch.Tensor): Index of the base/first frame in samples.
                Shape: (batch_size, )
        """
        out_logits, out_seg = outputs['pred_logits'], outputs['pred_segments']
        logits = outputs['logits'].sigmoid()
        assert len(out_logits) == len(num_frames)
        num_frames = num_frames.reshape((len(out_logits), 1))
        bs = len(out_logits)

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        raw_segments = out_seg.clone()
        segments = ops.prop_cl_to_se(out_seg)
        segments = ops.prop_relative_to_absolute(segments, base, self.window_size, self.interval)

        results = [{
            'scores': s,
            'labels': l,
            'segments': b
        } for s, l, b, rb in zip(scores, labels, segments, raw_segments)]

        return results, logits


def build(args):
    device = torch.device(args.device)

    backbone = I3D_BackBone(model_path=args.model_path, in_channels=3)

    decoder = build_decoder(args)

    model = PointTAD(backbone=backbone,
                decoder=decoder,
                hidden_dim=args.hidden_dim,
                num_classes=args.num_classes,
                window_size=args.window_size,
                num_queries=args.num_queries,
                num_querypoints=args.num_querypoints,
                aux_loss=args.aux_loss)
    
    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': 1,
        'loss_seg': args.seg_loss_coef,
        'loss_dense': args.dense_loss_coef
    }
    weight_dict['loss_giou'] = args.giou_loss_coef

    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v
                 for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'segments', 'cardinality','dense']

    criterion = SetCriterion(num_classes=args.num_classes,
                             window_size=args.window_size,
                             matcher=matcher,
                             weight_dict=weight_dict,
                             eos_coef=args.eos_coef,
                             losses=losses)
    criterion.to(device)
    postprocessors = {'results': PostProcess(args)}

    return model, criterion, postprocessors

