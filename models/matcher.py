# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
PointTAD Matcher Class for Label Assignment.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.ops import (segment_tiou,
                          prop_cl_to_se, prop_se_to_cl)


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the
    predictions of the network. For efficiency reasons, the targets don't
    include the no_object. Because of this, in general, there are more
    predictions than targets.

    In Optimal bipartite matching, we do a 1-to-1 matching
    of the best predictions, while the others are un- matched
    (and thus treated as non-objects).

    Args:
        cost_class (float): Relative weight of the classification error
            in the matching cost
        cost_seg (float): Relative weight of the L1 error of the pred segments
            coordinates in the matching cost
        cost_giou (float): Relative weight of the giou loss
            of the pred segments in the matching cost
    """
    def __init__(self,
                 cost_class,
                 cost_seg,
                 cost_giou):
        super().__init__()
        self.cost_class = cost_class
        self.cost_seg = cost_seg
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_seg != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching.

        Args:
            outputs (dict): A dict that contains at least these entries:
                "pred_logits": Tensor of classification logits.
                    Shape: (batch_size, num_queries, num_classes)
                "pred_segments": Tensor of the predicted segment coordinates.
                    Shape: (batch_size, num_queries, 2)

            targets: (list): A list of targets (len(targets) = batch_size),
            each target is a dict containing:
                "labels": Tensor of class labels.
                    Shape: (num_target_segments, ) (num_target_segments is the number of
                    ground-truth objects in the sample)
                "segments": Tensor of target segment coordinates.
                    Shape: (num_target_segments, 2)

        Returns:
            indices (list): A list of size batch_size.
                Each element is composed of two tensors,
                the first index_i is the indices of the selected predictions (in order),
                the second index_j is the indices of the corresponding selected targets (in order).
                For each batch element,
                it holds: len(index_i) = len(index_j) = min(num_queries, num_target_segments)
        """

        bs, num_queries = outputs['pred_logits'].shape[:2]

        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  
        out_seg = outputs['pred_segments'].flatten(0, 1)  

        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_seg = torch.cat([v['segments'] for v in targets])

        if len(tgt_seg) == 0:
            return None
 
        cost_class = -out_prob[:, tgt_ids]
        cost_seg = torch.cdist(out_seg, prop_se_to_cl(tgt_seg), p=1)
        cost_giou = -segment_tiou(prop_cl_to_se(out_seg), tgt_seg)

        C = (self.cost_seg * cost_seg + self.cost_class * cost_class +
             self.cost_giou * cost_giou)
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v['segments']) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]

        return [(torch.as_tensor(i, dtype=torch.int64),
                    torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_seg=args.set_cost_seg,
                            cost_giou=args.set_cost_giou)
