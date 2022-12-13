import json

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import util.misc as utils

import torch
from tqdm import tqdm
import math
from terminaltables import *

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / (segments_union+1e-4)
    return tIoU

def wrapper_compute_average_precision(ground_truth, prediction, num_classes, tiou_thresholds):
    """Computes average precision for each class in the subset.
    """
    ap = np.zeros((len(tiou_thresholds), num_classes))
    for cidx in tqdm(range(num_classes)):
        gt_idx = ground_truth['label'] == cidx
        pred_idx = prediction['label'] == cidx
        ap[:,cidx] = compute_average_precision_detection(
            ground_truth.loc[gt_idx].reset_index(drop=True),
            prediction.loc[pred_idx].reset_index(drop=True),
            tiou_thresholds=tiou_thresholds)
    return ap

def multi_formatting(results, dataset):
    results_list = []
    if dataset == 'multithumos':
        frame_dict = load_json('datasets/multithumos_frames.json')
        annotations = load_json('datasets/multithumos.json')
    else:
        annotations = load_json('datasets/charades.json')
    for vid, info in results.items():
        if dataset == 'multithumos':
            num_frames = frame_dict[vid]
            fps = float(num_frames / annotations[vid]['duration'])
        else:
            fps = 12
            num_frames = int(annotations[vid]['duration']*fps)

        for preds_dict in info:
            scores = preds_dict['scores']
            segments = preds_dict['segments']
            labels = preds_dict['labels']

            segments = segments.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            for sample_idx in range(segments.shape[0]):
                results_list.append([
                    float(segments[sample_idx][0]/fps),
                    float(segments[sample_idx][1]/fps),
                    float(scores[sample_idx]), int(labels[sample_idx]), num_frames, vid
                ])
    results_list = np.stack(results_list)

    results_pd = pd.DataFrame(
        results_list,
        columns=['t-start', 't-end', 'score', 'label','video-frames', 'video-id'])
    return results_pd

def evaluate_mAP(preds, dataset, num_classes):
    """Evaluates a prediction file. For the detection task we measure the
    interpolated mean average precision to measure the performance of a
    method.
    """
    gt = load_json(f'datasets/{dataset}.json')
    gt_df = []
    for vid, anno in gt.items():
        if anno['subset'] == 'testing':
            for act in anno['actions']:
                if dataset == 'multithumos':
                    gt_df.append((vid, act[0]-1, act[1], act[2]))
                else:
                    gt_df.append((vid, act[0], act[1], act[2]))
    gt_df = pd.DataFrame(gt_df, columns=['video-id','label','t-start','t-end'])
    
    iou_range = np.arange(0.1, 1.0, 0.1)
    ap = wrapper_compute_average_precision(gt_df, preds, num_classes, iou_range)
    mAP = ap.mean(axis=1)

    display_title = f"[RESULTS] Detection Performance on {dataset.capitalize()}"
    display_data = [["IoU thresh"], ["mean AP"]]
    for i in range(len(iou_range)):
        display_data[0].append("{:.02f}".format(iou_range[i]))
        display_data[1].append("{:.04f}".format(mAP[i]))
    display_data[0].append('Average')
    display_data[1].append("{:.04f}".format(mAP.mean()))
    table = AsciiTable(display_data, display_title)
    table.justify_columns[-1] = 'right'
    table.inner_footing_row_border = True
    print(table.table)
    print('\n')

    return mAP.mean()


def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    ap = np.zeros(len(tiou_thresholds))

    for tidx in range(len(tiou_thresholds)):
        # Computing prec-rec
        this_tp = np.cumsum(tp[tidx,:]).astype(np.float)
        this_fp = np.cumsum(fp[tidx,:]).astype(np.float)
        rec = this_tp / npos
        prec = this_tp / (this_tp + this_fp)
        ap[tidx] = interpolated_prec_rec(prec, rec)

    return ap


class Evaluator(object):
    def __init__(self):
        self.predictions = []

    def update(self, vid, predictions, bs):
        self.predictions += [(vid, predictions, bs.item())]

    def get_result(self):
        return self.predictions

    def synchronize_between_processes(self):
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def summarize(self):
        results = {}
        count = {}
        for vid, p, bs in self.predictions:
            if vid in results.keys():
                if bs not in count[vid]:
                    results[vid].append(p)
                    count[vid].append(bs)
                else:
                    pass
            else:
                results[vid] = list()
                results[vid].append(p)
                count[vid] = list()
                count[vid].append(bs)
        self.predictions = []
        return results
