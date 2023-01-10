import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import h5py as h5
from tqdm import tqdm
import cv2
from torchvision import transforms
from .transforms import GroupResizeShorterSide, GroupCenterCrop, GroupRandomCrop, GroupRandomHorizontalFlip, GroupPhotoMetricDistortion, GroupRotate

IMAGE_CROP_SIZE = 192
IMAGE_RESIZE_SIZE = 256

training_transforms = transforms.Compose([
    GroupResizeShorterSide(IMAGE_RESIZE_SIZE),
    GroupRandomCrop(IMAGE_CROP_SIZE),
    GroupPhotoMetricDistortion(brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18,
                    p=0.5),
    GroupRotate(limit=(-45, 45), border_mode='reflect101', p=0.5),
    GroupRandomHorizontalFlip(0.5),
])

testing_transforms = transforms.Compose([
    GroupResizeShorterSide(IMAGE_RESIZE_SIZE),
    GroupCenterCrop(IMAGE_CROP_SIZE),
])


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


class VideoRecord:
    def __init__(self, vid, num_frames, locations, gt, fps, window_size, interval):
        self.id = vid
        self.locations = locations
        self.base = float(locations[0])
        self.window_size = window_size
        self.interval = interval
        self.locations_norm = [
            (i - self.base) / (self.window_size * self.interval)
            for i in locations
        ]
        self.locations_offset = [
            location - self.base for location in locations
        ]
        self.num_frames = num_frames

        self.gt = gt
        self.gt_norm = copy.deepcopy(gt)

        for i in self.gt_norm:
            i[0][0] = (i[0][0] - self.base) / (self.window_size *
                                               self.interval)
            i[0][1] = (i[0][1] - self.base) / (self.window_size *
                                               self.interval)
 
        self.gt_s_e_frames = [i[0] for i in self.gt_norm]
        self.fps = fps


class MultiTHUMOS(torch.utils.data.Dataset):
    def __init__(self, anno_file, frame_file, frame_folder, split, window_size, interval, num_classes, img_tensor, tensor_folder):
        self.window_size = window_size
        self.interval = interval
        self.anno_file = load_json(anno_file)
        self.frame_file = load_json(frame_file)
        self.tensor_folder = tensor_folder
        self.img_tensor = img_tensor

        video_list = self.anno_file.keys()
        self.num_classes = num_classes
        self.split = split
        video_seq = list(video_list)
        video_seq.sort()
        self.video_dict = {video_seq[i]: i for i in range(len(video_seq))}
        self.frame_folder = frame_folder
        overlap_dict = {'training':4, 'testing':1}

        self.video_list = []
        self.video_frame_pool = {}
        for vid in video_list:
            if self.anno_file[vid]['subset'] == self.split:
                num_frames = int(self.frame_file[vid])
                fps = num_frames / self.anno_file[vid]['duration']
                annotations = [
                    [int(fps*item[1]), int(fps*item[2])]
                    for item in self.anno_file[vid]['actions']
                ]
                labels = [
                    int(item[0])-1
                    for item in self.anno_file[vid]['actions']
                ]
                frames = np.array(range(1,num_frames-self.interval, self.interval)).reshape(-1,1)
                seq_len = len(frames)
                if seq_len <= self.window_size:
                    locations = np.zeros((self.window_size, 1))
                    locations[:seq_len, :] = frames
                    locations[seq_len:, :] = frames[-1] 
                    gt = [(annotations[idx], labels[idx])
                          for idx in range(len(annotations))]
                    if self.split=='training' and len(gt) == 0:
                        print(vid)
                    self.video_list.append(
                        VideoRecord(vid, num_frames, locations, gt, fps, self.window_size, self.interval))

                else:
                    overlap_ratio = overlap_dict[self.split]
                    stride = self.window_size // overlap_ratio
                    ws_starts = [
                        i * stride
                        for i in range((seq_len // self.window_size - 1) * overlap_ratio + 1)
                    ]
                    ws_starts.append(seq_len - self.window_size)

                    for ws in ws_starts:
                        locations = frames[ws:ws + self.window_size]
                        gt = []
                        for idx in range(len(annotations)):
                            anno = annotations[idx]
                            label = labels[idx]
                            if anno[0] >= locations[0] and anno[1] <= locations[-1]:
                                gt.append((anno, label))
                        if self.split == 'testing':
                            self.video_list.append(
                                VideoRecord(vid, num_frames, locations, gt, fps, self.window_size, self.interval))
                        elif len(gt) > 0:
                            self.video_list.append(
                                VideoRecord(vid, num_frames, locations, gt, fps, self.window_size, self.interval))

    def get_data(self, video: VideoRecord):
        vid = video.id
        num_frames = video.num_frames
        base = video.base

        block_idx = list(range(int(video.locations[0]), int(video.locations[-1])))

        if self.img_tensor:
            video_frames = torch.load(f'{self.tensor_folder}/{self.split}/{vid}')
            img_sliced = [video_frames[i-1,...] for i in block_idx]
            img_sliced = torch.stack(img_sliced, dim=0)
            img = img_sliced[-1,...]
            img_stacked = img_sliced.new_zeros(self.window_size*self.interval,img_sliced.shape[1], img_sliced.shape[2],img_sliced.shape[3])
            img_stacked[:len(img_sliced),...] = img_sliced[...]
            for i in range(self.window_size*self.interval - len(img_sliced)):
                img_stacked[i+len(img_sliced),...] = img[...]
            img_stacked = img_stacked.detach().cpu().numpy()
        else:
            img_stacked = []
            path_stacked = [os.path.join(f'{self.frame_folder}/{self.split}', vid, 'img_{:05d}.jpg'.format(i)) for i in block_idx]
            for i in range(len(path_stacked)):
                path = path_stacked[i]
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                img_stacked.append(img)
            img = img_stacked[-1]
            for i in range(self.window_size*self.interval - len(path_stacked)):
                img_stacked.append(img)
            assert len(img_stacked)==self.window_size*self.interval
            img_stacked = np.stack(img_stacked,axis=0)

        if self.split == 'training':
            input_data = training_transforms(img_stacked)
        else:
            input_data = testing_transforms(img_stacked)
        input_data = torch.from_numpy(input_data).permute(3,0,1,2)
        input_data = (input_data / 255.0) * 2.0 - 1.0
        locations = torch.Tensor([location for location in video.locations_offset])

        gt_s_e_frames = [(s, e, l) for ((s, e),l) in video.gt_norm]
        dense_gt = torch.zeros((self.window_size, self.num_classes)).cpu()

        labels = []
        segments = []
        for (start, end, label) in gt_s_e_frames:
            labels.append(int(label))
            segments.append((start, end))
            dense_gt[int(start*self.window_size):int(end*self.window_size)+1, int(label)] = 1
        
        targets = {
            'labels': torch.LongTensor(labels),
            'segments': torch.Tensor(segments),
            'video_id': torch.Tensor([self.video_dict[vid]]),
            'dense_gt':dense_gt
        }

        return vid, locations, input_data, targets, num_frames, base

    def __getitem__(self, idx):
        return self.get_data(self.video_list[idx])

    def __len__(self):
        return len(self.video_list)


class Charades(torch.utils.data.Dataset):
    def __init__(self, anno_file, frame_folder, split, window_size, interval, num_classes, img_tensor, tensor_folder):
        self.window_size = window_size
        self.interval = interval
        self.num_classes = num_classes
        self.split = split      
        self.frame_folder = frame_folder
        self.tensor_folder = tensor_folder
        self.img_tensor = img_tensor

        self.anno_file = load_json(anno_file)
        video_list = self.anno_file.keys()
        video_seq = list(video_list)
        video_seq.sort()
        self.video_dict = {video_seq[i]: i for i in range(len(video_seq))}

        overlap_dict = {'training':3, 'testing':1}

        self.video_list = []
        self.frame_pool = {}
        for vid in video_list:
            if self.split == self.anno_file[vid]['subset']:
                fps = 12
                num_frames = int(self.anno_file[vid]['duration']*fps)
                action_list = list(filter(lambda x: (x[1]<=x[2]), self.anno_file[vid]['actions']))
                annotations = [
                    [int(item[1]*fps), min(int(item[2]*fps), num_frames)]
                    for item in action_list
                ]
                labels = [
                    int(item[0])
                    for item in action_list
                ]
                frames = np.array(range(0, num_frames-self.interval, self.interval)).reshape(-1,1)
                seq_len = len(frames)
                if seq_len <= self.window_size:
                    locations = np.zeros((self.window_size, 1))
                    locations[:seq_len, :] = frames
                    locations[seq_len:, :] = frames[-1]
                    gt = [(annotations[idx], labels[idx])
                          for idx in range(len(annotations))]
                    self.video_list.append(
                        VideoRecord(vid, num_frames, 
                        locations, gt, fps, self.window_size, self.interval))

                else:
                    overlap_ratio = overlap_dict[self.split]
                    stride = self.window_size // overlap_ratio
                    ws_starts = [
                        i * stride
                        for i in range((seq_len // self.window_size - 1) * overlap_ratio + 1)]
                    ws_starts.append(seq_len - self.window_size)

                    for ws in ws_starts:
                        locations = frames[ws:ws + self.window_size]
                        gt = []
                        for idx in range(len(annotations)):
                            anno = annotations[idx]
                            label = labels[idx]
                            if anno[0] >= locations[0] and anno[1] <= locations[-1]:
                                gt.append((anno, label))

                        if self.split == 'testing':
                            self.video_list.append(
                                VideoRecord(vid, num_frames, 
                                locations, gt, fps, self.window_size, self.interval))
                        elif len(gt) > 0:
                            self.video_list.append(
                                VideoRecord(vid, num_frames, 
                                locations, gt, fps, self.window_size, self.interval))

    def get_data(self, video: VideoRecord):
        vid = video.id
        num_frames = video.num_frames
        base = video.base

        block_idx = list(range(int(video.locations[0]), int(video.locations[-1])))

        if self.img_tensor:
            video_frames = torch.load(f'{self.tensor_folder}/{vid}')
            img_sliced = [video_frames[i*2-1,...] for i in block_idx]
            img_sliced = torch.stack(img_sliced, dim=0)
            img = img_sliced[-1,...]
            img_stacked = img_sliced.new_zeros(self.window_size*self.interval,img_sliced.shape[1], img_sliced.shape[2],img_sliced.shape[3])
            img_stacked[:len(img_sliced),...] = img_sliced[...]
            for i in range(self.window_size*self.interval - len(img_sliced)):
                img_stacked[i+len(img_sliced),...] = img[...]
            img_stacked = img_stacked.detach().cpu().numpy()
        else:
            img_stacked = []
            path_stacked = [os.path.join(self.frame_folder, vid, '{}-{:06d}.jpg'.format(vid,i*2+1)) for i in block_idx]
            for i in range(len(path_stacked)):
                path = path_stacked[i]
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                assert img is not None, '{} {}'.format(i, path)
                img_stacked.append(img)
            img = img_stacked[-1]
            for i in range(self.window_size*self.interval - len(path_stacked)):
                img_stacked.append(img)
            assert len(img_stacked)==self.window_size*self.interval

            img_stacked = np.stack(img_stacked,axis=0)
            
        if self.split == 'training':
            input_data = training_transforms(img_stacked)
        else:
            input_data = testing_transforms(img_stacked)

        input_data = torch.from_numpy(input_data).permute(3,0,1,2)
        input_data = (input_data / 255.0) * 2.0 - 1.0
        
        locations = torch.Tensor([location for location in video.locations_offset])

        gt_s_e_frames = [(s, e, l) for ((s, e),l) in video.gt_norm]
        dense_gt = torch.zeros((self.window_size, self.num_classes))

        labels = []
        segments = []
        for (start, end, label) in gt_s_e_frames:
            labels.append(int(label))
            segments.append((start, end))
            dense_gt[int(start*self.window_size):int(end*self.window_size)+1, int(label)] = 1

        targets = {
            'labels': torch.LongTensor(labels),
            'segments': torch.Tensor(segments),
            'video_id': torch.Tensor([self.video_dict[vid]]),
            'dense_gt':dense_gt
        }

        return vid, locations, input_data, targets, num_frames, base

    def __getitem__(self, idx):
        return self.get_data(self.video_list[idx])

    def __len__(self):
        return len(self.video_list)


def collate_fn(batch):
    vid_name_list, target_list, num_frames_list, base_list = [[] for _ in range(4)]
    batch_size = len(batch)
    max_props_num = batch[0][1].shape[0]
    raw_frames = []
    locations = torch.zeros(batch_size, max_props_num, 1, dtype=torch.double)

    for i, sample in enumerate(batch):
        vid_name_list.append(sample[0])
        target_list.append(sample[3])
        raw_frames.append(sample[2])
        locations[i, :max_props_num, :] = sample[1].reshape((-1, 1))
        num_frames_list.append(sample[4])
        base_list.append(sample[5])
        
    raw_frames = torch.stack(raw_frames, dim=0)
    num_frames_list = torch.from_numpy(np.array(num_frames_list))
    base_list = torch.from_numpy(np.array(base_list))

    return vid_name_list, locations, raw_frames, target_list, num_frames_list, base_list


def build_multithumos(split, args):
    return MultiTHUMOS(args.annotation_path, args.frame_file_path, args.frame_folder, split, args.window_size, args.interval, args.num_classes, args.img_tensor, args.tensor_folder)


def build_charades(split, args):
    return Charades(args.annotation_path, args.frame_folder, split, args.window_size, args.interval, args.num_classes, args.img_tensor, args.tensor_folder)