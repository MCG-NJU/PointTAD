'''
Helper Script to transform raw RGB frames into image tensors.

'''
import os
import numpy as np
import cv2
import torch
from tqdm import tqdm

dataset = 'multithumos' # or charades

if dataset == 'multithumos':
    splits = ['training','testing']
    frame_folder_path = 'data/multithumos_frames/' # path to rgb frame folder
    tensor_path = 'data/multithumos_tensors/' # path to image tensor folder
else:
    splits = ['']
    frame_folder_path = 'data/charades_v1_rgb/' # path to rgb frame folder
    tensor_path = 'data/charades_v1_rgb_tensors/' # path to image tensor folder

for split in splits:
    vid_list = os.listdir(os.path.join(frame_folder_path, split))
    for vid in tqdm(vid_list):
        if os.path.isfile(os.path.join(tensor_path, split, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(frame_folder_path, split,vid)))
        block_idx = list(range(1,num_frames+1,1))
        if dataset == 'multithumos':
            img_stacked = [os.path.join(frame_folder_path, split, vid, 'img_{:05d}.jpg'.format(i)) for i in block_idx]
        else:
            img_stacked = [os.path.join(frame_folder_path, vid, '{}-{:06d}.jpg'.format(vid,i)) for i in block_idx]

        img_list = []
        for i in range(len(img_stacked)):
            img_path = img_stacked[i]
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img_list.append(img)
        img_list = np.stack(img_list,axis=0)
        img_tensor = torch.from_numpy(img_list)
        print(vid, img_tensor.shape)
        torch.save(img_tensor, os.path.join(tensor_path, split, vid))
