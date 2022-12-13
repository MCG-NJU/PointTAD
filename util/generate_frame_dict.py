'''
Helper Script to generate 'number of frames' dictionary for extracted frames.

'''
import os 
import json
import pandas as pd

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

frame_path = 'data/multithumos_frames' # path to the extracted video frames
subsets = ['training','testing']
frame_dict = {}
for subset in subsets:
    video_list = os.listdir(os.path.join(frame_path, subset))
    for vid in video_list:
        num_frames = len(os.listdir(os.path.join(frame_path, subset, vid)))
        frame_dict[vid] = num_frames

f = open('datasets/multithumos_frames.json','w')
json.dump(frame_dict, sort_keys=True, indent=4, separators=(',', ':'), fp=f)
f.close()
