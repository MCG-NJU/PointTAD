# PointTAD [NeurIPS 2022]

This repo holds the codes of paper: "PointTAD: Multi-Label Temporal Action Detection with Learnable Query Points", which is accepted in NeurIPS 2022.

<img src="imgs/pointtad.png" alt="pointtad" style="zoom:60%;" />

[[Paper Link](https://openreview.net/pdf?id=_r8pCrHwq39)] [[Zhihu](https://zhuanlan.zhihu.com/p/591495791)]

## News

**[Dec. 13, 2022]**  We release the codes and checkpoints on MultiTHUMOS and Charades.

## Overview

This paper presents a query-based framework for multi-label temporal action detection, namely PointTAD, that leverages a set of **learnable query points** to handle both boundary frames and action semantic keyframes for finer action representation. Our model takes **RGB input only** and streamlines an **end-to-end trainable** framework for easy deployment. PointTAD surpasses previous multi-label TAD works by a large margin under detection-mAP and achieves comparable results under segmentation-mAP.
- [PointTAD \[NeurIPS 2022\]](#pointtad-neurips-2022)
	- [News](#news)
	- [Overview](#overview)
	- [Dependencies](#dependencies)
	- [Data Preparation](#data-preparation)
	- [Checkpoints](#checkpoints)
	- [Testing](#testing)
	- [Training](#training)
	- [Acknowledgements](#acknowledgements)
	- [Citations](#citations)
	- [Contacts](#contacts)

## Dependencies

PyTorch 1.8.1 or higher, opencv-python, scipy, terminaltables, ruamel-yaml, **ffmpeg**

run `start.sh` to install packages.

## Data Preparation

To prepare the RGB frames and corresponding annotations,

- Clone the repository and `cd PointTAD; mkdir data` 
- **MultiTHUMOS:** 
  - Download the raw videos of THUMOS14 into `/data/thumos14_videos`;
  - Extract the RGB frames of MultiTHUMOS from raw videos using  `utils/extract_frames.py`.  The frames will be placed in `/data/multithumos_frames`;
  - You also need to generate `multithumos_frames.json` for the extracted frames with  `/util/generate_frame_dict.py` and put the json file into `/dataset` folder.
- **Charades:** 
  - Download the RGB frames of Charades from [here](https://prior.allenai.org/projects/charades) , and place the frames at  `/data/charades_v1_rgb`.

**[Optional but Suggested]** Once you had the raw frames, you can convert them into tensors with `/util/frames2tensor.py` to **speed up IO**. By enabling  `--img_tensor` in `train.sh` and `test.sh`, the model takes in image tensors instead of frames.

✨ Don't forget to replace the frame folder path or image tensor path in `/data/dataset_cfg.yml`.

The structure of `data/` is displayed as follows:

```
|-- data
|   |-- thumos14_videos
|   |   |-- training
|   |   |-- testing
|   |-- multithumos_frames
|   |   |-- training
|   |   |-- testing
|   |-- multithumos_tensors [optional]
|   |   |-- training
|   |   |-- testing
|   |-- charades_v1_rgb
|   |-- charades_v1_rgb_tensors [optional]
```

## Checkpoints

The checkpoints are tested on server with 8 A100 cards (40GB).

| Dataset         | mAP@0.2 | mAP@0.5 | mAP@0.7 | Avg-mAP | Checkpoint                                                   |
| --------------- | ------- | ------- | ------- | ------- | ------------------------------------------------------------ |
| **MultiTHUMOS** | 36.80%  | 23.33%  | 10.95%  | 21.72%  | [Link](https://drive.google.com/file/d/1OoCZJvBDFaFZq0mPHlUAIW384dw_CzHo/view?usp=sharing) |
| **Charades**    | 15.93%  | 12.59%  | 8.49%   | 11.26%  | [Link](https://drive.google.com/file/d/1ceatzqOb9-BJ89Bdtzuzz9R4nrL1FiIS/view?usp=sharing) |

![image-20221213161122232](imgs/image-20221213161122232.png)

![image-20221213161244629](imgs/image-20221213161244629.png)

## Testing

Use `test.sh` to evaluate,

- **MultiTHUMOS**: 

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11302 --use_env main.py --img_tensor --dataset multithumos --eval --load multithumos_best.pth
```

- **Charades**:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11302 --use_env main.py --img_tensor --dataset charades --eval --load charades_best.pth
```

## Training 

Use `train.sh` to train PointTAD,

- **MultiTHUMOS**:

`````````
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11302 --use_env main.py --dataset multithumos
`````````

- **Charades**:

````
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=11302 --use_env main.py --dataset charades
````



## Acknowledgements

The codebase is built on top of [RTD-Net](https://github.com/MCG-NJU/RTD-Action), [DETR](https://github.com/facebookresearch/detr), [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN) and [AFSD](https://github.com/TencentYoutuResearch/ActionDetection-AFSD/), we thank them for providing useful codes.

## Citations

If you think our work is useful, please feel free to cite our paper:

```
@inproceedings{
	tan2022pointtad,
	title={Point{TAD}: Multi-Label Temporal Action Detection with Learnable Query Points},
	author={Jing Tan and Xiaotong Zhao and Xintian Shi and Bin Kang and Limin Wang},
	booktitle={Advances in Neural Information Processing Systems},
	editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
	year={2022},
	url={https://openreview.net/forum?id=_r8pCrHwq39}
}
```

## Contacts

Jing Tan: jtan@smail.nju.edu.cn
