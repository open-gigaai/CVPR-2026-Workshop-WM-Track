<div align="center" style="font-family: charter;">
    <h1> CVPR-2026-Workshop-WM-Track </h1>

[//]: # ([![License]&#40;https://img.shields.io/badge/License-Apache_2.0-blue.svg&#41;]&#40;https://opensource.org/licenses/Apache-2.0&#41;)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Project](https://img.shields.io/badge/Project-Page-99cc2)](https://gigaai-research.github.io/GigaBrain-Challenge-2026/index.html#about)
[![Models](https://img.shields.io/badge/Model-Huggingface-red?logo=huggingface)](https://huggingface.co/collections/open-gigaai/cvpr-2026-worldmodel-track)
[![Datasets](https://img.shields.io/badge/Dataset-Huggingface-blue?logo=huggingface)](https://huggingface.co/datasets/open-gigaai/CVPR-2026-WorldModel-Track-Dataset/tree/main)

</div>


## About

The repo contains the code and dataset for the World Models Track of GigaBrain Challenge 2026 CVPR Workshop. We provide the information of the dataset and the world models baseline code for training and inference on the track dataset. 


### Dataset

Download dataset from [huggingface](https://huggingface.co/datasets/open-gigaai/CVPR2026_WorldModel_Track). The data consists of multiple tasks.

Each sub-task dataset offering three functional splits as detailed below: the Train split provides full ground-truth (GT) videos and trajectories for supervised learning; the Video Quality split provides only first frames and full trajectories to benchmark conditional video generation; and the Evaluator split provides only initial frames and states to support closed-loop VLA (Vision-Language-Action) interaction and evaluation.

| Split | Ground Truth Videos | Trajectory Data | Initial State/Pose | Primary Usage                       |
| :--- |:-------------------:| :---: | :---: |:------------------------------------|
| **Train** |         ✅           | ✅  | ✅ | Model Training                      |
| **Video Quality** |          ❌          | ✅  | ✅ | Video Quality Benchmark             |
| **Evaluator** |          ❌          | ❌ | ✅ (Initial Only) | WM (as evaluator) & VLA interaction |

each task subdirectory has the following file structure,

```bash
task/
├── train/                    # Main training data
│   ├── metas/                # JSON files containing task instructions
│   │   ├── episode_0.json
│   │   └── ...
│   ├── trajectories/         # state sequences (.pkl)
│   │   ├── episode_0.pkl
│   │   └── ...
│   └── videos/               # Multiview video recordings (.mp4)
│       ├── cam_high/       
│       │   ├── episode_0.mp4
│       │   └── ...
│       ├── cam_left_wrist/  
│       └── cam_right_wrist/ 
├── evaluator/                # As evaluator test set
│   ├── episode_0/            # Test episode initial states
│   │   ├── cam_high.png      # Reference image (High view)
│   │   ├── cam_left_wrist.png
│   │   ├── cam_right_wrist.png
│   │   ├── meta.json        
│   │   └── initial_state.pkl 
│   └── ...                  
└── video_quality/            # Video quality evaluation set
    ├── episode_0/            
    │   ├── cam_high.png
    │   ├── cam_left_wrist.png
    │   ├── cam_right_wrist.png
    │   ├── meta.json
    │   └── traj.pkl
    └── ...
```
> **Bonus for training episode:** alongside the ground-truth videos, we also supply depth maps and simulator renderings.

![Demo Data](asserts/demo_data.gif)

### Environment setup

* Base environment

We provide baseline world model code for training and inference. [GigaTrain](https://github.com/open-gigaai/giga-train) and [GigaDataset](https://github.com/open-gigaai/giga-datasets) is used for framework of training and dataset loading respectively.
The base environment for training is the same as the [GigaTrain](https://github.com/open-gigaai/giga-train) repo and [GigaDataset](https://github.com/open-gigaai/giga-datasets).


```bash
conda create -n giga_torch python=3.11.10
conda activate giga_torch

# install giga-train
cd third_party/giga-train
pip3 install -e .

# install giga-datasets 
cd third_party/giga-datasets
pip3 install -e .
```

* Robotwin2.0 simulator environment

As the baseline world model use Robotwin2.0 simulator to render qpos action to images, we need to install the simulator following the [instruction](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).


### Train






### Inference


### Submission

> Please package the generated videos in the specified format and send them to **xxx@xxx.com** to complete the submission.

---

<div align="center">

**Connect & Collaborate**  
Join our vibrant community on [WeChat](https://www.wechat.com/en/) to share ideas, ask questions, and team up with fellow participants from around the globe.

<img src="asserts/community.png" alt="Community" width="300"/>
</div>
