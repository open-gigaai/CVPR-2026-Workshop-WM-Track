<div align="center" style="font-family: charter;">
    <h1> CVPR-2026-Workshop-WM-Track </h1>

[//]: # ([![License]&#40;https://img.shields.io/badge/License-Apache_2.0-blue.svg&#41;]&#40;https://opensource.org/licenses/Apache-2.0&#41;)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Project](https://img.shields.io/badge/Project-Page-99cc2)](https://gigaai-research.github.io/GigaBrain-Challenge-2026/index.html#about)
[![Models](https://img.shields.io/badge/Model-Huggingface-red?logo=huggingface)](https://huggingface.co/collections/open-gigaai/cvpr-2026-worldmodel-track)
[![Datasets](https://img.shields.io/badge/Dataset-Huggingface-blue?logo=huggingface)](https://huggingface.co/datasets/open-gigaai/CVPR-2026-WorldModel-Track-Dataset/tree/main)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Huggingface-yellow?logo=huggingface)](https://huggingface.co/spaces/open-gigaai/CVPR-2026-WorldModel-Track-LeaderBoard)

</div>

<div align="center">
<img src="asserts/illustration.gif" alt="illustration" height="400"/>
</div>

## About

The repo contains the code and dataset for the World Models Track of GigaBrain Challenge 2026 CVPR Workshop. We provide the information of the dataset and the world models baseline code for training and inference on the track dataset.


### Dataset

Download dataset from [huggingface](https://huggingface.co/datasets/open-gigaai/CVPR-2026-WorldModel-Track-Dataset). The data consists of multiple tasks.

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

## Environment setup and Pretrained Model Download

* **Base environment**

We provide baseline world model code for training and inference. [GigaTrain](https://github.com/open-gigaai/giga-train) and [GigaDataset](https://github.com/open-gigaai/giga-datasets) is used for framework of training and dataset loading respectively.
The base environment for training is the same as the [GigaTrain](https://github.com/open-gigaai/giga-train) and [GigaDataset](https://github.com/open-gigaai/giga-datasets).


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

* **Robotwin2.0 simulator environment**

As the baseline world model use Robotwin2.0 simulator to render qpos action to images, we need to install the simulator following the [instruction](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).

* **Download pretrained model**

We put the all needed pretrained model information in code `cvpr_2026_workshop_wm_track/model_config.py`. You can change the `HUGGINGFACE_MODEL_CACHE` to your own cache directory. And download the pretrained models by running the following command.

```bash
# change HUGGINGFACE_MODEL_CACHE to your own cache directory
HUGGINGFACE_MODEL_CACHE = "/shared_disk/models/huggingface" # line 3 of cvpr_2026_workshop_wm_track/model_config.py

# download pretrained models
python scripts/download_pretrained_models.py

# download gigabrain policy for online evaluation
python scripts/download_gigabrain_policy.py

```

## Train

We provide a simple training script that integrates with GigaTrain.  
Below are the steps to launch training after you have packed the data and downloaded pretrained models.


1. Pack training data in giga-datasets format 

Pack training data for each task. If you want to pack all tasks, you can set `--task all`. 

```bash
# USE DEFAULT DATA_DIR if not specified, You can also use the default DATA_DIR in model_config.py
python scripts/pack_training_data.py --task all 
# pack task4 data
python scripts/pack_training_data.py --data_dir /path/to/dataset --task task4 
# pack task1-8 data
python scripts/pack_training_data.py --data_dir /path/to/dataset --task all 
```

2. Modify training config 

Modify `cvpr_2026_workshop_wm_track/configs/baseline_wm_task4.py` to specify training setting:  

- <span style="color:#1f77b4">_project_dir_</span>: <span style="color:#ff7f0e">set logging and checkpoint save directory</span>
- <span style="color:#1f77b4">_launch.gpu_ids_</span>: <span style="color:#ff7f0e">set available devices</span>
- <span style="color:#1f77b4">_train.checkpoint_interval_</span>: <span style="color:#ff7f0e">save checkpointing interval per epoch</span>


3. Launch training

```bash
# launch baseline world model training on task4 dataset
python scripts/launch_train.py --config_path cvpr_2026_workshop_wm_track.configs.baseline_wm_task4.config
# launch baseline world model training on all task dataset
python scripts/launch_train.py --config_path cvpr_2026_workshop_wm_track.configs.baseline_wm_alltask.config
```

## Inference

After training, you can use the world model to simulate robot behavior. We provide two inference modes:

**Offline**: No interaction with any policy; the world model directly consumes the trajectory data (e.g., `traj.pkl`) to generate future video frames. This mode is used for the **Video Quality** benchmark—purely evaluating the model’s ability to predict visual dynamics given ground-truth actions.

```bash
python scripts/inference.py --transformer_model_path /path/to/transformer --device_list 0,1,2,3 --output_dir outputs/baseline_wm --task task4 --mode offline
```

---

**Online**: The world model runs in a closed loop with a **policy** that outputs actions in real time. This mode is used for the **Evaluator** benchmark—testing how well the world model supports downstream VLA (Vision-Language-Action) agents by providing accurate next-state predictions under the policy’s actual action distribution.

Before online inference, you need to prepare simulator server which render qpos to images and policy server which get action from initial image and state.

1. Start simulator server

As the environment is not compatible with the baseline world model, we provide a separate simulator server for online inference.

```bash
# start simulator server, default port is 9051
python simulator/script/run_simulator_server.py --host_port 9151
```

2. Start world model & policy interaction inference

```bash
# as the simualtor server is not multi-thread, we only use one device for inference 
python scripts/inference.py --transformer_model_path /path/to/transformer --device_list 0 --output_dir outputs/baseline_wm --task task4 --mode online --policy_ckpt_dir /path/to/policy --policy_norm_stats /path/to/norm_stat_gigabrain.json --simulator_ip 127.0.0.1 --simulator_port 9151
```


### Submission

After online and offline inference, you can get below structure outputs:

```
outputs
├── video_quality_eval
│   ├── task1
│   │   ├── episode_0.mp4
│   │   └── ...
│   └── ...
└── evaluator_test
    ├── task1
    │   ├── episode_0.mp4
    │   └── ...
    └── ...
```

> Follow the instructions on the [World Model Track Leaderboard](https://huggingface.co/spaces/open-gigaai/CVPR-2026-WorldModel-Track-LeaderBoard) to package and submit the generated videos for all tasks in the required format.

---

<div align="center">

**Connect & Collaborate**  
Join community on [WeChat](https://www.wechat.com/en/) to share ideas, and team up with fellow participants.

<img src="asserts/community.jpg" alt="Community" width="200"/>
</div>
