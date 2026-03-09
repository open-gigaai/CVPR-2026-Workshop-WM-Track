<div align="center" style="font-family: charter;">
    <h1> CVPR-2026-Workshop-WM-Track </h1>

[//]: # ([![License]&#40;https://img.shields.io/badge/License-Apache_2.0-blue.svg&#41;]&#40;https://opensource.org/licenses/Apache-2.0&#41;)
[![Project](https://img.shields.io/badge/Project-Page-99cc2)](https://giga-world-0.github.io/)
[![Models](https://img.shields.io/badge/Model-Huggingface-red?logo=huggingface)](https://huggingface.co/open-gigaai)
[![Datasets](https://img.shields.io/badge/Dataset-Huggingface-blue?logo=huggingface)](https://huggingface.co/open-gigaai)

</div>


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



### Inference

