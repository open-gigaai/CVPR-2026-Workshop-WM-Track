# script/run_simulator_server.py
"""Replay qpos trajectories in a minimal SAPIEN scene and record 3 cameras.

- Input: action sequence from socket request, stored in data["action"].
- Output: 3 mp4 videos per request: head_camera / left_camera / right_camera.
- This script intentionally builds a minimal environment
  (table + wall + robot + cameras) without loading task-specific actors.

Key improvements in this version:
1. Support instance-level save isolation via --save_tag
2. Support configurable root output directory via --save_root
3. Use microsecond timestamp + uuid to avoid filename collisions
4. Default save_tag falls back to sim_<host_port>
"""

from __future__ import annotations

import sys
import os
import uuid
import argparse
import importlib
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import imageio
import pickle
import json

# ---------------------------------------------------------------------
# External path setup
# ---------------------------------------------------------------------
sys.path.insert(0, "/shared_disk/users/yukun.zhou/codes/giga/giga-models/scripts/examples/diffusion/wa")
from sockets import RobotInferenceServer, RobotInferenceClient

default_host_port = 9151
DEFAULT_GRIPPER_CLOSE = 5e-4
DEFAULT_GRIPPER_OPEN = 6.9e-2
DEFAULT_GRIPPER_MAX_NORMALIZED = 0.8

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(
    1,
    "/shared_disk/users/zhenyu.wu/codes/giga/giga/giga-train/projects/CVPR-2026-Workshop-WM-Track/simulator",
)

sys.path.insert(0, "/shared_disk/users/yukun.zhou/codes/giga/giga-datasets/")
from giga_datasets import load_dataset
from giga_datasets import utils as gd_utils
import torch


# ---------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------
@dataclass
class ReplayConfig:
    episodes_dir: Path
    out_dir: Path
    max_steps: Optional[int]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _get_embodiment_config(robot_file: str) -> Dict:
    import yaml

    cfg_path = os.path.join(robot_file, "config.yml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _load_task_args(task_config_name: str) -> Dict:
    import yaml

    cfg_path = Path(__file__).resolve().parent.parent / "task_config" / f"{task_config_name}.yml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"task_config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _load_embodiment_mapping() -> Dict:
    import yaml
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _build_minimal_env(task_name: str, task_config_name: str):
    """Create env instance and initialize minimal scene (no task actors)."""

    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    env = env_class()

    args = _load_task_args(task_config_name)
    args["task_name"] = task_name

    embodiment_type = args.get("embodiment")
    emb_map = _load_embodiment_mapping()

    def get_embodiment_file(emb_type: str) -> str:
        robot_file = emb_map[emb_type]["file_path"]
        if robot_file is None:
            raise ValueError(f"Missing embodiment files for {emb_type}")
        return robot_file

    if not embodiment_type:
        raise ValueError("task_config.yml must contain 'embodiment'")

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])

    # Minimal deterministic settings
    args["need_plan"] = False
    args["render_freq"] = 0
    args["save_data"] = False
    args["eval_mode"] = False

    # Ensure 3 cameras are collected
    args.setdefault("camera", {})
    args["camera"]["collect_head_camera"] = True
    args["camera"]["collect_wrist_camera"] = True

    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = True
    args["data_type"]["qpos"] = True

    # Avoid task-specific actor loading by calling base init directly
    from envs._base_task import Base_Task

    Base_Task._init_task_env_(env, **args)

    # Make take_action not stop immediately
    env.step_lim = int(1e6)
    env.take_action_cnt = 0

    # Remove ground / wall / table
    env.scene.remove_entity(env.scene.entities[0])
    env.scene.remove_entity(env.scene.entities[3])
    env.scene.remove_entity(env.scene.entities[3])

    return env


def _save_pcd_as_ply(pcd: np.ndarray, ply_path: Path) -> None:
    """Save point cloud to PLY.

    pcd: (N,6) numpy array: xyz + rgb.
    """
    if pcd is None:
        return

    pcd = np.asarray(pcd)
    if pcd.ndim != 2 or pcd.shape[1] < 6:
        raise ValueError(f"pcd should be (N,6), got {pcd.shape}")

    xyz = pcd[:, :3].astype(np.float32, copy=False)
    rgb = pcd[:, 3:6]

    if np.issubdtype(rgb.dtype, np.floating):
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    else:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    ply_path.parent.mkdir(parents=True, exist_ok=True)

    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {xyz.shape[0]}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header\n",
        ]
    )

    with open(ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        data = np.empty(
            (xyz.shape[0],),
            dtype=[
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
            ],
        )
        data["x"] = xyz[:, 0]
        data["y"] = xyz[:, 1]
        data["z"] = xyz[:, 2]
        data["r"] = rgb_u8[:, 0]
        data["g"] = rgb_u8[:, 1]
        data["b"] = rgb_u8[:, 2]
        f.write(data.tobytes())


def _normalize_gripper_column(
    values: np.ndarray,
    close_value: float,
    open_value: float,
    max_normalized_value: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not 0.0 <= max_normalized_value <= 1.0:
        raise ValueError("max_normalized_value must be within [0, 1]")

    min_val = float(np.nanmin(values))
    max_val = float(np.nanmax(values))
    spread = max_val - min_val
    already_normalized = min_val >= -0.05 and max_val <= 1.05 and spread > 0.2
    if already_normalized:
        return np.clip(values, 0.0, max_normalized_value)

    denom = open_value - close_value
    if abs(denom) < 1e-8:
        raise ValueError("Invalid gripper normalization range: open_value equals close_value")
    normalized = (values - close_value) / denom
    return np.clip(normalized, 0.0, max_normalized_value)


def _normalize_qpos_for_replay(env, qpos: np.ndarray, gripper_max_normalized: float) -> np.ndarray:
    normalized_qpos = np.asarray(qpos, dtype=np.float32).copy()
    if normalized_qpos.ndim != 2:
        raise ValueError(f"qpos must be a 2D array [T, D], got shape {normalized_qpos.shape}")

    left_arm_dim = len(env.robot.get_left_arm_jointState()) - 1
    right_arm_dim = len(env.robot.get_right_arm_jointState()) - 1
    expected_dim = left_arm_dim + right_arm_dim + 2
    if normalized_qpos.shape[1] < expected_dim:
        raise ValueError(
            f"qpos last dimension is too small: got {normalized_qpos.shape[1]}, expected at least {expected_dim}"
        )

    left_idx = left_arm_dim
    right_idx = left_arm_dim + right_arm_dim + 1
    normalized_qpos[:, left_idx] = _normalize_gripper_column(
        normalized_qpos[:, left_idx],
        close_value=DEFAULT_GRIPPER_CLOSE,
        open_value=DEFAULT_GRIPPER_OPEN,
        max_normalized_value=gripper_max_normalized,
    )
    normalized_qpos[:, right_idx] = _normalize_gripper_column(
        normalized_qpos[:, right_idx],
        close_value=DEFAULT_GRIPPER_CLOSE,
        open_value=DEFAULT_GRIPPER_OPEN,
        max_normalized_value=gripper_max_normalized,
    )
    return normalized_qpos


def replay_one_episode(env, qpos: np.ndarray, max_steps: Optional[int]) -> Dict[str, List[np.ndarray]]:
    frames: Dict[str, List[np.ndarray]] = {
        "head_camera": [],
        "left_camera": [],
        "right_camera": [],
        "pcd": [],
        "world_pcd": [],
        "head_camera_depth": [],
        "left_camera_depth": [],
        "right_camera_depth": [],
    }

    if max_steps is not None:
        qpos = qpos[:max_steps]

    for t in range(qpos.shape[0]):
        env.take_action_simple(qpos[t], action_type="qpos")
        obs = env.get_obs()

        cam_obs = obs["observation"]

        pcd = obs["pointcloud"]
        frames["pcd"].append(pcd)

        world_pcd = obs["world_pcd"]
        frames["world_pcd"].append(world_pcd)

        for cam_name in ["head_camera", "left_camera", "right_camera"]:
            rgb = cam_obs[cam_name]["rgb"]
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            frames[cam_name].append(rgb)

            depth = cam_obs[cam_name]["depth"]
            frames[f"{cam_name}_depth"].append(depth)

    return frames


# ---------------------------------------------------------------------
# Core service class
# ---------------------------------------------------------------------
class ActionRender:
    def __init__(
        self,
        save_root: str = "tmp/replay_videos",
        save_tag: Optional[str] = None,
        use_uuid: bool = True,
        gripper_max_normalized: float = DEFAULT_GRIPPER_MAX_NORMALIZED,
    ):
        self.env = _build_minimal_env("replay_onlyrobot", "replay_onlyrobot")
        self.save_root = save_root
        self.save_tag = save_tag
        self.use_uuid = use_uuid
        self.gripper_max_normalized = gripper_max_normalized

    def _build_unique_id(self) -> str:
        time_str = datetime.datetime.now().strftime("%Y%m%dT%H%M%S_%f")
        if self.use_uuid:
            rand_str = uuid.uuid4().hex[:8]
            return f"{time_str}_{rand_str}"
        return time_str

    def _build_save_dir(self) -> str:
        if self.save_tag is not None and len(self.save_tag) > 0:
            return os.path.join(self.save_root, self.save_tag)
        return self.save_root

    def inference(self, data):
        qpos = _normalize_qpos_for_replay(self.env, data["action"], self.gripper_max_normalized)
        frames = replay_one_episode(self.env, qpos, max_steps=None)

        head_frames = []
        left_frames = []
        right_frames = []

        for frame_idx in range(len(frames["head_camera"])):
            head_frames.append(frames["head_camera"][frame_idx])
            left_frames.append(frames["left_camera"][frame_idx])
            right_frames.append(frames["right_camera"][frame_idx])

        unique_id = self._build_unique_id()
        save_dir = self._build_save_dir()

        left_dir = os.path.join(save_dir, "left_camera")
        right_dir = os.path.join(save_dir, "right_camera")
        head_dir = os.path.join(save_dir, "head_camera")

        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)
        os.makedirs(head_dir, exist_ok=True)

        left_save_path = os.path.join(left_dir, f"{unique_id}.mp4")
        right_save_path = os.path.join(right_dir, f"{unique_id}.mp4")
        head_save_path = os.path.join(head_dir, f"{unique_id}.mp4")

        abs_left_save_path = os.path.abspath(left_save_path)
        abs_right_save_path = os.path.abspath(right_save_path)
        abs_head_save_path = os.path.abspath(head_save_path)

        imageio.mimwrite(left_save_path, left_frames, fps=30, macro_block_size=1)
        imageio.mimwrite(right_save_path, right_frames, fps=30, macro_block_size=1)
        imageio.mimwrite(head_save_path, head_frames, fps=30, macro_block_size=1)

        final_result = {
            "sim_front_rgb": abs_head_save_path,
            "sim_left_rgb": abs_left_save_path,
            "sim_right_rgb": abs_right_save_path,
        }
        return final_result


# ---------------------------------------------------------------------
# Server entry
# ---------------------------------------------------------------------
def server():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_port", type=int, default=default_host_port)
    parser.add_argument("--save_root", type=str, default="tmp/replay_videos")
    parser.add_argument("--save_tag", type=str, default=None)
    parser.add_argument("--disable_uuid", action="store_true")
    parser.add_argument(
        "--gripper-max-normalized",
        type=float,
        default=DEFAULT_GRIPPER_MAX_NORMALIZED,
        help="Clamp normalized gripper opening to [0, this_value] before replay.",
    )
    args = parser.parse_args()

    host_port = args.host_port
    save_tag = args.save_tag if args.save_tag is not None else f"sim_{host_port}"

    net = ActionRender(
        save_root=args.save_root,
        save_tag=save_tag,
        use_uuid=not args.disable_uuid,
        gripper_max_normalized=args.gripper_max_normalized,
    )

    server = RobotInferenceServer(net, host="0.0.0.0", port=host_port)

    print(f"Starting server at {host_port}")
    print(f"save_root: {args.save_root}")
    print(f"save_tag:  {save_tag}")
    print(f"video_dir: {os.path.abspath(os.path.join(args.save_root, save_tag))}")
    print(f"gripper_max_normalized: {args.gripper_max_normalized}")

    server.run()


if __name__ == "__main__":
    server()


"""
Example usage

# simulator 1
CUDA_VISIBLE_DEVICES=4 python script/run_simulator_server.py \
    --host_port 9151 \
    --save_tag sim9151

CUDA_VISIBLE_DEVICES=5 python script/run_simulator_server.py \
    --host_port 9152 \
    --save_tag sim9152

CUDA_VISIBLE_DEVICES=6 python script/run_simulator_server.py \
    --host_port 9153 \
    --save_tag sim9153

CUDA_VISIBLE_DEVICES=7 python script/run_simulator_server.py \
    --host_port 9154 \
    --save_tag sim9154

# simulator 2
CUDA_VISIBLE_DEVICES=1 python script/run_simulator_server.py \
    --host_port 9154 \
    --save_tag sim9154

# simulator 3 with custom root
CUDA_VISIBLE_DEVICES=6 python script/run_simulator_server.py \
    --host_port 9152 \
    --save_root tmp/replay_videos_exp \
    --save_tag sim9152
"""
