# 新增文件: script/replay_qpos.py
"""Replay qpos trajectories in a minimal SAPIEN scene and record 3 cameras.

- Input: a directory containing many episode folders/files produced by Aloha/LeRobot-style HDF5,
  with dataset key: /observations/qpos (shape [T, 14]).
- Output: 3 mp4 videos per episode: head_camera / left_camera / right_camera.

This script intentionally builds a minimal environment (table + wall + robot + cameras) without
loading task-specific actors.
"""

from __future__ import annotations

# 设置此代码的运行路径位于 /mnt/pfs/users/boyuan.wang/project/Robotwin2_onlyreplay
import sys
import time

sys.path.insert(0, "/shared_disk/users/yukun.zhou/codes/giga/giga-models/scripts/examples/diffusion/wa")
from sockets import RobotInferenceServer, RobotInferenceClient

host_port = 9151

import argparse
import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import imageio
import sys
import pickle
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))


@dataclass
class ReplayConfig:
    episodes_dir: Path
    out_dir: Path
    max_steps: Optional[int]


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

    # Import a concrete env class, but we won't call its load_actors/play_once.
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    env = env_class()

    args = _load_task_args(task_config_name)
    args["task_name"] = task_name

    # Embodiment mapping
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

    # Ensure we do collect the 3 cameras
    args.setdefault("camera", {})
    args["camera"]["collect_head_camera"] = True
    args["camera"]["collect_wrist_camera"] = True

    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = True
    args["data_type"]["qpos"] = True

    # Avoid task-specific actor loading by calling the base init directly.
    from envs._base_task import Base_Task

    Base_Task._init_task_env_(env, **args)

    # Make take_action not stop immediately.
    env.step_lim = int(1e6)
    env.take_action_cnt = 0
    # [env.scene.entities[i].name for i in range(len(env.scene.entities))]
    env.scene.remove_entity(env.scene.entities[0])  # remove ground
    env.scene.remove_entity(env.scene.entities[3])  # remove wall
    env.scene.remove_entity(env.scene.entities[3])  # remove table
    return env


def _save_pcd_as_ply(pcd: np.ndarray, ply_path: Path) -> None:
    """Save point cloud to PLY.

    pcd: (N,6) numpy array: xyz + rgb.
      - xyz: float32/float64
      - rgb: either 0-1 float, or 0-255 uint8/int
    """
    if pcd is None:
        return
    pcd = np.asarray(pcd)
    if pcd.ndim != 2 or pcd.shape[1] < 6:
        raise ValueError(f"pcd should be (N,6), got {pcd.shape}")

    xyz = pcd[:, :3].astype(np.float32, copy=False)
    rgb = pcd[:, 3:6]

    # Normalize color to uint8 0-255
    if np.issubdtype(rgb.dtype, np.floating):
        # assume 0-1
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    else:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    ply_path.parent.mkdir(parents=True, exist_ok=True)

    # Write binary little endian PLY for speed/size
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
        # interleave xyz + rgb
        data = np.empty((xyz.shape[0],),
                        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        data["x"] = xyz[:, 0]
        data["y"] = xyz[:, 1]
        data["z"] = xyz[:, 2]
        data["r"] = rgb_u8[:, 0]
        data["g"] = rgb_u8[:, 1]
        data["b"] = rgb_u8[:, 2]
        f.write(data.tobytes())


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

        pcd = obs["pointcloud"]  # [1024,6] np array
        frames["pcd"].append(pcd)

        world_pcd = obs["world_pcd"]
        frames["world_pcd"].append(world_pcd)

        for cam_name in list(frames.keys())[:3]:
            rgb = cam_obs[cam_name]["rgb"]
            if rgb.dtype != np.uint8:
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            frames[cam_name].append(rgb)
            depth = cam_obs[cam_name]["depth"]
            frames[cam_name + "_depth"].append(depth)
    return frames


import sys

sys.path.insert(0, '/shared_disk/users/yukun.zhou/codes/giga/giga-datasets/', )
from giga_datasets import load_dataset
from giga_datasets import utils as gd_utils
import torch


class ActionRender():

    def __init__(self):
        self.env = _build_minimal_env('replay_onlyrobot', 'replay_onlyrobot')

    def inference(self, data):
        qpos = data['action']
        frames = replay_one_episode(self.env, qpos, max_steps=None)

        head_frames = []
        left_frames = []
        right_frames = []

        for frame_idx in range(len(frames["head_camera"])):
            head_frame = frames["head_camera"][frame_idx]
            left_frame = frames["left_camera"][frame_idx]
            right_frame = frames["right_camera"][frame_idx]
            head_frames.append(head_frame)
            left_frames.append(left_frame)
            right_frames.append(right_frame)

        save_dir = "tmp/replay_videos"
        os.makedirs(save_dir, exist_ok=True)
        import datetime
        idx = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

        left_save_path = os.path.join(save_dir, "left_camera", f"{idx}.mp4")
        right_save_path = os.path.join(save_dir, "right_camera", f"{idx}.mp4")
        head_save_path = os.path.join(save_dir, "head_camera", f"{idx}.mp4")
        os.makedirs(os.path.dirname(left_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(right_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(head_save_path), exist_ok=True)

        abs_left_save_path = os.path.abspath(left_save_path)
        abs_right_save_path = os.path.abspath(right_save_path)
        abs_head_save_path = os.path.abspath(head_save_path)

        final_result = {
            "sim_front_rgb": abs_head_save_path,
            "sim_left_rgb": abs_left_save_path,
            "sim_right_rgb": abs_right_save_path,
        }

        imageio.mimwrite(left_save_path, left_frames, fps=30, macro_block_size=1)
        imageio.mimwrite(right_save_path, right_frames, fps=30, macro_block_size=1)
        imageio.mimwrite(head_save_path, head_frames, fps=30, macro_block_size=1)

        return final_result



def main(device, world_size, rank) -> None:
    # parser.add_argument("--out_dir", type=str, default='./tmp/replay_videos', help="输出视频目录")
    # parser.add_argument("--max_steps", type=int, default=None, help="最多回放多少步（默认全部）")
    # torch.cuda.set_device(device)
    # episodes_dir = '/shared_disk/datasets/private_datasets/robot_data/agilex_data/upload/20251215T014_pick_place_cyt001_01'
    # /shared_disk/datasets/private_datasets/robot_data/agilex_data/clean_desk/clean_desk_T1/20250819T005_clean_desk_lsz001T2_04/episode_1.hdf5
    episodes_dir = '/shared_disk/datasets/private_datasets/robot_data/agilex_data/clean_desk/clean_desk_T1/20250819T005_clean_desk_lsz001T2_04'
    # episodes_dir ='/shared_disk/datasets/private_datasets/robot_data/agilex_data/upload/wm_as_evaluator/task1/banana_left/20260114T005_grab_banana_left_lsz001_01'
    # json_path = "/shared_disk/users/yukun.zhou/dataset/action_conditional/put banana into basket/data_split.json"
    # data_paths = json.load(open(json_path, 'r'))
    hdf5_files = os.listdir(episodes_dir)

    data_path = "/shared_disk/users/yukun.zhou/dataset/action_conditional/gigaworld-1-task1/train_part0/"
    dataset = load_dataset(data_path)
    hdf5_files = list(range(len(dataset)))
    hdf5_files = gd_utils.split_data(hdf5_files, world_size, rank)

    # train_val_set = 'val'
    # hdf5_files = [Path(p) for p in data_paths[train_val_set]]
    # save_dir = f"/shared_disk/users/yukun.zhou/dataset/action_conditional/put banana into basket/{train_val_set}/replay"
    # os.makedirs(save_dir, exist_ok=True)
    save_dir = '/shared_disk/users/yukun.zhou/dataset/action_conditional/gigaworld-1-task1/train/replay'
    # cfg.out_dir.mkdir(parents=True, exist_ok=True)

    env = _build_minimal_env('replay_onlyrobot', 'replay_onlyrobot')

    try:
        for hdf5_path in hdf5_files:
            idx = hdf5_path
            data_dict = dataset[idx]

            # episode_name = hdf5_path.split('.')[0]
            # hdf5_path = os.path.join(episodes_dir, hdf5_path)
            # print(f"[{idx+1}/{len(hdf5_files)}] replay: {episode_name} ({hdf5_path})")
            # # if episode_name!='episode_1':
            # #     continue
            # size = os.path.getsize(hdf5_path)
            # if size == 0:
            #     print(f"跳过空文件{hdf5_path}")
            #     continue
            #
            # with h5py.File(hdf5_path, "r") as f:
            #     qpos = f["/observations/qpos"][:]
            #     gt_head_video = f['observations']['images']['cam_high'][:]
            #     gt_left_video = f['observations']['images']['cam_left_wrist'][:]
            #     gt_right_video = f['observations']['images']['cam_right_wrist'][:]
            qpos = data_dict["qpos"]
            qpos = np.asarray(qpos)
            # if qpos.ndim != 2:
            # raise ValueError(f"qpos should be 2D [T, D], got {qpos.shape} from {hdf5_path}")

            frames = replay_one_episode(env, qpos, max_steps=None)

            # out_dir=cfg.out_dir
            # out_dir.mkdir(parents=True, exist_ok=True)

            # Concatenate the 3 camera images horizontally and save as a single video
            # concatenated_frames = []
            head_frames = []
            left_frames = []
            right_frames = []

            # depth 保存（建议保存为 npz，保留 float 深度）
            head_depth_frames = []
            left_depth_frames = []
            right_depth_frames = []

            for frame_idx in range(len(frames["head_camera"])):
                head_frame = frames["head_camera"][frame_idx]
                left_frame = frames["left_camera"][frame_idx]
                right_frame = frames["right_camera"][frame_idx]
                head_frames.append(head_frame)
                left_frames.append(left_frame)
                right_frames.append(right_frame)

                head_camera_depth = frames["head_camera_depth"][frame_idx]
                left_camera_depth = frames["left_camera_depth"][frame_idx]
                right_camera_depth = frames["right_camera_depth"][frame_idx]
                head_depth_frames.append(np.asarray(head_camera_depth))
                left_depth_frames.append(np.asarray(left_camera_depth))
                right_depth_frames.append(np.asarray(right_camera_depth))

                world_pcd = frames["world_pcd"][frame_idx]
                # _save_pcd_as_ply(pcd, out_dir / episode_name / f"pcd_{frame_idx:04d}.ply")
                # _save_pcd_as_ply(world_pcd, out_dir / episode_name / f"world_pcd_{frame_idx:04d}.ply")

            # out_path = out_dir / f"{episode_name}_3view.mp4"
            # imageio.mimwrite(str(out_path), concatenated_frames, fps=30, macro_block_size=1)
            # imageio.mimwrite(str(out_dir / f"gt_{episode_name}_3view.mp4"),
            #                  [np.concatenate((gt_head_video[i], gt_left_video[i], gt_right_video[i]), axis=1) for i in range(len(gt_head_video))],
            #                  fps=30, macro_block_size=1)
            # Reset internal counters (scene/robot state continues, but take_action uses counter only for logging)

            left_save_path = os.path.join(save_dir, "left_camera", f"{idx}.mp4")
            right_save_path = os.path.join(save_dir, "right_camera", f"{idx}.mp4")
            head_save_path = os.path.join(save_dir, "head_camera", f"{idx}.mp4")
            os.makedirs(os.path.dirname(left_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(right_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(head_save_path), exist_ok=True)

            imageio.mimwrite(left_save_path, left_frames, fps=30, macro_block_size=1)
            imageio.mimwrite(right_save_path, right_frames, fps=30, macro_block_size=1)
            imageio.mimwrite(head_save_path, head_frames, fps=30, macro_block_size=1)

            # 保存 depth（每个 episode 一份文件；shape: [T,H,W]）
            depth_dir = os.path.join(save_dir, "depth")

            head_depth_frames = np.stack(head_depth_frames, axis=0)
            left_depth_frames = np.stack(left_depth_frames, axis=0)
            right_depth_frames = np.stack(right_depth_frames, axis=0)

            os.makedirs(depth_dir, exist_ok=True)

            def get_depth_render_array(depth_frames):
                new_depth_frames = []
                for i in range(depth_frames.shape[0]):
                    depths = depth_frames[i]
                    d_min, d_max = depths.min(), depths.max()
                    depth_norm = ((depths - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                    end_depth = depth_norm
                    gray = end_depth  # 变成 (480, 640)
                    rgb = np.stack([gray] * 3, axis=-1)  # 变成 (480, 640, 3)
                    new_depth_frames.append(rgb)
                return np.stack(new_depth_frames, axis=0)

            head_depth_render = get_depth_render_array(head_depth_frames)
            left_depth_render = get_depth_render_array(left_depth_frames)
            right_depth_render = get_depth_render_array(right_depth_frames)

            imageio.mimwrite(os.path.join(depth_dir, f"{idx}_head_depth.mp4"), head_depth_render, fps=30,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(depth_dir, f"{idx}_left_depth.mp4"), left_depth_render, fps=30,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(depth_dir, f"{idx}_right_depth.mp4"), right_depth_render, fps=30,
                             macro_block_size=1)

            # np.savez_compressed(
            #     os.path.join(depth_dir, f"{idx}.npz"),
            #     head=np.stack(head_depth_frames, axis=0),
            #     left=np.stack(left_depth_frames, axis=0),
            #     right=np.stack(right_depth_frames, axis=0),
            # )

            env.take_action_cnt = 0

    finally:
        env.close_env(clear_cache=True)


def server():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host_port', type=int, default=host_port)
    args = parser.parse_args()
    host_port = args.host_port
    net = ActionRender()
    server = RobotInferenceServer(net, host="0.0.0.0", port=host_port)
    print("Starting server at {}".format(host_port))
    server.run()


if __name__ == "__main__":
    server()

# /mnt/pfs/users/boyuan.wang/project/Robotwin2
# python script/sim_replay.py