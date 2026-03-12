import json
import os
import paths
from giga_datasets import Dataset, FileWriter, PklWriter, load_dataset
from cvpr_2026_workshop_wm_track.model_config import DATA_DIR
from glob import glob
from tqdm import tqdm
import pickle
import argparse

def get_three_view_video_paths(local_dir_name, task_dir, episode_name, data_type_name):
    data_dict = {}
    data_dict[f'cam_high_{data_type_name}_path'] = os.path.join(task_dir, local_dir_name, 'cam_high', f"{episode_name}.mp4")
    data_dict[f"cam_left_wrist_{data_type_name}_path"] = os.path.join(task_dir, local_dir_name, 'cam_left_wrist', f"{episode_name}.mp4")
    data_dict[f"cam_right_wrist_{data_type_name}_path"] = os.path.join(task_dir, local_dir_name, 'cam_right_wrist', f"{episode_name}.mp4")
    assert os.path.exists(data_dict[f'cam_high_{data_type_name}_path']), data_dict[f'cam_high_{data_type_name}_path']
    assert os.path.exists(data_dict[f"cam_left_wrist_{data_type_name}_path"]), data_dict[f"cam_left_wrist_{data_type_name}_path"]
    assert os.path.exists(data_dict[f"cam_right_wrist_{data_type_name}_path"]), data_dict[f"cam_right_wrist_{data_type_name}_path"]
    return data_dict



def pack_data_per_task(task_dir, ):
    label_writer = PklWriter(os.path.join(task_dir, "labels"))
    data_list = glob(os.path.join(task_dir, "metas", "*.json"))
    data_list = [os.path.basename(path).replace(".json", "") for path in data_list]
    data_index = 0
    for episode_name in tqdm(data_list, desc=f"Packing data on {task_dir}"):
        try:
            meta_info = json.load(open(os.path.join(task_dir, "metas", episode_name + ".json"), 'r'))
            qpos = pickle.load(open(os.path.join(task_dir, "trajectories", episode_name + ".pkl"), "rb"))
            depth_dict = get_three_view_video_paths('depth', task_dir, episode_name, 'depth')
            simulator_dict = get_three_view_video_paths('simulator', task_dir, episode_name, 'simulator')
            video_dict = get_three_view_video_paths('videos', task_dir, episode_name, 'video')
        except Exception as e:
            print(f"报错详情: {e}, {episode_name}")
        data_dict = {
            **depth_dict,
            **simulator_dict,
            **video_dict,
            **meta_info,
            "qpos": qpos,
            "episode_name": episode_name,
        }
        label_writer.write_dict(data_dict)
        data_index += 1

    label_writer.write_config()
    label_writer.close()

    label_dataset = load_dataset(os.path.join(task_dir, "labels"))
    dataset = Dataset([label_dataset])
    dataset.save(task_dir)
    print(f"Packed {data_index} episodes in {task_dir}")
    print(f"Dataset saved in {task_dir}")
    return task_dir


def pack_training_data(args):
    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR
    task = args.task if args.task != "all" else ['task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7', 'task8']
    if isinstance(task, str):
        task = [task]
    for task_name in task:
        task_dir = os.path.join(data_dir, task_name, 'train')
        pack_data_per_task(task_dir)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--data_dir', type=str, default=None)
    args.add_argument('--task', type=str, default='task1')
    args = args.parse_args()
    pack_training_data(args)