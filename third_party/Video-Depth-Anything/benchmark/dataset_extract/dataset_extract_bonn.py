import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import cv2
import csv
import json
import glob
import shutil
from natsort import natsorted

from eval_utils import gen_json, get_sorted_files, even_or_odd, copy_crop_files

def extract_bonn(
    root,
    depth_root,
    saved_dir,
    sample_len,
    datatset_name,
):
    scenes_names = os.listdir(depth_root)
    all_samples = []
    for i, seq_name in enumerate(tqdm(scenes_names)):
        # load all images
        all_img_names = get_sorted_files(
            root=osp.join(depth_root, seq_name, "rgb"), suffix=".png"
        )
        all_depth_names = get_sorted_files(
            root=osp.join(depth_root, seq_name, "depth"), suffix=".png"
        )

        seq_len = len(all_img_names)
        step = sample_len if sample_len > 0 else seq_len

        for ref_idx in range(0, seq_len, step):
            print(f"Progress: {seq_name}, {ref_idx // step + 1} / {seq_len//step}")
            
            if (ref_idx + step) <= seq_len:
                ref_e = ref_idx + step
            else:
                continue

            for idx in range(ref_idx, ref_e):
                im_path = osp.join(
                    root, seq_name, "rgb", all_img_names[idx]
                )
                depth_path = osp.join(
                    depth_root, seq_name, "depth", all_depth_names[idx]
                )
                out_img_path = osp.join(
                    saved_dir, datatset_name,seq_name, "rgb", all_img_names[idx]
                )
                out_depth_path = osp.join(
                    saved_dir, datatset_name,seq_name, "depth", all_depth_names[idx]
                )

                copy_crop_files(
                    im_path=im_path,
                    depth_path=depth_path,
                    out_img_path=out_img_path,
                    out_depth_path=out_depth_path,
                    dataset=datatset_name,
                )

    # 110 frames like DepthCraft
    out_json_path = osp.join(saved_dir, datatset_name, "bonn_video.json")
    gen_json(
        root_path=osp.join(saved_dir, datatset_name), dataset=datatset_name,
        start_id=30, end_id=140, step=1, save_path=out_json_path)
    
    #~500 frames in paper
    out_json_path = osp.join(saved_dir, datatset_name, "bonn_video_500.json")
    gen_json(
        root_path=osp.join(saved_dir, datatset_name), dataset=datatset_name,
        start_id=0, end_id=500, step=1, save_path=out_json_path)


if __name__ == "__main__":
    extract_bonn(
        root="path/to/Bonn-RGBD",
        depth_root="path/to/Bonn-RGBD",
        saved_dir="./benchmark/datasets/",
        sample_len=-1,
        datatset_name="bonn",
    )