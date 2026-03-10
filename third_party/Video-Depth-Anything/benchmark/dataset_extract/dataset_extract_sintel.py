# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# # Data loading based on https://github.com/NVIDIA/flownet2-pytorch


import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import csv
import imageio
import cv2
import json
import glob
import shutil

from eval_utils import gen_json, get_sorted_files

TAG_FLOAT = 202021.25
TAG_CHAR = "PIEH"

def depth_read(filename):
    """Read depth data from file, return as numpy array."""
    f = open(filename, "rb")
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert (
        check == TAG_FLOAT
    ), " depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? ".format(
        TAG_FLOAT, check
    )
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert (
        width > 0 and height > 0 and size > 1 and size < 100000000
    ), " depth_read:: Wrong input size (width = {0}, height = {1}).".format(
        width, height
    )
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth

def extract_sintel(
    root,
    depth_root,
    sample_len=-1,
    datatset_name="",
    saved_dir="",
):
    scenes_names = os.listdir(root)
    all_samples = []
    for i, seq_name in enumerate(tqdm(scenes_names)):
        all_img_names = get_sorted_files(
            os.path.join(root, seq_name), suffix=".png")

        seq_len = len(all_img_names)
        step = sample_len if sample_len > 0 else seq_len

        for ref_idx in range(0, seq_len, step):
            print(f"Progress: {seq_name}, {ref_idx // step} / {seq_len // step}")

            if (ref_idx + step) <= seq_len:
                ref_e = ref_idx + step
            else:
                continue

            for idx in range(ref_idx, ref_e):
                im_path = osp.join(
                    root, seq_name, all_img_names[idx]
                )
                depth_path = osp.join(
                    depth_root, seq_name, all_img_names[idx][:-3] + "dpt"
                )
                out_img_path = osp.join(
                    saved_dir, datatset_name,'clean', seq_name, all_img_names[idx]
                )
                out_depth_path = osp.join(
                    saved_dir, datatset_name,'depth', seq_name, all_img_names[idx][:-3] + "png"
                )
                depth = depth_read(depth_path)
                img = np.array(Image.open(im_path))
                
                os.makedirs(osp.dirname(out_img_path), exist_ok=True)
                os.makedirs(osp.dirname(out_depth_path), exist_ok=True)

                cv2.imwrite(
                    out_img_path,
                    img,
                )
                cv2.imwrite(
                    out_depth_path,
                    depth.astype(np.uint16)
                )
    
    gen_json(
        root_path=osp.join(saved_dir, datatset_name), dataset=datatset_name,
        start_id=0,end_id=100,step=1,
        save_path=osp.join(saved_dir, datatset_name, "sintel_video.json"),)

if __name__ == "__main__":
    extract_sintel(
        root="path/to/training/clean",
        depth_root="path/to/depth",
        saved_dir="./benchmark/datasets/",
        sample_len=-1,
        datatset_name="sintel",
    )