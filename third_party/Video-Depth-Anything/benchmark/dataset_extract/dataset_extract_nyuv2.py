import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import csv
import cv2
import json
import glob
from natsort import natsorted
import shutil

from eval_utils import gen_json, get_sorted_files, copy_crop_files

def extract_nyuv2(
    root,
    sample_len=-1,
    datatset_name="",
    saved_dir="",
):
    scenes_names = os.listdir(root)
    scenes_names = sorted(scenes_names)
    all_samples = []
    for i, seq_name in enumerate(tqdm(scenes_names)):
        all_img_names = get_sorted_files(
            osp.join(root, seq_name, "rgb"), suffix=".jpg")

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
                    root, seq_name, "depth", all_img_names[idx][:-3] + "png"
                )
                out_img_path = osp.join(
                    saved_dir, datatset_name, seq_name, "rgb", all_img_names[idx]
                )
                out_depth_path = osp.join(
                    saved_dir, datatset_name, seq_name, "depth", all_img_names[idx][:-3] + "png"
                )
            
                copy_crop_files(
                    im_path=im_path,
                    depth_path=depth_path,
                    out_img_path=out_img_path,
                    out_depth_path=out_depth_path,
                    dataset=dataset_name,
                )

    #~500 frames in paper
    out_json_path = osp.join(saved_dir, datatset_name, "nyuv2_video_500.json")    
    gen_json(
        root_path=osp.join(saved_dir, datatset_name), dataset=datatset_name,
        start_id=0,end_id=500,step=1,
        save_path=out_json_path)

if __name__ == "__main__":
    # we use matlab to extract 8 scenes from NYUv2
    #--basement_0001a, bookstore_0001a, cafe_0001a, classroom_0001a, kitchen_0003, office_0004, playroom_0002, study_0002
    extract_scannet(
        root="path/to/nyuv2",
        saved_dir="./benchmark/datasets/",
        sample_len=-1,
        datatset_name="nyuv2",
    )