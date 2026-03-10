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

from eval_utils import gen_json, gen_json_scannet_tae, get_sorted_files, copy_crop_files

def extract_scannet(
    root,
    sample_len=-1,
    datatset_name="",
    saved_dir="",
):
    scenes_names = os.listdir(root)
    scenes_names = sorted(scenes_names)[:100]
    all_samples = []
    for i, seq_name in enumerate(tqdm(scenes_names)):
        all_img_names = get_sorted_files(
            osp.join(root, seq_name, "color"), suffix=".jpg")
        all_img_names = all_img_names[:510]   

        seq_len = len(all_img_names)
        step = sample_len if sample_len > 0 else seq_len

        for ref_idx in range(0, seq_len, step):
            print(f"Progress: {seq_name}, {ref_idx // step + 1} / {seq_len//step}")

            video_imgs = []
            video_depths = []

            if (ref_idx + step) <= seq_len:
                ref_e = ref_idx + step
            else:
                continue

            for idx in range(ref_idx, ref_e):
                im_path = osp.join(
                    root, seq_name, "color", all_img_names[idx]
                )
                depth_path = osp.join(
                    root, seq_name, "depth", all_img_names[idx][:-3] + "png"
                )
                pose_path = osp.join(
                    root, seq_name, "pose", all_img_names[idx][:-3] + "txt"
                )
                out_img_path = osp.join(
                    saved_dir, datatset_name, seq_name, "color", all_img_names[idx]
                )
                out_depth_path = osp.join(
                    saved_dir, datatset_name, seq_name, "depth", all_img_names[idx][:-3] + "png"
                )
                
                copy_crop_files(
                    im_path=im_path,
                    depth_path=depth_path,
                    out_img_path=out_img_path,
                    out_depth_path=out_depth_path,
                    dataset=datatset_name,
                )

                origin_img = np.array(Image.open(im_path))
                out_img_origin_path = osp.join(
                    saved_dir, datatset_name, seq_name, "color_origin", all_img_names[idx]
                )
                out_pose_path = osp.join(
                    saved_dir, datatset_name, seq_name, "pose", all_img_names[idx][:-3] + "txt"
                )
                
                os.makedirs(osp.dirname(out_img_origin_path), exist_ok=True)
                os.makedirs(osp.dirname(out_pose_path), exist_ok=True)

                cv2.imwrite(
                    out_img_origin_path,
                    origin_img,
                )
                shutil.copyfile(pose_path, out_pose_path)
            
            intrinsic_path = osp.join(
                root, seq_name, "intrinsic", "intrinsic_depth.txt"
            )
            out_intrinsic_path = osp.join(
                saved_dir, datatset_name, seq_name, "intrinsic", "intrinsic_depth.txt"
            )
            os.makedirs(osp.dirname(out_intrinsic_path), exist_ok=True)
            shutil.copyfile(intrinsic_path, out_intrinsic_path)

    # 90 frames like DepthCraft
    out_json_path = osp.join(saved_dir, datatset_name, "scannet_video.json")
    gen_json(
        root_path=osp.join(saved_dir, datatset_name), dataset=datatset_name,
        start_id=0,end_id=90*3,step=3,
        save_path=out_json_path,
    )      

    #~500 frames in paper
    out_json_path = osp.join(saved_dir, datatset_name, "scannet_video_500.json")    
    gen_json(
        root_path=osp.join(saved_dir, datatset_name), dataset=datatset_name,
        start_id=0,end_id=500,step=1,
        save_path=out_json_path,
    )

    # tae 
    out_json_path = osp.join(saved_dir, datatset_name, "scannet_video_tae.json")
    gen_json_scannet_tae(
        root_path=osp.join(saved_dir, datatset_name),
        start_id=0,end_id=192,step=1,
        save_path=out_json_path,
    )

if __name__ == "__main__":
    extract_scannet(
        root="path/to/scannet",
        saved_dir="./benchmark/datasets/",
        sample_len=-1,
        datatset_name="scannet",
    )