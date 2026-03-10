import os
import numpy as np
import os.path as osp
import json
import glob
import cv2
import shutil
from PIL import Image
from natsort import natsorted

def even_or_odd(num):
    if num % 2 == 0:
        return num
    else:
        return num - 1


def gen_json(root_path, dataset, start_id, end_id, step, save_path=None):
    rgb_name = "rgb"
    if dataset == "kitti":
        factor = 256.0
    elif dataset == "nyuv2":
        factor = 6000.0
    elif dataset == "bonn":
        factor = 5000.0
    elif dataset == 'sintel':
        factor = 65535 / 650
        rgb_name = "clean"
    elif dataset == 'scannet':
        factor = 1000.0
        rgb_name = "color"
    else:
        raise NotImplementedError
    
    data = {}
    data[dataset] = []
    pieces  = glob.glob(osp.join(root_path, "*"))
    count = 0
    for piece in pieces:
        if not osp.isdir(piece):
            continue
        name = piece.split('/')[-1]
        name_dict = {name:[]}
        images = glob.glob(osp.join(piece, rgb_name, "*.png")) + glob.glob(osp.join(piece, rgb_name, "*.jpg"))
        images = natsorted(images)
        depths = glob.glob(osp.join(piece, "depth/*.png"))
        depths = natsorted(depths)
        images = images[start_id:end_id:step]
        depths = depths[start_id:end_id:step]
        
        for i in range(len(images)):
            image = images[i]
            xx = image[len(root_path)+1:]
            depth = depths[i][len(root_path)+1:]
            tmp = {}
            tmp["image"] = xx
            tmp["gt_depth"] = depth
            tmp["factor"] = factor
            name_dict[name].append(tmp)
        data[dataset].append(name_dict)
    with open(save_path, "w") as f:
        json.dump(data, f, indent= 4)  


def gen_json_scannet_tae(root_path, start_id, end_id, step, save_path=None):
    data = {}
    data["scannet"] = []
    pieces  = glob.glob(osp.join(root_path, "*"))

    color =  'color_origin'

    for piece in pieces:
        if not osp.isdir(piece):
            continue
        name = piece.split('/')[-1]
        name_dict = {name:[]}
        images = glob.glob(osp.join(piece,color, "*.jpg"))
        images = natsorted(images)
        depths = glob.glob(osp.join(piece, "depth/*.png"))
        depths = natsorted(depths)
        images = images[start_id:end_id:step]
        depths = depths[start_id:end_id:step]
        print(f"sequence frame number: {piece}")
        count = 0
        for i in range(len(images)):
            image = images[i]
            xx = image[len(root_path)+1:]
            depth = depths[i][len(root_path)+1:]
            
            base_path = osp.dirname(image)
            base_path = base_path.replace(color, 'intrinsic')
            K = np.loadtxt(base_path + '/intrinsic_depth.txt')

            pose_path = image.replace(color, 'pose').replace('.jpg', '.txt')
            pose = np.loadtxt(pose_path)
            
            tmp = {}
            tmp["image"] = xx
            tmp["gt_depth"] = depth
            tmp["factor"] = 1000.0
            tmp["K"] = K.tolist()
            tmp["pose"] = pose.tolist()
            name_dict[name].append(tmp)
        data["scannet"].append(name_dict)
        
    with open(save_path, "w") as f:
        json.dump(data, f, indent= 4) 


def get_sorted_files(root_path, suffix):
    all_img_names = os.listdir(root_path)
    all_img_names = [x for x in all_img_names if x.endswith(suffix)]
    print(f"sequence frame number: {len(all_img_names)}")

    all_img_names.sort()
    all_img_names = sorted(all_img_names, key=lambda x: int(x.split(".")[0][-4:]))

    return all_img_names

def copy_crop_files(im_path, depth_path, out_img_path, out_depth_path, dataset):
    img = np.array(Image.open(im_path))
    
    if dataset == "kitti" or dataset == "bonn":
        height, width = img.shape[:2]
        height = even_or_odd(height)
        width = even_or_odd(width)
        img = img[:height, :width]
    elif dataset == "nyuv2":
        img = img[45:471, 41:601, :]
    elif dataset == "scannet":
        img = img[8:-8, 11:-11, :]
    
    os.makedirs(osp.dirname(out_img_path), exist_ok=True)
    os.makedirs(osp.dirname(out_depth_path), exist_ok=True)
    cv2.imwrite(
        out_img_path,
        img,
    )
    shutil.copyfile(depth_path, out_depth_path)
    