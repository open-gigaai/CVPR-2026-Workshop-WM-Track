import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import argparse
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import os
import gc
import time
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_errors_torch(gt, pred):
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    return abs_rel
    
def get_infer(infer_path,args, target_size = None):
    if infer_path.split('.')[-1] == 'npy':
        img_gray = np.load(infer_path)
        img_gray = img_gray.astype(np.float32)
        infer_factor = 1.0
    else: 
        img = cv2.imread(infer_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = img_gray.astype(np.float32)
        infer_factor = 1.0 / 255.0

    infer = img_gray / infer_factor
    if args.hard_crop:
        infer = infer[args.a:args.b, args.c:args.d]
    
    if target_size is not None:
        if infer.shape[0] != target_size[0] or infer.shape[1] != target_size[1]:
            infer = cv2.resize(infer, (target_size[1], target_size[0]))
    return infer

def get_gt(depth_gt_path, gt_factor, args):
    if depth_gt_path.split('.')[-1] == 'npy':
        depth_gt = np.load(depth_gt_path)
    else:
        depth_gt = cv2.imread(depth_gt_path, -1)
        depth_gt = np.array(depth_gt)
    depth_gt = depth_gt / gt_factor
    
    depth_gt[depth_gt==0] = 0
    return depth_gt

def depth2disparity(depth, return_mask=False):
    if isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

def tae_torch(depth1, depth2, R_2_1, T_2_1, K, mask):
    H, W = depth1.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Generate meshgrid
    xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H))
    xx, yy = xx.t(), yy.t()  # Transpose to match the shape (H, W)

    # Convert meshgrid to tensor
    xx = xx.to(dtype=depth1.dtype, device=depth1.device)
    yy = yy.to(dtype=depth1.dtype, device=depth1.device)
    # Calculate 3D points in frame 1
    X = (xx - cx) * depth1 / fx
    Y = (yy - cy) * depth1 / fy
    Z = depth1
    points3d = torch.stack((X.flatten(), Y.flatten(), Z.flatten()), dim=1)  # Shape (H*W, 3)
    T = torch.tensor(T_2_1, dtype=depth1.dtype, device=depth1.device)

    # Transform 3D points to frame 2
    points3d_transformed = torch.matmul(points3d, R_2_1.T) + T
    X_world, Y_world, Z_world = points3d_transformed[:, 0], points3d_transformed[:, 1], points3d_transformed[:, 2]
    # Project 3D points to 2D plane using intrinsic matrix
    X_plane = (X_world * fx) / Z_world + cx
    Y_plane = (Y_world * fy) / Z_world + cy

    # Round and convert to integers
    X_plane = torch.round(X_plane).to(dtype=torch.long)
    Y_plane = torch.round(Y_plane).to(dtype=torch.long)

    # Filter valid indices
    valid_mask = (X_plane >= 0) & (X_plane < W) & (Y_plane >= 0) & (Y_plane < H)
    if valid_mask.sum() == 0:
        return 0

    depth_proj = torch.zeros((H, W), dtype=depth1.dtype, device=depth1.device)

    valid_X = X_plane[valid_mask]
    valid_Y = Y_plane[valid_mask]
    valid_Z = Z_world[valid_mask]

    depth_proj[valid_Y, valid_X] = valid_Z

    valid_mask = (depth_proj > 0) & (depth2 > 0) & (mask)
    if valid_mask.sum() == 0:
        return 0
    abs_errors = compute_errors_torch(depth2[valid_mask], depth_proj[valid_mask])
    
    return abs_errors

def eval_TAE(infer_paths, depth_gt_paths, factors, masks, Ks, poses, args):
    gts = []
    infs = []
    dataset_max_depth = args.max_depth_eval
    gt_paths_cur = []
    Ks_cur = []
    poses_cur = []
    masks_cur = []
    
    for i in range(len(infer_paths)):
        # DAV missing some frames
        if not os.path.exists(infer_paths[i]):
            continue
        
        depth_gt = get_gt(depth_gt_paths[i], factors[i], args)
        depth_gt = depth_gt[args.a:args.b, args.c:args.d]
        
        gt_paths_cur.append(depth_gt_paths[i])
        infer = get_infer(infer_paths[i], args, target_size=depth_gt.shape)

        gts.append(depth_gt)
        infs.append(infer)
        Ks_cur.append(Ks[i])
        poses_cur.append(poses[i])
        if args.mask:
            masks_cur.append(masks[i])
    
    gts = np.stack(gts, axis=0)
    infs = np.stack(infs, axis=0)

    valid_mask = np.logical_and((gts>1e-3), (gts<dataset_max_depth))
    
    gt_disp_masked = 1. / (gts[valid_mask].reshape((-1,1)).astype(np.float64) + 1e-8)
    infs = np.clip(infs, a_min=1e-3, a_max=None)
    pred_disp_masked = infs[valid_mask].reshape((-1,1)).astype(np.float64)

    _ones = np.ones_like(pred_disp_masked)
    A = np.concatenate([pred_disp_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_disp_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = scale * infs + shift
    aligned_pred = np.clip(aligned_pred, a_min=1e-3, a_max=None)

    pred_depth = depth2disparity(aligned_pred)
    gt_depth = gts
    pred_depth = np.clip(
            pred_depth, a_min=1e-3, a_max=dataset_max_depth
        )
    
    error_sum = 0.
    for i in range(len(gt_paths_cur) -1):
        depth1 = pred_depth[i]
        depth2 = pred_depth[i+1]

        gt_depth1 = gt_paths_cur[i]
        gt_depth2 = gt_paths_cur[i+1]
        T_1 = poses_cur[i]
        T_2 = poses_cur[i+1]

        T_2_1 = np.linalg.inv(T_2) @ T_1
        
        R_2_1 = T_2_1[:3,:3]
        t_2_1 = T_2_1[:3, 3]
        K = Ks_cur[i]
        
        if args.mask:
            mask_path1 = masks_cur[i]
            mask_path2 = masks_cur[i+1]
            mask1 = cv2.imread(mask_path1, -1)
            mask2 = cv2.imread(mask_path2, -1)
            mask1 = mask1[args.a:args.b, args.c:args.d]
            if mask2 is None:
                mask2 = np.ones_like(mask1)
            else:
                mask2 = mask2[args.a:args.b, args.c:args.d]

            mask1 = mask1 > 0
            mask2 = mask2 > 0
        else:
            mask1 = np.ones_like(depth1)
            mask2 = np.ones_like(depth2)

            mask1 = mask1 > 0
            mask2 = mask2 > 0
        
        depth1 = torch.from_numpy(depth1).to(device=device)
        depth2 = torch.from_numpy(depth2).to(device=device)
        R_2_1 = torch.from_numpy(R_2_1).to(device=device)
        t_2_1 = torch.from_numpy(t_2_1).to(device=device)
        mask1 = torch.from_numpy(mask1).to(device=device)
        mask2 = torch.from_numpy(mask2).to(device=device)

        error1 = tae_torch(depth1, depth2, R_2_1, t_2_1, K, mask2)
        T_1_2 = np.linalg.inv(T_2_1)
        R_1_2 = T_1_2[:3,:3]
        t_1_2 = T_1_2[:3, 3]

        R_1_2 = torch.from_numpy(R_1_2).to(device=device)
        t_1_2 = torch.from_numpy(t_1_2).to(device=device)

        error2 = tae_torch(depth2, depth1, R_1_2, t_1_2, K, mask1)
        
        error_sum += error1
        error_sum += error2
    
    gc.collect()
    result = error_sum / (2 * (len(gt_paths_cur) -1))
    return result*100


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='')
    parser.add_argument('--benchmark_path', type=str, default='')

    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet', 'sintel'])
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=180)
    parser.add_argument('--eval_scenes_num', type=int, default=20)
    parser.add_argument('--hard_crop', action='store_true', default=False)

    args = parser.parse_args()

    results_save_path = os.path.join(args.infer_path, 'results.txt')

    for dataset in args.datasets:

        file = open(results_save_path, 'a')
        if dataset == 'scannet':
            args.json_file = os.path.join(args.benchmark_path,'scannet/scannet_video.json')
            args.root_path = os.path.join(args.benchmark_path, 'scannet/')
            args.max_depth_eval = 10.0
            args.min_depth_eval = 0.1
            args.max_eval_len = 200
            args.mask = False
            #DepthCrafer crop
            args.a = 8
            args.b = -8
            args.c = 11
            args.d = -11
        
        with open(args.json_file, 'r') as fs:
            path_json = json.load(fs)
        
        json_data = path_json[dataset]
        count = 0
        line = '-' * 50
        print(f'<{line} {dataset} start {line}>')
        file.write(f'<{line} {dataset} start {line}>\n')
        results_all = 0.

        for data in tqdm(json_data[:args.eval_scenes_num]):
            for scene_name in data.keys():
                value = data[scene_name]
                infer_paths = []
                depth_gt_paths = []
                factors = []
                Ks = []
                poses = []
                masks = []
                for images in value:
                    infer_path = (args.infer_path + '/'+ dataset + '/' + images['image']).replace('.jpg', '.npy').replace('.png', '.npy')
                    
                    infer_paths.append(infer_path)
                    depth_gt_paths.append(args.root_path + '/' + images['gt_depth'])
                    factors.append(images['factor'])
                    Ks.append(np.array(images['K']))
                    poses.append(np.array(images['pose']))
                    
                    if args.mask:
                        masks.append(args.root_path + '/' + images['mask'])
            
            infer_paths = infer_paths[args.start_idx:args.end_idx]
            depth_gt_paths = depth_gt_paths[args.start_idx:args.end_idx]
            factors = factors[args.start_idx:args.end_idx]
            poses = poses[args.start_idx:args.end_idx]
            Ks = Ks[args.start_idx:args.end_idx]
            error = eval_TAE(infer_paths, depth_gt_paths, factors,masks,Ks,poses,args)
            results_all += error
            count += 1

        print(dataset,': ','tae ', results_all / count)
        file.write(f'{dataset}: {results_all / count}\n')
        file.write(f'<{line} {dataset} finish {line}>\n')


