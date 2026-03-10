import argparse
import os
import cv2
import json
import torch
from tqdm import tqdm
import numpy as np

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_path', type=str, default='')
    
    parser.add_argument('--json_file', type=str, default='')
    parser.add_argument('--datasets', type=str, nargs='+', default=['scannet', 'nyuv2'])
    
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

    args = parser.parse_args()
   
    for dataset in args.datasets:

        with open(args.json_file, 'r') as fs:
            path_json = json.load(fs)

        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
        video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
        video_depth_anything = video_depth_anything.to(DEVICE).eval()
        
        json_data = path_json[dataset]
        root_path = os.path.dirname(args.json_file)
        for data in tqdm(json_data):
             for key in data.keys():
                value = data[key]
                infer_paths = []
                
                videos = []
                for images in value:
                    
                    image_path = os.path.join(root_path, images['image'])
                    infer_path = (args.infer_path + '/'+ dataset + '/' + images['image']).replace('.jpg', '.npy').replace('.png', '.npy')
                    infer_paths.append(infer_path)
                    
                    img = cv2.imread(image_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    videos.append(img)
                videos = np.stack(videos, axis=0)
                target_fps=1
                depths, fps = video_depth_anything.infer_video_depth(videos, target_fps, input_size=args.input_size, device=DEVICE, fp32=True)

                for i in range(len(infer_paths)):
                    infer_path = infer_paths[i]
                    os.makedirs(os.path.dirname(infer_path), exist_ok=True)
                    depth = depths[i]
                    np.save(infer_path, depth)
                    
