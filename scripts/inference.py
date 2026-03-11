import paths
from cvpr_2026_workshop_wm_track.pipelines import BaselineWMPipeline
from cvpr_2026_workshop_wm_track.model_config import model_config, DATA_DIR
from diffusers import AutoencoderKLWan, WanPipeline
import torch
from cvpr_2026_workshop_wm_track.models import WanConditionModel
from cvpr_2026_workshop_wm_track.utils import resize_with_pad, split_data
from cvpr_2026_workshop_wm_track.image_utils import concat_images_grid
from PIL import Image
import numpy as np
import argparse
import multiprocessing 
import os
import glob
from multiprocessing import Process


class InferenceEngine:
    def __init__(self, transformer_model_path, device, dtype=torch.bfloat16, num_views=3, mode='offline', seed=1024):
        print(f"Loading model from {transformer_model_path}")
        model_id = model_config['wan2.2-5b-diffusers']
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
        transformer = WanConditionModel.from_pretrained(transformer_model_path).to(dtype)
        self.pipe = BaselineWMPipeline.from_pretrained(model_id, vae=vae, transformer=transformer, torch_dtype=dtype)
        self.pipe.to(device)

        self.dst_size = (224, 224)
        self.wm_frame_per_time = 8 # sequence [结构序]
        self.num_inference_steps = 30 # 16fps
        self.guidance_scale = 0.0
        self.num_views = num_views
        self.generator = torch.Generator(device=device).manual_seed(seed)

    
    def wm_inference_per_time(self, replay, depth, ref_image):
        output_images = self.pipe(
                replay=replay,
                depth=depth, # [T (1 + 8), 14]
                height=self.dst_size[1],
                width=self.dst_size[0] * self.num_views,
                num_frames=self.wm_frame_per_time + 1,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                image=ref_image,
                generator=self.generator,
                output_type='pil',
                prompt=''
            ).frames[0]
        return output_images
    
    def resize_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = Image.fromarray(resize_with_pad(image, self.dst_size[0], self.dst_size[1]))
        return image

    def resize_images(self, images):
        images = [self.resize_image(image) for image in images]
        return images
    
    def get_depth_image(self, image):
        depths, fps = self.video_depth_anything.infer_video_depth(np.array([image]), 1, input_size=256,
                                                                    device='cuda',
                                                                    fp32=False)
        d_min, d_max = depths.min(), depths.max()
        depth_norm = ((depths - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        gray = depth_norm.squeeze(0)  # 变成 (480, 640)
        rgb = np.stack([gray] * 3, axis=-1)  # 变成 (480, 640, 3)
        # 转为 PIL Image（可选）
        first_depth = Image.fromarray(rgb)
        return first_depth
    
    def get_three_view_depth_images(self, img_front, img_left, img_right):
        front_ref_depth = self.get_depth_image(img_front)
        left_ref_depth = self.get_depth_image(img_left)
        right_ref_depth = self.get_depth_image(img_right)
        first_depth = concat_images_grid(
            [front_ref_depth, left_ref_depth, right_ref_depth], cols=3, pad=0
        )
        return first_depth

    def wm_inference(self, ref_images, action_images, ref_depth=None):
        img_front = ref_images['front']
        img_left = ref_images['left']
        img_right = ref_images['right']
        img_front = self.resize_image(img_front)
        img_left = self.resize_image(img_left)
        img_right = self.resize_image(img_right)
        if ref_depth is None:
            ref_depth = self.get_three_view_depth_images(img_front, img_left, img_right)
        action_chunk = len(action_images)
        wm_inference_time = (action_chunk - 1) // self.wm_frame_per_time
        if action_chunk % self.wm_frame_per_time != 0:
            Warning(f"action_chunk {action_chunk} is not divisible by wm_frame_per_time {self.wm_frame_per_time}")
        all_output_images = []
        depth_condition_images = []
        replay_condition_images = []
        for step in range(wm_inference_time):
            start = step * self.wm_frame_per_time
            end = (step + 1) * self.wm_frame_per_time + 1
            action_images_chunk = action_images[start:end]
            ref_depth_chunk = [ref_depth] * (self.wm_frame_per_time + 1)
            output_images = self.wm_inference_per_time(action_images_chunk, ref_depth_chunk, img_front)
            if step == wm_inference_time - 1:
                output_images = output_images
                depth_condition_images.extend(ref_depth_chunk)
                replay_condition_images.extend(action_images_chunk)
            else:
                output_images = output_images[:-1]
                depth_condition_images.extend(ref_depth_chunk[:-1])
                replay_condition_images.extend(action_images_chunk[:-1])
            all_output_images.extend(output_images)
        condition_images_dict = {
            'depth': depth_condition_images,
            'replay': replay_condition_images,
        }
        return all_output_images, condition_images_dict


def inference(args, device, world_size, rank):
    mode = args.mode
    eval_data_dir = os.path.join(args.data_dir, args.task)
    if mode == 'offline':
        eval_data_dir = os.path.join(args.data_dir, args.task, 'video_quality')
    elif mode == 'online':
        eval_data_dir = os.path.join(args.data_dir, args.task, 'evaluator')
    episode_list = os.listdir(eval_data_dir)
    breakpoint()
        
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_model_path', type=str, default='wan2.2-5b-diffusers')
    parser.add_argument('--device_list', type=str, default='0,1,2,3')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--mode', type=str, default='offline')
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--task", type=str, default='task4')
    args = parser.parse_args()

    inference(args, "cuda:0", 1, 0)
    exit()

    devices = args.device_list.split(',')
    multiprocessing.set_start_method('spawn')
    process_list = []
    gpu_ids = devices
    world_size = len(gpu_ids)
    for i in range(world_size):
        device = f'cuda:{gpu_ids[i]}'
        rank = i
        process = Process(target=inference, args=(args, device, world_size, rank))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()
    
    print("Inference done")





        
