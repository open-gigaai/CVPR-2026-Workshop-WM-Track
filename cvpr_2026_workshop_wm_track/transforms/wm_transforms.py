import copy
import random

import numpy as np
import torch
from decord import VideoReader
from giga_datasets import video_utils
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from giga_train import TRANSFORMS
from decord import VideoReader
import decord
from PIL import Image
from typing import List, Tuple, Optional, Union
import re
import ftfy
import html
from transformers import AutoTokenizer, UMT5EncoderModel

from giga_datasets import Dataset, FileWriter, PklWriter, load_dataset
from giga_datasets import utils as gd_utils
import pandas as pd
from decord import VideoReader as DecordVideoReader
from torchvision.io import VideoReader as TorchVideoReader
from ..utils import resize_with_pad
import torch

from ..model_config import DEFAULT_PROMPT_EMBEDDING_PATH

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


def save_video_per_frame(images, path):
    import cv2
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(path, fourcc, 16, (width, height))
    for img in images:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)
    video_writer.release()

import os
import pickle
@TRANSFORMS.register
class WMTransforms:
    def __init__(
        self,
        is_train=False,
        dst_size=None,
        num_frames=1,
        fps=16,
        image_cfg=None,
        max_stride=None,
        sub_frames=1,
        num_views=1,
    ):
        self.fps = fps
        self.is_train = is_train
        self.normalize = transforms.Normalize([0.5], [0.5])
        self.dst_size = dst_size
        self.num_frames = num_frames
        self.image_cfg = image_cfg
        self.mask_generator = MaskGenerator(**image_cfg['mask_generator'])
        self.max_stride = max_stride
        self.sub_frames = sub_frames

        self.default_prompt_embeds = torch.load(DEFAULT_PROMPT_EMBEDDING_PATH)['prompt_embeds']
        self.num_views = num_views

    def __call__(self, data_dict):
        dst_width, dst_height = data_dict['video_info']
        if self.num_views == 1:
            if 'front_video_path' in data_dict:
                front_images = DecordVideoReader(data_dict['cam_high_video_path'])
                replay_images = DecordVideoReader(data_dict['cam_high_replay_path'])
                depth_images = DecordVideoReader(data_dict['cam_high_simulator_path'])
                video_legnth = len(front_images)
                assert len(front_images) == len(depth_images) == len(replay_images)
        elif self.num_views == 3:
            front_view_images = DecordVideoReader(data_dict['cam_high_video_path'])
            left_view_images = DecordVideoReader(data_dict['cam_left_wrist_video_path'])
            right_view_images = DecordVideoReader(data_dict['cam_right_wrist_video_path'])
            front_depth_images = DecordVideoReader(data_dict['cam_high_depth_path'])
            left_depth_images = DecordVideoReader(data_dict['cam_left_wrist_depth_path'])
            right_depth_images = DecordVideoReader(data_dict['cam_right_wrist_depth_path'])
            front_replay_images = DecordVideoReader(data_dict['cam_high_simulator_path'])
            left_replay_images = DecordVideoReader(data_dict['cam_left_wrist_simulator_path'])
            right_replay_images = DecordVideoReader(data_dict['cam_right_wrist_simulator_path'])
            assert len(front_view_images) == len(left_view_images) == len(right_view_images)
            assert len(front_replay_images) == len(left_replay_images) == len(right_replay_images)
            assert len(front_depth_images) == len(left_depth_images) == len(right_depth_images)

            assert len(front_view_images) == len(front_depth_images)
            video_legnth = len(front_view_images)


        if self.max_stride is not None:
            stride = random.randint(1, self.max_stride)
            start_frame = random.randint(0, max(0, video_legnth - stride * (self.num_frames - 1) - 1))
            end_frame = start_frame + stride * (self.num_frames - 1)
            end_frame = min(video_legnth-1, end_frame)
            sample_indexes = np.linspace(start_frame, end_frame, num=self.num_frames, dtype=int)
        else:
            sample_indexes = np.linspace(0, video_legnth - 1, self.num_frames, dtype=int)

        def get_input_images(video):
            if isinstance(video, VideoReader):
                input_images = video_utils.sample_video(video, sample_indexes, method=2)
            else:
                input_images = video[sample_indexes]
            input_images = resize_with_pad(input_images, 224, 224)
            data_dict['input_fps'] = self.fps
            input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous()
            input_images = input_images / 255.0
            input_images = self.normalize(input_images)
            return input_images

        if self.num_views == 1:
            data_dict['input_front_images'] = get_input_images(front_images)
            data_dict['input_depth_images'] = get_input_images(depth_images)
            data_dict['input_replay_images'] = get_input_images(replay_images)

        elif self.num_views == 3:
            front_images = get_input_images(front_view_images)
            left_images = get_input_images(left_view_images)
            right_images = get_input_images(right_view_images)
            data_dict['input_front_images'] = torch.cat([front_images, left_images, right_images], dim=-1)
            front_depth = get_input_images(front_depth_images)
            left_depth = get_input_images(left_depth_images)
            right_depth = get_input_images(right_depth_images)
            data_dict['input_depth_images'] = torch.cat([front_depth, left_depth, right_depth], dim=-1)
            front_replay = get_input_images(front_replay_images)
            left_replay = get_input_images(left_replay_images)
            right_replay = get_input_images(right_replay_images)
            data_dict['input_replay_images'] = torch.cat([front_replay, left_replay, right_replay], dim=-1)

        if self.image_cfg is not None:
            ref_masks, ref_latent_masks = self.mask_generator.get_mask(data_dict['input_front_images'].shape[0])
            ref_masks = ref_masks[:, None, None, None]
            ref_latent_masks = ref_latent_masks[None, :, None, None]
            ref_images = copy.deepcopy(data_dict['input_front_images'])
            ref_images = ref_images * ref_masks
            data_dict['input_front_ref_images'] = ref_images
            data_dict['input_front_ref_masks'] = ref_latent_masks

        if self.is_train:
            new_data_dict = {}
            if 'input_fps' in data_dict:
                new_data_dict['fps'] = data_dict['input_fps']
            if 'input_front_images' in data_dict:
                new_data_dict['front_images'] = data_dict['input_front_images']
            if 'input_front_ref_images' in data_dict:
                new_data_dict['front_ref_images'] = data_dict['input_front_ref_images']
                new_data_dict['front_ref_masks'] = data_dict['input_front_ref_masks']
            if 'input_depth_images' in data_dict:
                new_data_dict['depth'] = data_dict['input_depth_images']
                new_data_dict['replay'] = data_dict['input_replay_images']
                new_data_dict['prompt_embeds'] = self.default_prompt_embeds

        else:
            assert False
        keys = list(new_data_dict.keys())
        for key in keys:
            if new_data_dict[key] is None:
                new_data_dict.pop(key)
        return new_data_dict

class MaskGenerator:
    def __init__(self, max_ref_frames, factor=8, start=1):
        assert max_ref_frames > 0 and (max_ref_frames - 1) % factor == 0
        self.max_ref_frames = max_ref_frames
        self.factor = factor
        self.start = start
        self.max_ref_latents = 1 + (max_ref_frames - 1) // factor
        assert self.start <= self.max_ref_latents

    def get_mask(self, num_frames):
        assert num_frames > 0 and (num_frames - 1) % self.factor == 0 and num_frames >= self.max_ref_frames
        num_latents = 1 + (num_frames - 1) // self.factor
        num_ref_latents = random.randint(self.start, self.max_ref_latents)
        if num_ref_latents > 0:
            num_ref_frames = 1 + (num_ref_latents - 1) * self.factor
        else:
            num_ref_frames = 0
        ref_masks = torch.zeros((num_frames,), dtype=torch.float32)
        ref_masks[:num_ref_frames] = 1
        ref_latent_masks = torch.zeros((num_latents,), dtype=torch.float32)
        ref_latent_masks[:num_ref_latents] = 1
        return ref_masks, ref_latent_masks