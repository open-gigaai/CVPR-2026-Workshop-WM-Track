import copy
import functools
import os
import random
import sys
import torch
from diffusers.models import AutoencoderKLWan
from ..models import WanConditionModel
from einops import rearrange
from giga_models import utils as gm_utils

from giga_train import Trainer, ModuleDict
import torch.nn as nn
import einops
from PIL import Image
import numpy as np
from diffusers.video_processor import VideoProcessor
from giga_datasets import image_utils, video_utils
from diffusers.schedulers import UniPCMultistepScheduler
import random
from torch.nn import functional as F
import imageio
from ..pipelines import get_video_depth_anything


class CWRolloutMixedTrainerV2(Trainer):
    def get_models(self, model_config):
        pretrained = gm_utils.get_model_path(model_config.pretrained)
        self.flow_shift = model_config.flow_shift
        self.ref_aug_strength = 0.1
        self.expand_timesteps = model_config.get("expand_timesteps", False)
        self.view_interval = 100
        self.view_dir = model_config.view_dir
        self.sub_frames = model_config.sub_frames
        self.rollout_step = model_config.rollout
        self.timestep_scale = 1000
        model = dict()
        # vae
        vae_pretrained = model_config.get('vae_pretrained', os.path.join(pretrained, 'vae'))
        vae_dtype = model.get('vae_dtype', self.dtype)
        vae = AutoencoderKLWan.from_pretrained(vae_pretrained)
        vae.requires_grad_(False)
        vae.to(self.device, dtype=vae_dtype)
        self.vae = vae
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, dtype=vae_dtype)
        self.latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(self.device, dtype=vae_dtype)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        # transformer
        transformer_pretrained = model_config.transformer_model_path if getattr(model_config, "transformer", None) else os.path.join(pretrained, 'transformer')
        if model_config.get("unpretrain", False):
            print("Load unet from config only.")
            transformer = WanConditionModel.from_config(transformer_pretrained, torch_dtype=self.dtype)
        else:
            transformer = WanConditionModel.from_pretrained(transformer_pretrained, torch_dtype=self.dtype)
            transformer.depth_embedding = copy.deepcopy(transformer.patch_embedding)
            transformer.replay_embedding = copy.deepcopy(transformer.patch_embedding)
        model.update(transformer=transformer)
        # model
        checkpoint = model_config.get('checkpoint', None)
        strict = model_config.get('strict', True)
        self.load_checkpoint(checkpoint, list(model.values()), strict=strict)
        model = ModuleDict(model)
        model.train()

        self.video_depth_anything = get_video_depth_anything(self.device)
        return model

    @property
    def transformer(self):
        return functools.partial(self.model, 'transformer')

    def prepare_conditioning(self, batch_dict):
        condition = dict()
        front_images = batch_dict['front_images']
        depth = batch_dict['depth']
        replay = batch_dict['replay']
        # inputs
        front_latents = self.forward_vae(front_images)
        depth_latents = self.forward_vae(depth)
        replay_latents = self.forward_vae(replay)
        latents = front_latents
        num_ref_images = (torch.sum(batch_dict['front_ref_masks']).int() - 1).item() * 4 + 1
        num_ref_latent_frames = (num_ref_images - 1) // self.vae_scale_factor_temporal + 1
        num_latent_frames = latents.shape[2]
        latent_height = latents.shape[-2]
        latent_width = latents.shape[-1]
        first_frame_mask = torch.ones(
            1, 1, num_latent_frames, latent_height, latent_width, dtype=latents.dtype, device=latents.device
        )
        first_frame_mask[:, :, :num_ref_latent_frames] = 0
        front_ref_images = batch_dict['front_ref_images'][:, :num_ref_images]
        front_ref_latents = self.forward_vae(front_ref_images)
        ref_latents = front_ref_latents

        prompt_embeds = batch_dict['prompt_embeds']

        condition['ref_latents'] = ref_latents
        condition['replay_latents'] = replay_latents
        condition['first_frame_mask'] = first_frame_mask
        condition['x0'] = latents
        condition['depth_latents'] = depth_latents
        condition['prompt_embeds'] = prompt_embeds.to(dtype=latents.dtype, device=latents.device)
        return condition

    def denoise_net(self, transformer, xt, sigma, condition, add_ref_aug=False, return_x0=False):
        depth_latents = condition['depth_latents']
        t = sigma * self.timestep_scale
        ref_latents = condition['ref_latents']
        first_frame_mask = condition['first_frame_mask']
        prompt_embeds = condition['prompt_embeds']
        replay_latents = condition['replay_latents']
        if add_ref_aug:
            noisy_ref_latents = torch.randn_like(ref_latents)
            aug_noise = random.random() * self.ref_aug_strength
            ref_latents = ref_latents + aug_noise * noisy_ref_latents
        input_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * xt
        temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
        timestep = temp_ts.unsqueeze(0).expand(xt.shape[0], -1)
        input_noisy_latents = input_noisy_latents.to(self.dtype)
        model_pred = transformer(
            hidden_states=input_noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            depth_latents=depth_latents,
            replay_latents=replay_latents,
        )[0]
        if return_x0:
            pred_x0 = xt - model_pred * sigma
            pred_x0 = self.repare_first_frame(pred_x0, condition)
            return model_pred, pred_x0
        else:
            return model_pred, None

    def rollout(self, batch_dict, id):
        transformer = self.transformer
        condition = self.prepare_conditioning(batch_dict)
        latents = condition['x0']
        depth_latents = condition['depth_latents']
        replay_latents = condition['replay_latents']
        self.vae_decode(latents, sign=f'input_rollout_step_{id}')
        self.vae_decode(depth_latents, sign=f'condition_step_{id}')
        self.vae_decode(replay_latents, sign=f'replay_step_{id}')
        timestep, sigma = self.get_timestep_and_sigma(latents.shape[0], latents.ndim)
        noise = torch.randn_like(latents)
        target = noise - latents
        noisy_latents = noise * sigma + latents * (1 - sigma)
        model_pred, pred_x0 = self.denoise_net(transformer, noisy_latents, sigma, condition,
                                               add_ref_aug=True, return_x0=True)
        pred_x0 = pred_x0.detach()
        # ignore first frame loss
        loss = ((model_pred.float() - target.float()) * condition['first_frame_mask'])  ** 2
        return loss, pred_x0

    def predict_depth(self, images):
        fps = 1
        depths, fps = self.video_depth_anything.infer_video_depth(images, fps, input_size=640,
                                                                  device=str(self.device),
                                                                  fp32=False)
        d_min, d_max = depths.min(), depths.max()
        depth_norm = ((depths - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        return depth_norm

    def forward_step(self, batch_dict):
        sub_latents = self.sub_frames // self.vae_scale_factor_temporal + 1
        front_ref_masks = batch_dict['front_ref_masks'][:, :, :sub_latents]
        front_ref_images = batch_dict['front_ref_images'][:, :self.sub_frames + 1]
        front_images = batch_dict['front_images']
        depth = batch_dict['depth']
        replay = batch_dict['replay']
        first_depth = depth[:, :1]
        prompt_embeds = batch_dict['prompt_embeds']
        assert depth.shape == front_images.shape
        num_ref_images = (torch.sum(batch_dict['front_ref_masks']).int() - 1).item() * 4 + 1
        loss_dict = {}
        loss_weight_per_roll = 1.0 / self.rollout_step
        for i in range(self.rollout_step):
            roll_dict = dict()
            roll_dict['front_ref_masks'] = front_ref_masks
            roll_dict['front_ref_images'] = front_ref_images
            roll_dict['prompt_embeds'] = prompt_embeds
            start_frame = i * self.sub_frames
            end_frame = (i + 1) * self.sub_frames + 1
            roll_dict['front_images'] = front_images[:, start_frame:end_frame]
            # get condition
            current_replay = replay[:, start_frame:end_frame]
            roll_dict['replay'] = current_replay
            roll_dict['depth'] = first_depth.repeat(1, self.sub_frames + 1, 1, 1, 1)
            loss, pred_x0 = self.rollout(roll_dict, i)
            loss_dict[f'roll_{i}'] = loss * loss_weight_per_roll
            # decode pred_x0
            with torch.no_grad():
                tensor_video = self.vae_decode(latents=pred_x0, sign=f'rollout_step_{i}', return_tensor=True)
                front_ref_images[:, :num_ref_images] = tensor_video.transpose(1,2)[:, -num_ref_images:]
                end_frame_pixel = tensor_video[:, :, -num_ref_images:]
                end_frame_pixel = np.array(self.video_processor.postprocess_video(end_frame_pixel, output_type='pil')[0])
                end_depth = self.predict_depth(end_frame_pixel)
                gray = end_depth.squeeze(0)  # 变成 (480, 640)
                rgb = np.stack([gray] * 3, axis=-1)  # 变成 (480, 640, 3)
                # 转为 PIL Image（可选）
                img = Image.fromarray(rgb)
                depth_tensor = self.video_processor.preprocess_video(img).transpose(1,2)
                if random.random() > 0.5:
                    first_depth = depth_tensor.to(self.device)
                else:
                    first_depth = depth[:, end_frame-1:end_frame]
        return loss_dict

    def repare_first_frame(self, latents, condition):
        if self.expand_timesteps:
            first_frame_mask = condition['first_frame_mask']
            ref_latents = condition['ref_latents']
            latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * latents
        return latents

    def forward_vae(self, images):
        images = images.to(self.vae.dtype)
        with torch.no_grad():
            images = rearrange(images, 'b t c h w -> b c t h w')
            latents = self.vae.encode(images).latent_dist.mode()
        latents = (latents - self.latents_mean) * self.latents_std
        return latents

    def get_timestep_and_sigma(self, batch_size, ndim):
        sigma = torch.rand(batch_size).to(self.device)
        # flow_shift: 5.0 for 720P, 3.0 for 480P
        sigma = self.flow_shift * sigma / (1 + (self.flow_shift - 1) * sigma)
        timestep = torch.round(sigma * 1000).long()
        sigma = timestep.float() / 1000
        while len(sigma.shape) < ndim:
            sigma = sigma.unsqueeze(-1)
        return timestep, sigma

    def vae_decode(self, latents=None, images=None, sign=None, return_tensor=False):
        if self.process_index == 0 and self.cur_step % self.view_interval == 0 or self.cur_step == 1 and len(self._outputs) == 0:
            save_dir = os.path.join(self.view_dir, "images", "{}".format(self.cur_step))
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "{}.mp4".format(sign))
            if latents is not None:
                latents = latents.to(self.vae.dtype)
                latents = latents / self.latents_std + self.latents_mean
                with torch.no_grad():
                    tensor_video = self.vae.decode(latents, return_dict=False)[0].detach()
                video = self.video_processor.postprocess_video(tensor_video, output_type='pil')[0]
                vis_images = video
                imageio.mimsave(save_path, vis_images, fps=16)
                if return_tensor:
                    return tensor_video
                return vis_images
            if images is not None:
                image_tensor = images
                # [T, 3, H, W] to video
                images = (images + 1.0) / 2.0 * 255
                images = images.astype(np.uint8)
                images = [Image.fromarray(images[i]) for i in range(images.shape[0])]
                imageio.mimsave(save_path, images, fps=16)
                return image_tensor
        else:
            latents = latents.to(self.vae.dtype)
            latents = latents / self.latents_std + self.latents_mean
            with torch.no_grad():
                tensor_video = self.vae.decode(latents, return_dict=False)[0].detach()
            return tensor_video




def process_transformer(transformer, transformer_cfg):
    in_channels = transformer_cfg.get('in_channels', transformer.config.in_channels)
    if transformer.config.in_channels != in_channels:
        assert False
    num_checkpointing = transformer_cfg.get('num_checkpointing', None)
    if num_checkpointing is not None:
        transformer.enable_gradient_checkpointing()
        transformer.num_checkpointing = num_checkpointing
    return transformer
