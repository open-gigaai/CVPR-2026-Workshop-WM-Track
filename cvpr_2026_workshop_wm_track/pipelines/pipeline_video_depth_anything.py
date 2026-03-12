from video_depth_anything.video_depth import VideoDepthAnything
import torch
from ..model_config import model_config

model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }


def get_video_depth_anything(device, metric=False):
    checkpoint_name = 'metric_video_depth_anything' if metric else 'video_depth_anything'
    encoder = 'vitl'
    video_depth_anything = VideoDepthAnything(**model_configs[encoder], metric=metric)
    video_depth_model_dir = model_config['video-depth-anything']
    video_depth_anything.load_state_dict(
        torch.load(
            # f'/shared_disk/users/yukun.zhou/models/Video-Depth-Anything/{checkpoint_name}_{encoder}.pth',
            f'{video_depth_model_dir}/{checkpoint_name}_{encoder}.pth',
            map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(device).eval()
    return video_depth_anything