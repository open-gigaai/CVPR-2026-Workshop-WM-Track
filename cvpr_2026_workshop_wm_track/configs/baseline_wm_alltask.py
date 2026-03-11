from ..model_config import model_config, DATA_DIR

dst_size = (224, 224)
num_frames = 8
rollout = 4
total_frames = num_frames * rollout + 1
fps = 16
project_dir="experiments/baseline_wm/alltask/"
config = dict(
    project_dir=project_dir,
    runners=["cvpr_2026_workshop_wm_track.trainer.BaselineWMTrainer"],
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        # gpu_ids=[0,],
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file='accelerate_configs/zero2.json',
        ),
        until_completion=True,
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[
                f"{DATA_DIR}/task1/train",
                f"{DATA_DIR}/task2/train",
                f"{DATA_DIR}/task3/train",
                f"{DATA_DIR}/task4/train",
                f"{DATA_DIR}/task5/train",
                f"{DATA_DIR}/task6/train",
                f"{DATA_DIR}/task7/train",
                f"{DATA_DIR}/task8/train",
            ],
            batch_size_per_gpu=1,
            num_workers=8,
            # num_workers=0,
            filter=dict(
                mode='overall_func',
                func='cvpr_2026_workshop_wm_track.configs.baseline_wm_alltask.filter_data',
                dst_size=dst_size,
                min_num_frames=num_frames,
                min_area=dst_size[0] * dst_size[1],
                min_size=4,
            ),
            transform=dict(
                type='WMTransforms',
                dst_size=dst_size,
                num_frames=total_frames,
                sub_frames=num_frames,
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=1,
                        factor=4,
                    ),
                ),
                is_train=True,
                fps=fps,
                max_stride=4,
                num_views=3
            ),
            sampler=dict(
                type='BucketSampler',
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained=model_config['wan2.2-5b-diffusers'],
        flow_shift=5.0,
        expand_timesteps=True,
        view_dir=project_dir,
        sub_frames=num_frames,
        rollout=rollout,
    ),
    optimizers=dict(
        type='CAME8Bit',
        lr=2 ** (-14.5),
        weight_decay=1e-2,
    ),
    # optimizers=dict(
    #     type='AdamW',
    #     lr=2e-5,
    #     weight_decay=1e-2,
    # ), # 21509MiB
    schedulers=dict(
        type='ConstantScheduler',
    ),
    train=dict(
        resume=True,
        max_epochs=10000,
        gradient_accumulation_steps=4,
        # mixed_precision='fp8',  # fp16, bf16
        # fp8_ignore_modules=['t_embedder.1.linear', 'adaLN_modulation'],
        mixed_precision='bf16',  # fp16, bf16
        checkpoint_interval=10,
        checkpoint_total_limit=-1,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=False,
        activation_class_names=["WanTransformerBlock"],
    ),
    test=dict(),
)


def filter_data(
    all_data_list,
    dst_size=(1280, 704),
    min_num_frames=121,
    multiple=16,
    min_area=-1,
    min_size=1,
):
    from giga_datasets import image_utils

    video_info_dict = dict()
    for n, data_list in enumerate(all_data_list):
        for m, data_dict in enumerate(data_list):
            video_info = dst_size
            if video_info not in video_info_dict:
                video_info_dict[video_info] = []
            video_info_dict[video_info].append((n, m))

    new_all_data_list = [[] for _ in range(len(all_data_list))]
    bucket_index = 0
    for video_info, data_indexes in video_info_dict.items():
        if len(data_indexes) >= min_size:
            for n, m in data_indexes:
                data_dict = all_data_list[n][m]
                data_dict['bucket_index'] = bucket_index
                data_dict['video_info'] = video_info
                # data_dict['prompt'] = data_dict['prompt_sharegpt4video']
                new_all_data_list[n].append(data_dict)
            bucket_index += 1
    return new_all_data_list
