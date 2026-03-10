#!/bin/sh
set -x
set -e

pred_disp_root=$1 # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] prediction
benchmark_root=$2 # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] ground truth

#eval scannet
python3 benchmark/eval/eval_tae.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets scannet \
    --start_idx 10 \
    --end_idx 180 \
    --eval_scenes_num 20 \
    --hard_crop


