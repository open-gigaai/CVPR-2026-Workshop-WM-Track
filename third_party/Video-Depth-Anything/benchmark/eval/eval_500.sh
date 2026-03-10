#!/bin/sh
set -x
set -e

pred_disp_root=$1 # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] prediction
benchmark_root=$2 # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] ground truth

#eval scannet
python3 benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets scannet_500

#eval kitti
python3 benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets kitti_500

#eval bonn
python3 benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets bonn_500

#eval nyu
python3 benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets nyuv2_500
