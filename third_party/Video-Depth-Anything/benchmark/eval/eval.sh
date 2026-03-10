#!/bin/sh
set -x
set -e

pred_disp_root=$1 # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] prediction
benchmark_root=$2 # The parent directory that contaning [sintel, scannet, KITTI, bonn, NYUv2] ground truth

#eval sintel
python3 benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets sintel

#eval scannet
python3 benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets scannet

#eval kitti
python3 benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets kitti

#eval bonn
python3 benchmark/eval/eval.py \
    --infer_path $pred_disp_root \
    --benchmark_path $benchmark_root \
    --datasets bonn
