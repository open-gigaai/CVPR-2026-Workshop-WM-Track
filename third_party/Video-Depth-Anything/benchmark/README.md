# Benchmark

## Prepare Datasets
Download datasets from the following links:
[Sintel](http://sintel.is.tue.mpg.de/), [KITTI](https://www.cvlibs.net/datasets/kitti/), [Bonn](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/index.html), [ScanNet](http://www.scan-net.org/), [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

```bash
pip3 install natsort
cd benchmark/dataset_extract
python3 dataset_extrtact${dataset}.py
```
This script will extract the dataset to the `benchmark/dataset_extract/dataset` folder. It will also generate the json file for the dataset.

## Run inference
```bash
python3 benchmark/infer/infer.py \
    --infer_path ${out_path} \
    --json_file ${json_path} \
    --datasets ${dataset}
```
Options:
- `--infer_path`: path to save the output results
- `--json_file`: path to the json file for the dataset, like `sintel_video.json`, `scannet_video_500.json`, `scannet_video_tae.json`
- `--datasets`: dataset name, choose from `sintel`, `kitti`, `bonn`, `scannet`, `nyuv2`

## Run evaluation
```bash
## tae
bash benchmark/eval/eval_tae.sh ${out_path} benchmark/dataset_extract/dataset
## ~110frame like DepthCrafter
bash benchmark/eval/eval.sh ${out_path} benchmark/dataset_extract/dataset
## ~500frame 
bash benchmark/eval/eval_500.sh ${out_path} benchmark/dataset_extract/dataset
```
