#!/bin/bash

export PATH=/usr/local/cuda/bin:$PATH

exp_id=230808_01_f33_d12_init
exp_dir=/nrs/saalfeld/maisl/flylight_benchmark/ffn/experiments/$exp_id

root=/nrs/saalfeld/maisl/flylight_benchmark/ffn/data/completely
coord_file=$root/coordinates/coords

raw_key=volumes/raw_normalized
label_key=volumes/gt_instances_rm_5

batch_size=4
max_steps=100000

log_dir=$root/log
samples=($(cat $root/train_samples.txt ))

for sample in "${samples[@]}";
do
    IFS=','
    read -a sample_array <<< "$sample"

    vol_name=${sample_array[0]}
    label_file=${sample_array[1]}
    raw_file=${sample_array[2]}

    sample_name="$(basename $partition_file)"
    sample_name="${sample_name%.*}".hdf
    partition_file="$(dirname $partition_file)"
    partition_file="${partition_file%/*}"/partitions/$sample_name

    #IFS=" "
    label_files+=$vol_name:$label_file:$label_key,
    raw_files+=$vol_name:$raw_file:$raw_key,
done

echo $label_files
echo $raw_files

log_file=$log_dir/log_train_${exp_id}.out
echo $log_file

bsub -n 5 -gpu "num=1" -q gpu_rtx -W 144:00 -o $log_file python train.py --train_coords $coord_file --data_volumes $raw_files --label_volumes $label_files --model_name convstack_3d.ConvStack3DFFNModel --model_args "{\"depth\": 12, \"fov_size\": [33, 33, 33], \"deltas\": [8, 8, 8]}" --train_dir $exp_dir --max_steps $max_steps
