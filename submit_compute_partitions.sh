#!/bin/bash

root=/nrs/saalfeld/maisl/flylight_benchmark/ffn/data/completely
in_dir=$root/labels
out_dir=$root/partitions
in_key=volumes/gt_instances_rm_5
out_key=volumes/af

log_dir=$root/log
samples=($(ls $in_dir ))

mkdir $log_dir
mkdir $out_dir

for sample in "${samples[@]}";
do
    echo $sample
    sample_name="$(basename $sample)"
    sample_name="${sample_name%.*}"
    echo $sample_name
    log_file=$log_dir/${sample_name}_log_partitions.out
    echo $log_file
    bsub -n 5 -W 4:00 -o $log_file python compute_partitions.py --input_volume $in_dir/$sample:$in_key --output_volume $out_dir/$sample_name.hdf:$out_key --thresholds 0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --lom_radius 24,24,24 --min_size 500
done


