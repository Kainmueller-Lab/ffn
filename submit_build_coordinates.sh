#!/bin/bash

root=/nrs/saalfeld/maisl/flylight_benchmark/ffn/data/completely
in_dir=$root/partitions_lom_32
out_dir=$root/coordinates_lom_32
in_key=volumes/af
lom=32

log_dir=$root/log_lom_32
samples=($(cat $root/train_samples.txt ))

mkdir $out_dir

for sample in "${samples[@]}";
do
    IFS=','
    read -a sample_array <<< "$sample"

    vol_name=${sample_array[0]}
    partition_file=${sample_array[1]}
    sample_name="$(basename $partition_file)"
    sample_name="${sample_name%.*}".hdf
    partition_file="$(dirname $partition_file)"
    partition_file="${partition_file%/*}"/partitions/$sample_name

    #IFS=" "
    partition_files+=$vol_name:$partition_file:$in_key,
done

echo $partition_files
log_file=$log_dir/log_coordinates.out
echo $log_file
bsub -n 10 -W 12:00 -o $log_file python build_coordinates.py --partition_volumes $partition_files --coordinate_output $out_dir/coords --margin $lom,$lom,$lom

