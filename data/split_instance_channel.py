import os
from glob import glob
import numpy as np
import zarr
from numcodecs import Blosc

compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)

in_folder = "/nrs/saalfeld/maisl/data/flylight/flylight_complete/train"
#in_files = ["/nrs/saalfeld/maisl/data/flylight/flylight_complete/train/R75E01-20181030_64_D1.zarr", ]
in_key = "volumes/gt_instances_rm_5"
out_folder_base = "/nrs/saalfeld/maisl/flylight_benchmark/ffn/data/completely/"
out_folder = os.path.join(out_folder_base, "labels")

in_files = glob(in_folder + "/*.zarr")
sample_cnt = 1
samples = []

for in_file in in_files:
    print(sample_cnt)
    labels = np.array(zarr.open(in_file, "r")[in_key])
    if np.max(np.sum(labels > 0, axis=0)) > 1:
        print("overlapping sample %s" % os.path.basename(in_file))
        for i in range(labels.shape[0]):
            label = labels[i]
            out_file = os.path.join(out_folder, 
                    os.path.basename(in_file).split(".")[0] + "_%i.zarr" % i
                    )
            print(out_file)
            outf = zarr.open(out_file)
            print(label.dtype)
            outf.create_dataset(
                    in_key,
                    data=label,
                    compressor=compressor,
                    dtype=np.uint8,
                    overwrite=True
                    )
            samples.append([sample_cnt, out_file, in_file])
            sample_cnt +=1

    else:
        print("non-overlapping sample %s" % os.path.basename(in_file))
        labels = np.max(labels, axis=0)
        out_file = os.path.join(out_folder, os.path.basename(in_file))
        outf = zarr.open(out_file)
        print(labels.dtype)
        outf.create_dataset(
                in_key,
                data=labels,
                compressor=compressor,
                dtype=np.uint8,
                overwrite=True
                )
        samples.append([sample_cnt, out_file, in_file])
        sample_cnt +=1

    # save train sample ids and links to file
    with open(os.path.join(out_folder_base, "train_samples.txt"), 'w') as f:
        for sample in samples:
            f.write("train" + str(sample[0]) + "," + sample[1] + "," + sample[2] + "\n")

