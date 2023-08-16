import numpy as np
import os
from ffn.inference import storage
from skimage import io

in_folder = "/nrs/saalfeld/maisl/flylight_benchmark/ffn/experiments/230808_01_f33_d12_init/validate"

seg, _ = storage.load_segmentation(in_folder, (0, 0, 0))

print(seg.shape, seg.dtype, seg.min(), seg.max())

print(np.unique(seg))

mip = np.max(seg, axis=0)
print(mip.shape, mip.min(), mip.max())
io.imsave(os.path.join(in_folder, "mip_ffn_test.tif"), mip.astype(np.int32))

