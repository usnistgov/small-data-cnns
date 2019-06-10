
import os
import skimage.io
import numpy as np


maskDir = '/home/mmajursk/small-data-cnns/data/HES/train/masks/'
fns = os.listdir(maskDir)
keys_flat = [fn for fn in fns if fn.endswith(".tif")]

for fn in keys_flat:
    M = skimage.io.imread(os.path.join(maskDir, fn), as_gray=True)
    M = np.asarray(M > 0, dtype=np.uint8)
    skimage.io.imsave(os.path.join(maskDir, fn), M)

maskDir = '/home/mmajursk/small-data-cnns/data/HES/test/masks/'
fns = os.listdir(maskDir)
keys_flat = [fn for fn in fns if fn.endswith(".tif")]

for fn in keys_flat:
    M = skimage.io.imread(os.path.join(maskDir, fn), as_gray=True)
    M = np.asarray(M > 0, dtype=np.uint8)
    skimage.io.imsave(os.path.join(maskDir, fn), M)