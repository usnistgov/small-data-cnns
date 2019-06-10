
import os
import random
import shutil
import numpy as np
import skimage.io

imageDir = '/home/mmajursk/small-data-cnns/data/hes_orig/validate/phase/'
maskDir = '/home/mmajursk/small-data-cnns/data/hes_orig/validate/mask/'
fns = os.listdir(imageDir)
keys_flat = [fn for fn in fns if fn.endswith(".tif")]

validated_keys = list()
for fn in keys_flat:
    M = skimage.io.imread(os.path.join(maskDir, fn))
    nnz = np.count_nonzero(M)
    if nnz > 0:
        validated_keys.append(fn)

ofp = '/home/mmajursk/small-data-cnns/data/HES_raw/'
for fn in validated_keys:
    shutil.copy(os.path.join(imageDir, fn), os.path.join(ofp, fn))

