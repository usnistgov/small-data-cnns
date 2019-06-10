
import os
import random
import shutil
import numpy as np
import skimage.io

imageDir = '/home/mmajursk/small-data-cnns/data/HES/images'
maskDir = '/home/mmajursk/small-data-cnns/data/HES/masks'
fns = os.listdir(imageDir)
keys_flat = [fn for fn in fns if fn.endswith(".tif")]

validated_keys = list()
for fn in keys_flat:
    M = skimage.io.imread(os.path.join(maskDir, fn))
    nnz = np.count_nonzero(M)
    if nnz > int(0.1 * 256*256):
        validated_keys.append(fn)


random.shuffle(validated_keys)
validated_keys = validated_keys[0:1000]

test_keys = validated_keys[0:500]
train_keys = validated_keys[500:]

src_fldr = '/home/mmajursk/small-data-cnns/data/HES/images'
tgt_fldr = '/home/mmajursk/small-data-cnns/data/HES/train/images'
for fn in train_keys:
    shutil.copy(os.path.join(src_fldr, fn), os.path.join(tgt_fldr, fn))

tgt_fldr = '/home/mmajursk/small-data-cnns/data/HES/test/images'
for fn in test_keys:
    shutil.copy(os.path.join(src_fldr, fn), os.path.join(tgt_fldr, fn))


src_fldr = '/home/mmajursk/small-data-cnns/data/HES/masks'
tgt_fldr = '/home/mmajursk/small-data-cnns/data/HES/train/masks'
for fn in train_keys:
    shutil.copy(os.path.join(src_fldr, fn), os.path.join(tgt_fldr, fn))

tgt_fldr = '/home/mmajursk/small-data-cnns/data/HES/test/masks'
for fn in test_keys:
    shutil.copy(os.path.join(src_fldr, fn), os.path.join(tgt_fldr, fn))