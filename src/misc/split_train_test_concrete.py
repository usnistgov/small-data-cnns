
import os
import random
import shutil

imageDir = '/home/mmajursk/small-data-cnns/data/Concrete_orig/images'
fns = os.listdir(imageDir)
keys_flat = [fn for fn in fns if fn.endswith(".tif")]
random.shuffle(keys_flat)

train_keys = keys_flat[0:500]
test_keys = keys_flat[500:]


src_fldr = '/home/mmajursk/small-data-cnns/data/Concrete_orig/images'
tgt_fldr = '/home/mmajursk/small-data-cnns/data/Concrete/train/images'
for fn in train_keys:
    shutil.copy(os.path.join(src_fldr, fn), os.path.join(tgt_fldr, fn))

tgt_fldr = '/home/mmajursk/small-data-cnns/data/Concrete/test/images'
for fn in test_keys:
    shutil.copy(os.path.join(src_fldr, fn), os.path.join(tgt_fldr, fn))


src_fldr = '/home/mmajursk/small-data-cnns/data/Concrete_orig/masks'
tgt_fldr = '/home/mmajursk/small-data-cnns/data/Concrete/train/masks'
for fn in train_keys:
    shutil.copy(os.path.join(src_fldr, fn), os.path.join(tgt_fldr, fn))

tgt_fldr = '/home/mmajursk/small-data-cnns/data/Concrete/test/masks'
for fn in test_keys:
    shutil.copy(os.path.join(src_fldr, fn), os.path.join(tgt_fldr, fn))