import shutil
import os

ifp = "/home/mmajursk/Downloads/cvpr-2019/HES_raw/images/"
ofp = "/home/mmajursk/Downloads/cvpr-2019/HES_raw/"

image_files = os.listdir(ifp)
image_files = [fn for fn in image_files if fn.endswith(".tif")]

[shutil.move(os.path.join(ifp, fn), os.path.join(ofp, fn)) for fn in image_files]

shutil.rmtree(ifp)