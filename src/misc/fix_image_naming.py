import os
import shutil


def rename_images(folder):
    image_files = os.listdir(folder)
    image_files.sort()

    image_files = [fn for fn in image_files if fn.endswith(".tif")]

    for i in range(len(image_files)):
        image = image_files[i]
        fn = str(i).zfill(8) + '.tif'
        shutil.move(os.path.join(folder, image), os.path.join(folder, fn))




fldr = '/home/mmajursk/small-data-cnns/data/RPE/test/absorbance/'
rename_images(fldr)

fldr = '/home/mmajursk/small-data-cnns/data/RPE/test/labeled-cell-masks/'
rename_images(fldr)

fldr = '/home/mmajursk/small-data-cnns/data/RPE/test/masks/'
rename_images(fldr)

fldr = '/home/mmajursk/small-data-cnns/data/RPE/train/absorbance/'
rename_images(fldr)

fldr = '/home/mmajursk/small-data-cnns/data/RPE/train/masks/'
rename_images(fldr)