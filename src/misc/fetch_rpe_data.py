# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import skimage.io

tile_size = 256


def process_tiling(img):
    # get the height of the image
    height = img.shape[0]
    width = img.shape[1]
    delta = int(tile_size)

    img_list = []

    for x_st in range(0, (width - tile_size), delta):
        for y_st in range(0, (height - tile_size), delta):
            x_end = x_st + tile_size
            y_end = y_st + tile_size
            if x_st < 0 or y_st < 0:
                continue
            if x_end > width or y_end > height:
                continue

            # crop out the tile
            pixels = img[y_st:y_end, x_st:x_end]
            img_list.append(pixels)
    return img_list


src_fp = '/mnt/Proj-002-RPE-Simon/disseminatedData/'
root_fldr = 'Live RPE - Narrowband'
channel = 'Blue 488'
type = 'Absorption Images'

tgt_fp = '/home/mmajursk/small-data-cnns/data/RPE_raw/'


for i in range(1,8):
    rootDir = os.path.join(src_fp, root_fldr, 'LORD-{}'.format(i))
    fldrs = [fn for fn in os.listdir(rootDir) if fn.startswith('Date ')]

    for j in range(len(fldrs)):
        fldr = fldrs[j]

        img_fldr = os.path.join(rootDir, fldr, channel, type)
        imgs = [fn for fn in os.listdir(img_fldr) if fn.endswith('.tif')]
        imgs = [fn for fn in imgs if 'BLANKWELL' not in  fn]
        for k in range(len(imgs)):
            img_fn = imgs[k]

            I = skimage.io.imread(os.path.join(img_fldr, img_fn), as_grey=True)
            img_tiles = process_tiling(I)

            for l in range(len(img_tiles)):
                tile = img_tiles[l]
                ofn = '{}_{}_{}_{}_id{}.tif'.format(root_fldr, 'LORD-{}'.format(i), fldr, img_fn.replace('.tif',''), str(l).zfill(8))
                skimage.io.imsave(os.path.join(tgt_fp, ofn), tile)
