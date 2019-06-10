
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
