
from pycocotools.coco import COCO
import skimage.color
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)


dataDir='/home/mmajursk/small-data-cnns/data/COCO/'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
print('Found {} Categories'.format(len(cats)))

#
# nms=[cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))
#
# # get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
# imgIds = coco.getImgIds(catIds=catIds )
# imgIds = coco.getImgIds(imgIds = [324158])
#
#
# # list all images in the dataset
# imgIds = coco.getImgIds()
#
# img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
#
# catIds = coco.getCatIds()
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
#
# imgDir = annFile='{}/{}/'.format(dataDir,dataType)
# fn = img['file_name']
# import os
# I = io.imread(os.path.join(imgDir, fn))
#
# I = io.imread(img['coco_url'])
# I = skimage.color.rgb2grey(I)
# M = np.zeros(I.shape, dtype=np.int32)
# for ann in anns:
#     ann_M = coco.annToMask(ann)
#     M[ann_M > 0] = ann['category_id']
#
# plt.imshow(I); plt.axis('off')
# plt.imshow(M); plt.axis('off')
#
#
#
# # load and display image
# # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# # use url to load image
# I = io.imread(img['coco_url'])
# plt.axis('off')
#
# # load and display instance annotations
# plt.imshow(I); plt.axis('off')
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)



# coco.showAnns(anns)
