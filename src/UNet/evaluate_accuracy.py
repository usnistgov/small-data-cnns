# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import matplotlib.pyplot as plt
import os
import numpy as np
import skimage.io
import sklearn.metrics
import skimage.morphology
import skimage.measure
from multiprocessing import Pool
from functools import partial



GPU_ID = 0
NUM_CLASSES = 2

model_fldr = '/scratch/small-data-cnns/models/rpe/'


image_folder = '/scratch/small-data-cnns/source_data/RPE/test/masks/'
img_fns = [fn for fn in os.listdir(image_folder) if fn.endswith('.tif')]
output_folder = '/scratch/small-data-cnns/models/rpe/'

types = {'noaug', 'aug', 'gan', 'gan-aug', 'tl', 'tl-aug'}
annotation_counts = [50, 100, 200, 300, 400, 500]
reps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def worker(i_fn, model_fn):
    r_m = skimage.io.imread(os.path.join(image_folder, i_fn), as_grey=True)
    r_m = (r_m > 0).astype(np.uint8)
    c_m = skimage.io.imread(os.path.join(model_fldr, model_fn, 'inference', i_fn), as_grey=True)

    f1 = sklearn.metrics.f1_score(r_m.ravel(), c_m.ravel())

    r_rm = (skimage.morphology.skeletonize(r_m) == 0).astype(np.uint8)
    r_cm = (skimage.morphology.skeletonize(c_m) == 0).astype(np.uint8)

    r_rm = skimage.measure.label(r_rm, connectivity=1).astype(np.int32)
    r_cm = skimage.measure.label(r_cm, connectivity=1).astype(np.int32)

    ari = sklearn.metrics.adjusted_rand_score(r_rm.ravel(), r_cm.ravel())

    return (f1, ari)


work_pool = Pool(40)


with open(os.path.join(output_folder, 'results_per_img.csv'), 'w') as csvfile_per_img:
    csvfile_per_img.write('Type, Annotation Count, Mean Boundary Dice, Std Boundary Dice, Mean Region ARI, Std Region ARI\n')

    with open(os.path.join(output_folder, 'results_per_rep.csv'), 'w') as csvfile_per_rep:
        csvfile_per_rep.write('Type, Annotation Count, Mean Boundary Dice, Std Boundary Dice, Mean Region ARI, Std Region ARI\n')

        for t in types:
            for a in annotation_counts:
                boundary_f1_vals = []
                region_ari_vals = []

                mean_boundary_f1_vals = []
                mean_region_ari_vals = []

                for r in reps:
                    model_fn = 'unet-rpe-{}-{}-{}'.format(a, r, t)
                    if not os.path.exists(os.path.join(model_fldr, model_fn)):
                        continue

                    print('**************************')
                    print('Model Folder: {}'.format(model_fn))

                    vals = work_pool.map(partial(worker, model_fn=model_fn), img_fns)

                    per_rep_mean_f1_vals = []
                    per_rep_mean_ari_vals = []
                    for v in vals:
                        f1, ari = v
                        boundary_f1_vals.append(f1)
                        per_rep_mean_f1_vals.append(f1)

                        region_ari_vals.append(ari)
                        per_rep_mean_ari_vals.append(ari)

                    mean_boundary_f1_vals.append(np.mean(np.asarray(per_rep_mean_f1_vals)))
                    mean_region_ari_vals.append(np.mean(np.asarray(per_rep_mean_ari_vals)))

                print('mean Rep Boundary Dice: {:6f} +- {:6f}'.format(np.mean(mean_boundary_f1_vals), np.std(mean_boundary_f1_vals)))
                print('mean Rep Region ARI: {:6f} +- {:6f}'.format(np.mean(mean_region_ari_vals), np.std(mean_region_ari_vals)))

                csvfile_per_rep.write('{}, {}, {}, {}, {}, {}\n'.format(t, a, np.mean(mean_boundary_f1_vals), np.std(mean_boundary_f1_vals), np.mean(mean_region_ari_vals), np.std(mean_region_ari_vals)))

                boundary_f1_vals = np.asarray(boundary_f1_vals)
                region_ari_vals = np.asarray(region_ari_vals)

                print('mean Img Boundary Dice: {:6f} +- {:6f}'.format(np.mean(boundary_f1_vals), np.std(boundary_f1_vals)))
                print('mean Img Region ARI: {:6f} +- {:6f}'.format(np.mean(region_ari_vals), np.std(region_ari_vals)))

                csvfile_per_img.write('{}, {}, {}, {}, {}, {}\n'.format(t, a, np.mean(boundary_f1_vals), np.std(boundary_f1_vals), np.mean(region_ari_vals), np.std(region_ari_vals)))




