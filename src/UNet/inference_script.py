import os
import inference

GPU_ID = 0
NUM_CLASSES = 2


fp = '/scratch/small-data-cnns/models/rpe/'

fns = [fn for fn in os.listdir(fp) if fn.startswith('unet-rpe')]

image_folder = '/scratch/small-data-cnns/source_data/RPE/test/images/'


for fn in fns:
    print('Model: {}'.format(fn))
    checkpoint_filepath = os.path.join(fp, fn, 'checkpoint')
    if os.path.exists(checkpoint_filepath):
        if not os.path.exists(os.path.join(fp, fn, 'inference')):
            checkpoint_filepath = os.path.join(checkpoint_filepath, 'model.ckpt')
            output_folder = os.path.join(fp, fn, 'inference')
            inference.main(GPU_ID, checkpoint_filepath, image_folder, output_folder, NUM_CLASSES)