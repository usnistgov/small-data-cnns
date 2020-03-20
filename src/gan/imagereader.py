# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import multiprocessing
from multiprocessing import Process
import queue
import random
import traceback
import os
import numpy as np
import augment
import skimage.io
import skimage.transform


def zscore_normalize(image_data):
    image_data = image_data.astype(np.float32)

    std = np.std(image_data)
    mv = np.mean(image_data)
    if std <= 1.0:
        # normalize (but dont divide by zero)
        image_data = (image_data - mv)
    else:
        # z-score normalize
        image_data = (image_data - mv) / std

    return image_data


def imread(fp):
    return skimage.io.imread(fp, as_grey=True)


class ImageReader:

    def __init__(self, imageDir, maskDir, batch_size, use_augmentation=True, shuffle=True, num_workers=1, sample_to_N_images=None):
        self.use_augmentation = use_augmentation
        self.queue_starvation = False

        self.shuffle = shuffle
        self.imageDir = imageDir
        self.maskDir = maskDir

        # list the *.tif files in the dataDir
        fns = os.listdir(self.imageDir)
        self.keys_flat = [fn for fn in fns if fn.endswith(".tif")]
        if sample_to_N_images is not None:
            random.shuffle(self.keys_flat)
            self.keys_flat = self.keys_flat[0:sample_to_N_images]

        # assumption that image names match between the two directories
        # fns = os.listdir(self.maskDir)
        # self.mask_filenames = [fn for fn in fns if fn.endswith(".tif")]

        # determine the image size
        I = imread(os.path.join(self.imageDir, self.keys_flat[0]))
        self.imageHeight = I.shape[0]
        self.imageWidth = I.shape[1]

        random.seed()

        print('Dataset has {} examples'.format(len(self.keys_flat)))

        self.batchsize = batch_size
        self.nb_workers = num_workers
        # setup queue
        self.terminateQ = multiprocessing.Queue(maxsize=self.nb_workers)  # limit output queue size
        self.maxOutQSize = 50
        self.outQ = multiprocessing.Queue(maxsize=self.maxOutQSize)  # limit output queue size
        self.idQ = multiprocessing.Queue(maxsize=self.nb_workers)

        self.workers = None
        self.done = False

    def get_epoch_size(self):
        # tie epoch size to the number of images
        return int(len(self.keys_flat) / self.batchsize)

    def get_fn_list(self):
        return self.keys_flat

    def startup(self):
        self.workers = None
        self.done = False

        [self.idQ.put(i) for i in range(self.nb_workers)]
        # launch workers
        self.workers = [Process(target=self.__image_loader) for i in range(self.nb_workers)]

        # start workers
        for w in self.workers:
            w.start()

    def shutdown(self):
        print('Reader shutdown called')
        # tell workers to shutdown
        for w in self.workers:
            self.terminateQ.put(None)
        print('  terminate queue populated')

        # empty the output queue (to allow blocking workers to terminate
        nb_none_received = 0
        # empty output queue
        while nb_none_received < len(self.workers):
            try:
                while True:
                    val = self.outQ.get_nowait()
                    if val is None:
                        print('  worker {} termination confirmation recieved'.format(nb_none_received))
                        nb_none_received += 1
            except queue.Empty:
                pass  # do nothing

        print('  All workers confirmed termination')
        print('  Joining all worker threads')
        # wait for the workers to terminate
        for w in self.workers:
            w.join()
        print(' Shutdown complete')

    @staticmethod
    def __format_image(image_data):
        # # reshape into tensor (NCHW)
        # image_data = image_data.reshape((-1, 1, image_data.shape[0], image_data.shape[1]))
        # reshape into tensor (NHWC)
        image_data = image_data.reshape((-1, image_data.shape[0], image_data.shape[1], 1))
        return image_data

    @staticmethod
    def __format_label(label_data):
        # # reshape into tensor (NCHW)
        # image_data = label_data.reshape((-1, label_data.shape[0], label_data.shape[1]))
        # reshape into tensor (NHWC)
        image_data = label_data.reshape((-1, label_data.shape[0], label_data.shape[1]))
        return image_data

    def __get_next_key(self):
        if self.shuffle:
            # select a key at random from the list (does not account for class imbalance)
            fn = self.keys_flat[random.randint(0, len(self.keys_flat) - 1)]
        else:  # no shuffle
            # without shuffle you cannot balance classes
            fn = self.keys_flat[self.key_idx]
            self.key_idx += self.nb_workers
            self.key_idx = self.key_idx % len(self.keys_flat)

        return fn

    def __image_loader(self):
        termimation_flag = False  # flag to control the worker shutdown
        self.key_idx = self.idQ.get()  # setup non-shuffle index to stride across flat keys properly
        try:
            # while the worker has not been told to terminate, loop infinitely
            while not termimation_flag:

                # poll termination queue for shutdown command
                try:
                    if self.terminateQ.get_nowait() is None:
                        termimation_flag = True
                        break
                except queue.Empty:
                    pass  # do nothing

                # allocate tensors for the current batch
                pixel_tensor = list()
                label_tensor = list()

                # build a batch selecting the labels using round robin through the shuffled order
                for i in range(self.batchsize):
                    fn = self.__get_next_key()

                    # load the image from disk
                    I = imread(os.path.join(self.imageDir, fn))

                    # load the mask from disk
                    M = imread(os.path.join(self.maskDir, fn))

                    if self.use_augmentation:
                        # setup the image data augmentation parameters
                        rotation_flag = False
                        reflection_flag = True
                        jitter_augmentation_severity = 0.1  # x% of a FOV
                        noise_augmentation_severity = 0.02  # vary noise by x% of the dynamic range present in the image
                        scale_augmentation_severity = 0.1 # vary size by x%
                        blur_max_sigma = 2.5 # pixels
                        # intensity_augmentation_severity = 0.05

                        I = I.astype(np.float32)
                        M = M.astype(np.int32)

                        # perform image data augmentation
                        I, M = augment.augment_image(I, M=M,
                            rotation_flag=rotation_flag,
                            reflection_flag=reflection_flag,
                              jitter_augmentation_severity=jitter_augmentation_severity,
                              noise_augmentation_severity=noise_augmentation_severity,
                              scale_augmentation_severity=scale_augmentation_severity,
                              blur_augmentation_max_sigma=blur_max_sigma)


                    # format the image into a tensor
                    I = self.__format_image(I)
                    I = zscore_normalize(I)
                    M = self.__format_label(M)

                    # append the image and label to the batch being built
                    pixel_tensor.append(I)
                    label_tensor.append(M)

                # convert the list of images into a numpy array tensor ready for Caffe2
                pixel_tensor = np.concatenate(pixel_tensor, axis=0).astype(np.float32)
                # pixel_tensor = self.__zscore_normalize(pixel_tensor)

                label_tensor = np.concatenate(label_tensor, axis=0).astype(np.int32)

                # add the batch in the output queue
                # this put block until there is space in the output queue (size 50)
                self.outQ.put((pixel_tensor, label_tensor))

        except Exception as e:
            print('***************** Reader Error *****************')
            print(e)
            traceback.print_exc()
            print('***************** Reader Error *****************')
        finally:
            # when the worker terminates add a none to the output so the parent gets a shutdown confirmation from each worker
            self.outQ.put(None)

    def get_batch(self):
        # get a ready to train batch from the output queue and pass to to the caller
        if self.outQ.qsize() < int(0.2*self.maxOutQSize):
            if not self.queue_starvation:
                print('Input Queue Starvation !!!!')
            self.queue_starvation = True
        if self.queue_starvation and self.outQ.qsize() > int(0.8*self.maxOutQSize):
            print('Input Queue Starvation Over')
            self.queue_starvation = False
        return self.outQ.get()

    def generator(self):
        while True:
            batch = self.get_batch()
            if batch is None:
                return
            yield batch

    def get_queue_size(self):
        return self.outQ.qsize()

