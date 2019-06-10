import multiprocessing
from multiprocessing import Process
import queue
import random
import traceback
import lmdb
import numpy as np
import augment
import os
import skimage.io
import skimage.transform
from isg_ai_pb2 import Image
import unet_gan_model


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


def min_max_normalize(image_data):
    image_data = image_data.astype(np.float32)

    mv = np.mean(image_data)
    image_data = image_data - mv
    image_data = image_data / np.max(np.fabs(image_data))

    return image_data


def imread(fp):
    return skimage.io.imread(fp, as_grey=True)


def imwrite(img, fp):
    skimage.io.imsave(fp, img)


class ImageReader:
    # setup the image data augmentation parameters
    _reflection_flag = True
    _rotation_flag = True
    _jitter_augmentation_severity = 0.1  # x% of a FOV
    _noise_augmentation_severity = 0.02  # vary noise by x% of the dynamic range present in the image
    _scale_augmentation_severity = 0.1  # vary size by x%
    _blur_max_sigma = 2  # pixels
    _intensity_augmentation_severity = None # vary intensity by x% of the dynamic range present in the image

    def __init__(self, img_db, batch_size, use_augmentation=True, shuffle=True, num_workers=1):
        random.seed()

        # copy inputs to class variables
        self.image_db = img_db
        self.use_augmentation = use_augmentation
        self.shuffle = shuffle
        self.batchsize = batch_size
        self.nb_workers = num_workers

        # init class state
        self.queue_starvation = False
        self.maxOutQSize = num_workers * 10
        self.workers = None
        self.done = False

        # setup queue mechanism
        self.terminateQ = multiprocessing.Queue(maxsize=self.nb_workers)  # limit output queue size
        self.outQ = multiprocessing.Queue(maxsize=self.maxOutQSize)  # limit output queue size
        self.idQ = multiprocessing.Queue(maxsize=self.nb_workers)

        # confirm that the input database exists
        if not os.path.exists(self.image_db):
            print('Could not load database file: ')
            print(self.image_db)
            raise IOError("Missing Database")

        # get a list of keys from the lmdb
        self.keys_flat = list()

        self.lmdb_env = lmdb.open(self.image_db, map_size=int(2e10), readonly=True) # 20 GB
        self.lmdb_txns = list()

        self.number_classes = 1

        datum = Image()  # create a datum for decoding serialized protobuf objects
        print('Initializing image database')

        with self.lmdb_env.begin(write=False) as lmdb_txn:
            cursor = lmdb_txn.cursor()

            # move cursor to the first element
            cursor.first()
            # get the first serialized value from the database and convert from serialized representation
            datum.ParseFromString(cursor.value())
            # record the image size
            self.image_size = [datum.img_height, datum.img_width]

            # iterate over the database getting the keys
            for key, val in cursor:
                self.keys_flat.append(key)

        print('Dataset has {} examples'.format(len(self.keys_flat)))

    def get_epoch_size(self):
        # tie epoch size to the number of images
        return int(len(self.keys_flat) / self.batchsize)

    def get_image_size(self):
        return self.image_size

    def get_batch_size(self):
        return self.batchsize

    def startup(self):
        self.workers = None
        self.done = False

        [self.idQ.put(i) for i in range(self.nb_workers)]
        [self.lmdb_txns.append(self.lmdb_env.begin(write=False)) for i in range(self.nb_workers)]
        # launch workers
        self.workers = [Process(target=self.__image_loader) for i in range(self.nb_workers)]

        # start workers
        for w in self.workers:
            w.start()

    def shutdown(self):
        # tell workers to shutdown
        for w in self.workers:
            self.terminateQ.put(None)

        # empty the output queue (to allow blocking workers to terminate
        nb_none_received = 0
        # empty output queue
        while nb_none_received < len(self.workers):
            try:
                while True:
                    val = self.outQ.get_nowait()
                    if val is None:
                        nb_none_received += 1
            except queue.Empty:
                pass  # do nothing

        # wait for the workers to terminate
        for w in self.workers:
            w.join()


    @staticmethod
    def __format_image(image_data):
        # reshape into tensor (NCHW)
        image_data = image_data.reshape((-1, 1, image_data.shape[0], image_data.shape[1]))
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
            datum = Image()  # create a datum for decoding serialized caffe_pb2 objects

            local_lmdb_txn = self.lmdb_txns[self.key_idx]

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
                noise_tensor = list()

                # build a batch selecting the labels using round robin through the shuffled order
                for i in range(self.batchsize):
                    fn = self.__get_next_key()

                    # extract the serialized image from the database
                    value = local_lmdb_txn.get(fn)
                    # convert from serialized representation
                    datum.ParseFromString(value)

                    # convert from string to numpy array
                    I = np.fromstring(datum.image, dtype=datum.img_type)
                    # reshape the numpy array using the dimensions recorded in the datum
                    I = I.reshape(datum.img_height, datum.img_width)

                    if self.use_augmentation:
                        I = I.astype(np.float32)

                        # perform image data augmentation
                        I = augment.augment_image(I,
                                                     reflection_flag=self._reflection_flag,
                                                     rotation_flag=self._rotation_flag,
                                                     jitter_augmentation_severity=self._jitter_augmentation_severity,
                                                     noise_augmentation_severity=self._noise_augmentation_severity,
                                                     scale_augmentation_severity=self._scale_augmentation_severity,
                                                     blur_augmentation_max_sigma=self._blur_max_sigma,
                                                     intensity_augmentation_severity=self._intensity_augmentation_severity)

                    # format the image into a tensor
                    I = self.__format_image(I)
                    # I = zscore_normalize(I)
                    I = min_max_normalize(I)

                    # append the image and label to the batch being built
                    pixel_tensor.append(I)

                    # create noise vector
                    # M = np.random.uniform(-1, 1, size=(1,unet_gan_model.Z_DIM)).astype(np.float32)
                    M = np.random.randn(1, unet_gan_model.Z_DIM).astype(np.float32)
                    noise_tensor.append(M)

                # convert the list of images into a numpy array tensor ready for tensorflow
                pixel_tensor = np.concatenate(pixel_tensor, axis=0).astype(np.float32)
                noise_tensor = np.concatenate(noise_tensor, axis=0).astype(np.float32)

                # add the batch in the output queue
                # this put block until there is space in the output queue (size 50)
                self.outQ.put((pixel_tensor, noise_tensor))

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
        if self.outQ.qsize() < int(0.1*self.maxOutQSize):
            if not self.queue_starvation:
                print('Input Queue Starvation !!!!')
            self.queue_starvation = True
        if self.queue_starvation and self.outQ.qsize() > int(0.5*self.maxOutQSize):
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

