import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os


# Function to query tensorflow to obtain the number of GPUs the system has
def get_available_gpu_count():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpus_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpus_names)

# Get the number of system gpus, so we know later how many towers we need to build, 1 per gpu
NUM_GPUS = get_available_gpu_count()
print('Found {} GPUS'.format(NUM_GPUS))
# build a list of GPU_IDs which are numbered 0 through NUM_GPUS-1
GPU_IDS = list(range(NUM_GPUS))

# set the system environment so that the PCIe GPU ids match the Nvidia ids in nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
# define the number of disk readers (which are each single threaded) to match the number of GPUs, so we have one single threaded reader per gpu
READER_COUNT = NUM_GPUS
if len(GPU_IDS) == 0:
    GPU_IDS = -1
    READER_COUNT = 1
print('Reader Count: {}'.format(READER_COUNT))


LEARNING_RATE = 1e-4
TILE_SIZE = 512 # fixed based on the lmdb
terminate_after_num_epochs_without_test_loss_improvement = 10



parser = argparse.ArgumentParser(prog='train_unet', description='Script which trains a unet model')

parser.add_argument('--batch_size', dest='batch_size', type=int,
                    help='training batch size', default=8)
parser.add_argument('--epoch_count', dest='epoch_count', type=int,
                    help='number of epochs to train the model for', default=100)
parser.add_argument('--data_dir', dest='data_dir', type=str,
                    required=True)
parser.add_argument('--output_dir', dest='output_folder', type=str,
                    help='Folder where outputs will be saved (Required)', required=True)
parser.add_argument('--gradient_update_location', dest='gradient_update_location', type=str,
                    help="Where to perform gradient averaging and update. Options: ['cpu', 'gpu:#']. Use the GPU if you have a fully connected topology, cpu otherwise.", default='gpu:0')


args = parser.parse_args()

batch_size = args.batch_size
epoch_count = args.epoch_count
data_dir = args.data_dir
output_folder = args.output_folder
gradient_update_location = args.gradient_update_location



# verify gradient_update_location is valid
if GPU_IDS == -1:
    # no gpu found, so average on cpu
    gradient_update_location = 'cpu:0'
    print('Found no GPUs overwriting gradient averaging location request to use cpu')
else:
    valid_location = False
    if gradient_update_location == 'cpu':
        # if the GPUs do not have a fully connected topology (e.g. NVLink), its faster to perform gradient averaging on the CPU
        valid_location = True
        gradient_update_location = gradient_update_location + ':0' # append the useless id number
    for id in GPU_IDS:
        if gradient_update_location == 'gpu:{}'.format(id):
            valid_location = True
    if not valid_location:
        raise Exception("Invalid option for 'gradient_update_location': {}".format(gradient_update_location))


print('Arguments:')
print('batch_size = {}'.format(batch_size))
print('epoch_count = {}'.format(epoch_count))
print('data_dir = {}'.format(data_dir))
print('output folder = {}'.format(output_folder))
print('gradient_update_location = {}'.format(gradient_update_location))


import numpy as np
import tensorflow as tf
import unet_model
import shutil
import imagereader
import csv
import time


def save_csv_file(output_folder, data, filename):
    np.savetxt(os.path.join(output_folder, filename), np.asarray(data), fmt='%.6g', delimiter=",")


def save_text_csv_file(output_folder, data, filename):
    with open(os.path.join(output_folder, filename), 'w') as csvfile:
        for i in range(len(data)):
            csvfile.write(data[i])
            csvfile.write('\n')


def plot(output_folder, name, train_val, test_val, epoch_size, log_scale=True):
    mpl.rcParams['agg.path.chunksize'] = 10000  # fix for error in plotting large numbers of points

    train_val = np.asarray(train_val)
    test_val = np.asarray(test_val)
    epoch_size = np.array(epoch_size, dtype=np.float32)
    iterations = np.arange(0, len(train_val))
    test_iterations = np.arange(0, len(test_val)) * epoch_size

    dot_size = 4
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = plt.gca()
    ax.scatter(iterations, train_val, c='b', s=dot_size)
    ax.plot(test_iterations, test_val, 'r-', marker='o', markersize=12)
    plt.ylim((np.min(train_val), np.max(train_val)))
    if log_scale:
        ax.set_yscale('log')
    plt.ylabel('{}'.format(name))
    plt.xlabel('Iterations')
    fig.savefig(os.path.join(output_folder, '{}.png'.format(name)))

    plt.close(fig)


def train_model():
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)


    print('Setting up test image reader')
    test_reader = imagereader.ImageReader(data_dir, 'val2017', batch_size=batch_size,
                                          use_augmentation=False, shuffle=False, num_workers=READER_COUNT,
                                          target_size=TILE_SIZE)
    print('Test Reader has {} batches'.format(test_reader.get_epoch_size()))

    print('Setting up training image reader')
    train_reader = imagereader.ImageReader(data_dir, 'train2017', batch_size=batch_size,
                                           use_augmentation=False, shuffle=True, num_workers=READER_COUNT,
                                           target_size=TILE_SIZE)
    print('Train Reader has {} batches'.format(train_reader.get_epoch_size()))

    # start the input queues
    try:
        print('Starting Readers')
        train_reader.startup()
        test_reader.startup()

        print('Creating model')
        with tf.Graph().as_default(), tf.device('/' + gradient_update_location):
            train_init_op, test_init_op, train_op, loss_op, accuracy_op, is_training_placeholder = unet_model.build_towered_model(train_reader, test_reader, GPU_IDS, LEARNING_RATE, train_reader.get_number_of_classes(), TILE_SIZE)


            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            init = tf.global_variables_initializer()
            sess.run(init)

            # setup network accuracy tracking variables
            train_loss = list()
            train_accuracy = list()
            test_loss = list()
            test_accuracy = list()
            test_loss.append(np.inf)
            test_accuracy.append(0)

            train_epoch_size = train_reader.get_epoch_size()
            test_epoch_size = test_reader.get_epoch_size()

            epoch = 0
            print('Running Network')
            while True:  # loop until early stopping
                print('---- Epoch: {} ----'.format(epoch))

                print('initializing training data iterator')
                sess.run(train_init_op)
                print('   iterator init complete')

                adj_batch_count = train_epoch_size
                for step in range(adj_batch_count):
                    _, loss_val, accuracy_val = sess.run([train_op, loss_op, accuracy_op], feed_dict={is_training_placeholder: True})
                    train_loss.append(loss_val)
                    train_accuracy.append(accuracy_val)
                    print('Train Epoch: {} Batch {}/{}: loss = {}'.format(epoch, step, adj_batch_count, loss_val))
                print('Train Epoch: {} : Accuracy = {}'.format(epoch, np.mean(train_accuracy[-train_epoch_size:])))

                sess.run(test_init_op)
                adj_batch_count = int(np.ceil(test_epoch_size / NUM_GPUS))
                epoch_test_loss = list()
                epoch_test_accuracy = list()
                for step in range(adj_batch_count):
                    loss_val, accuracy_val = sess.run([loss_op, accuracy_op], feed_dict={is_training_placeholder: False})
                    epoch_test_loss.append(loss_val)
                    epoch_test_accuracy.append(accuracy_val)
                    print('Test Epoch: {} Batch {}/{}: loss = {}'.format(epoch, step, adj_batch_count, loss_val))
                test_loss.append(np.mean(epoch_test_loss))
                test_accuracy.append(np.mean(epoch_test_accuracy))
                print('Test Epoch: {} : Accuracy = {}'.format(epoch, test_accuracy[-1]))

                plot(output_folder, 'loss', train_loss, test_loss, train_epoch_size)
                plot(output_folder, 'accuracy', train_accuracy, test_accuracy, train_epoch_size, log_scale=False)

                save_csv_file(output_folder, train_accuracy, 'train_accuracy.csv')
                save_csv_file(output_folder, train_loss, 'train_loss.csv')
                save_csv_file(output_folder, test_loss, 'test_loss.csv')
                save_csv_file(output_folder, test_accuracy, 'test_accuracy.csv')

                # determine early stopping
                print('Best Current Epoch Selection:')
                print('Test Loss:')
                print(test_loss)
                best_epoch = np.argmin(test_loss)
                print('Best epoch: {}'.format(best_epoch))
                # determine if to record a new checkpoint based on best test loss
                if (len(test_loss) - 1) == best_epoch:
                    # save tf checkpoint
                    saver = tf.train.Saver(tf.global_variables())
                    checkpoint_filepath = os.path.join(output_folder, 'checkpoint', 'model.ckpt')
                    saver.save(sess, checkpoint_filepath)

                # break
                if len(test_loss) - best_epoch > terminate_after_num_epochs_without_test_loss_improvement:
                    break  # break the epoch loop
                epoch = epoch + 1

    finally:
        train_reader.shutdown()
        test_reader.shutdown()


train_model()
