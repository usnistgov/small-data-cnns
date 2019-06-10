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
print('Reader Count: {}'.format(READER_COUNT))


# Setup the Argument parsing
parser = argparse.ArgumentParser(prog='train_unet', description='Script which trains a unet model')

parser.add_argument('--batch_size', dest='batch_size', type=int, help='training batch size', default=4)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-5)
parser.add_argument('--output_dir', dest='output_folder', type=str, help='Folder where outputs will be saved (Required)', required=True)
parser.add_argument('--test_every_n_steps', dest='test_every_n_steps', type=int, help='number of gradient update steps to take between test epochs', default=500)

parser.add_argument('--use_augmentation', dest='use_augmentation', type=int, help='whether to use data augmentation [0 = false, 1 = true]', default=0)

parser.add_argument('--train_database', dest='train_database_filepath', type=str, help='lmdb database to use for (Required)', required=True)
parser.add_argument('--early_stopping', dest='terminate_after_num_epochs_without_test_loss_improvement', type=int, help='Perform early stopping when the test loss does not improve for N epochs.', default=10)
parser.add_argument('--gradient_update_location', dest='gradient_update_location', type=str, help="Where to perform gradient averaging and update. Options: ['cpu', 'gpu:#']. Use the GPU if you have a fully connected topology, cpu otherwise.", default='gpu:0')
parser.add_argument('--restore_checkpoint_filepath', dest='restore_checkpoint_filepath', type=str, help='checkpoint to resume from', default=None)

args = parser.parse_args()
batch_size = args.batch_size
output_folder = args.output_folder
gradient_update_location = args.gradient_update_location
terminate_after_num_epochs_without_test_loss_improvement = args.terminate_after_num_epochs_without_test_loss_improvement
train_lmdb_filepath = args.train_database_filepath
learning_rate = args.learning_rate
restore_checkpoint_filepath = args.restore_checkpoint_filepath
test_every_n_steps = args.test_every_n_steps
use_augmentation = args.use_augmentation

# verify gradient_update_location is valid
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
print('learning_rate = {}'.format(learning_rate))
print('test_every_n_steps = {}'.format(test_every_n_steps))
print('use_augmentation = {}'.format(use_augmentation))

print('train_database = {}'.format(train_lmdb_filepath))
print('output folder = {}'.format(output_folder))

print('gradient_update_location = {}'.format(gradient_update_location))
print('early_stopping count = {}'.format(terminate_after_num_epochs_without_test_loss_improvement))
print('restore_checkpoint_filepath = {}'.format(restore_checkpoint_filepath))


import numpy as np
import tensorflow as tf
import unet_gan_model
import imagereader_gan



def save_csv_file(output_folder, data, filename):
    np.savetxt(os.path.join(output_folder, filename), np.asarray(data), fmt='%.6g', delimiter=",")


def save_text_csv_file(output_folder, data, filename):
    with open(os.path.join(output_folder, filename), 'w') as csvfile:
        for i in range(len(data)):
            csvfile.write(data[i])
            csvfile.write('\n')


def plot(output_folder, name, A, B=None, log_scale=False):
    mpl.rcParams['agg.path.chunksize'] = 10000  # fix for error in plotting large numbers of points

    A = np.asarray(A)
    if B is not None:
        B = np.asarray(B)

    dot_size = 4
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = plt.gca()
    ax.scatter(list(range(A.size)), A, c='b', s=dot_size)
    if B is not None:
        ax.scatter(list(range(B.size)), B, c='r', s=dot_size)
    if log_scale:
        ax.set_yscale('log')
    plt.ylabel('{}'.format(name))
    plt.xlabel('Iterations')
    fig.savefig(os.path.join(output_folder, '{}.png'.format(name)))

    plt.close(fig)


def train_model():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(os.path.join(output_folder, 'g_imgs')):
        os.makedirs(os.path.join(output_folder, 'g_imgs'))

    print('Setting up training image reader')
    train_reader = imagereader_gan.ImageReader(train_lmdb_filepath, batch_size=batch_size, use_augmentation=use_augmentation, shuffle=True, num_workers=READER_COUNT)
    print('Train Reader has {} batches'.format(train_reader.get_epoch_size()))

    try: # if any errors happen we want to catch them and shut down the multiprocess readers
        print('Starting Readers')
        train_reader.startup()

        print('Creating model')
        with tf.Graph().as_default(), tf.device('/' + gradient_update_location):
            train_init_op, train_op, loss_g_op, loss_d_op, is_training_placeholder, G_op = unet_gan_model.build_towered_model(train_reader, GPU_IDS, learning_rate)

            print('*********************************')
            print('UNet-GAN Vars')
            vars = tf.global_variables()
            for v in vars:
                print('{}   {}'.format(v._shared_name, v.shape))
            print('*********************************')

            print('Starting Session')
            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            init = tf.global_variables_initializer()
            sess.run(init)

            if restore_checkpoint_filepath is not None:
                print('Loading checkpoint weights')
                # build list of variables to restore
                vars = tf.global_variables()
                vars = [v for v in vars if 'Adam' not in v._shared_name]  # remove Adam variables
                vars = [v for v in vars if 'logits' not in v._shared_name]  # remove output layer variables

                saver = tf.train.Saver(vars)
                saver.restore(sess, restore_checkpoint_filepath)

            # setup network accuracy tracking variables
            train_loss = list()
            train_loss_d = list()
            train_loss_g = list()

            epoch_loss_d = list()
            epoch_loss_g = list()

            train_epoch_size = train_reader.get_epoch_size()
            train_epoch_size = test_every_n_steps

            epoch = 0
            print('Running Network')
            while True: # loop until early stopping
                print('---- Epoch: {} ----'.format(epoch))

                print('initializing training data iterator')
                sess.run(train_init_op)
                print('   iterator init complete')

                adj_batch_count = train_epoch_size
                for step in range(adj_batch_count):
                    if step % 100 == 0:
                        _, loss_g_val, loss_d_val, G = sess.run([train_op, loss_g_op, loss_d_op, G_op], feed_dict={is_training_placeholder: True})
                        G = np.squeeze(G[0, ])
                        G = G * 64 + 127
                        G = G.astype(np.uint8)
                        imagereader_gan.imwrite(G, os.path.join(output_folder, 'g_imgs', 'img_e{:04d}_s{:06d}.tif'.format(epoch, step)))
                    else:
                        _, loss_g_val, loss_d_val = sess.run([train_op, loss_g_op, loss_d_op], feed_dict={is_training_placeholder: True})

                    train_loss.append(loss_g_val + loss_d_val)
                    train_loss_g.append(loss_g_val)
                    train_loss_d.append(loss_d_val)
                    print('Train Epoch: {} Batch {}/{}: loss_g = {}, loss_d = {}'.format(epoch, step, adj_batch_count, loss_g_val, loss_d_val))

                epoch_loss_g.append(np.mean(train_loss_g[-test_every_n_steps:]))
                epoch_loss_d.append(np.mean(train_loss_d[-test_every_n_steps:]))
                plot(output_folder, 'loss', train_loss)
                plot(output_folder, 'loss_g', train_loss_g)
                plot(output_folder, 'loss_d', train_loss_d)
                plot(output_folder, 'loss_components', train_loss_g, train_loss_d)

                plot(output_folder, 'epoch_loss_g', epoch_loss_g)
                plot(output_folder, 'epoch_loss_d', epoch_loss_d)
                plot(output_folder, 'epoch_loss_components', epoch_loss_g, epoch_loss_d)

                save_csv_file(output_folder, epoch_loss_g, 'epoch_loss_g.csv')
                save_csv_file(output_folder, epoch_loss_d, 'epoch_loss_d.csv')
                save_csv_file(output_folder, train_loss_g, 'train_loss_g.csv')
                save_csv_file(output_folder, train_loss_d, 'train_loss_d.csv')
                save_csv_file(output_folder, train_loss, 'train_loss.csv')

                # save tf checkpoint
                saver = tf.train.Saver(tf.global_variables())
                checkpoint_filepath = os.path.join(output_folder, 'checkpoint', 'model.ckpt')
                saver.save(sess, checkpoint_filepath)

                epoch = epoch + 1
                if epoch > 100:
                    break
    finally: # if any erros happened during training, shut down the disk readers
        print('Shutting down train_reader')
        train_reader.shutdown()


if __name__ == "__main__":
    train_model()
    import gc
    gc.collect() # https://github.com/tensorflow/tensorflow/issues/21277
