# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os


def get_available_gpu_count():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpus_names = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpus_names)

NUM_GPUS = get_available_gpu_count()
print('Found {} GPUS'.format(NUM_GPUS))
GPU_IDS = list(range(NUM_GPUS))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
str_gpu_ids = ','.join(str(g) for g in GPU_IDS)
os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu_ids  # "0, 1" for multiple
LEARNING_RATE = 1e-4
READER_COUNT = NUM_GPUS
print('Reader Count: {}'.format(READER_COUNT))
TILE_SIZE = 512 # fixed based on the lmdb



parser = argparse.ArgumentParser(prog='train_segnet', description='Script which trains a segnet model')

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
valid_location = False
if gradient_update_location == 'cpu':
    valid_location = True
    gradient_update_location = gradient_update_location + ':0' # append the useless id number
for id in GPU_IDS:
    if gradient_update_location == 'gpu:{}'.format(id):
        valid_location = True
if not valid_location:
    print("Invalid option for 'gradient_update_location': {}".format(gradient_update_location))
    exit(1)


print('Arguments:')
print('batch_size = {}'.format(batch_size))
print('epoch_count = {}'.format(epoch_count))
print('data_dir = {}'.format(data_dir))
print('output folder = {}'.format(output_folder))
print('gradient_update_location = {}'.format(gradient_update_location))


import numpy as np
import tensorflow as tf
import segnet_model
import shutil
import imagereader
import csv
import time


def save_frozen_model(sess, frozen_model_folder, epoch):
    #     tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #     for t in tensors:
    #         print(t)

    from tensorflow.python.tools import optimize_for_inference_lib

    # You can feed data to the IteratorGetNext node using feed_dict
    input_node_name = 'tower_0/IteratorGetNext'
    output_node_name = 'tower_0/segnet/logits/BiasAdd'

    # Run inference on the trained model:
    graph = tf.get_default_graph()
    # batch_x = graph.get_tensor_by_name(input_node_name + ':0')
    # networkout = graph.get_tensor_by_name(output_node_name + ':0')
    # testdata = np.random.randn(batch_size, 1, TILE_SIZE, TILE_SIZE)
    # testlabel = np.random.randn(batch_size, int(TILE_SIZE/yolov3rpn.NETWORK_DOWNSAMPLE_FACTOR), int(TILE_SIZE/yolov3rpn.NETWORK_DOWNSAMPLE_FACTOR))
    # # This will evaluate the model
    # label = sess.run(networkout, feed_dict={batch_x: testdata})

    # Freeze model and create a UFF file:
    graph_def = graph.as_graph_def(add_shapes=True)  # Convert the graph to a serialized pb
    # graph_def = graph.as_graph_def()
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                    graph_def, [output_node_name])
    frozen_graph_def = tf.graph_util.remove_training_nodes(frozen_graph_def)
    # print('Saving frozen tensorflow graph_def')
    # proto_file = os.path.join(frozen_model_folder, 'frozen_graph-{}.pb'.format(epoch))
    # with open(proto_file, "wb") as file:
    #     file.write(frozen_graph_def.SerializeToString())
    # print('   done')

    opt_graph_def = optimize_for_inference_lib.optimize_for_inference(
        frozen_graph_def, [input_node_name], [output_node_name],
        tf.float32.as_datatype_enum)
    print('Saving optimized frozen tensorflow graph_def')
    proto_file = os.path.join(frozen_model_folder, 'opt_frozen_graph-{}.pb'.format(epoch))
    with open(proto_file, "wb") as file:
        file.write(opt_graph_def.SerializeToString())
    print('   done')




def load_block_list(filepath):
    A = []
    if os.path.exists(filepath):
        with open(filepath) as csvfile:
            A = csvfile.readlines()
        A = [x.strip().replace('.csv', '') for x in A]
    return A


def save_csv_file(output_folder, data, filename):
    np.savetxt(os.path.join(output_folder, filename), np.asarray(data), fmt='%.6g', delimiter=",")


def save_conf_csv_file(output_folder, TP, TN, FP, FN, filename):
    a = np.reshape(np.asarray(TP), (len(TP), 1))
    b = np.reshape(np.asarray(TN), (len(TN), 1))
    c = np.reshape(np.asarray(FP), (len(FP), 1))
    d = np.reshape(np.asarray(FN), (len(FN), 1))
    dat = np.hstack((a, b, c, d))
    np.savetxt(os.path.join(output_folder, filename), dat, fmt='%.6g', delimiter=",", header='TP, TN, FP, FN')


def save_text_csv_file(output_folder, data, filename):
    with open(os.path.join(output_folder, filename), 'w') as csvfile:
        for i in range(len(data)):
            csvfile.write(data[i])
            csvfile.write('\n')


def generate_plots(output_folder, train_loss, train_accuracy, test_loss, test_loss_sigma, test_accuracy, test_accuracy_sigma, nb_batches):
    mpl.rcParams['agg.path.chunksize'] = 10000  # fix for error in plotting large numbers of points

    # generate the loss and accuracy plots
    train_loss = np.array(train_loss)
    train_accuracy = np.array(train_accuracy)
    test_loss = np.array(test_loss)
    test_loss_sigma = np.array(test_loss_sigma) * 1.96 # convert to 95% CI
    test_accuracy = np.array(test_accuracy)
    if test_accuracy_sigma is None:
        test_accuracy_sigma = np.zeros((test_accuracy.shape))
    test_accuracy_sigma = np.array(test_accuracy_sigma) * 1.96 # convert to 95% CI
    nb_batches = np.array(nb_batches, dtype=np.float32)

    iterations = np.arange(0, len(train_loss))
    test_iterations = np.arange(0, len(test_accuracy))

    dot_size = 4
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = plt.gca()
    ax.scatter(iterations / nb_batches, train_accuracy, c='b', s=dot_size)
    ax.errorbar(test_iterations, test_accuracy, yerr=test_accuracy_sigma, fmt='r--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    fig.savefig(os.path.join(output_folder, 'accuracy.png'))

    fig.clf()
    ax = plt.gca()
    ax.scatter(iterations / nb_batches, train_accuracy, c='b', s=dot_size)
    ax.errorbar(test_iterations, test_accuracy, yerr=test_accuracy_sigma, fmt='r--')
    plt.ylim((0.9, 1.0))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    fig.savefig(os.path.join(output_folder, 'accuracy_above0.9.png'))

    fig.clf()
    ax = plt.gca()
    ax.scatter(iterations / nb_batches, train_accuracy, c='b', s=dot_size)
    ax.errorbar(test_iterations, test_accuracy, yerr=test_accuracy_sigma, fmt='r--')
    plt.ylim((0.99, 1.0))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    fig.savefig(os.path.join(output_folder, 'accuracy_above0.99.png'))

    fig.clf()
    ax = plt.gca()
    ax.scatter(iterations / nb_batches, train_loss, c='b', s=dot_size)
    ax.errorbar(test_iterations, test_loss, yerr=test_loss_sigma, fmt='r--')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    fig.savefig(os.path.join(output_folder, 'loss.png'))

    fig.clf()
    ax = plt.gca()
    ax.scatter(iterations / nb_batches, train_loss, c='b', s=dot_size)
    ax.errorbar(test_iterations, test_loss, yerr=test_loss_sigma, fmt='r--')
    ax.set_yscale('log')
    plt.ylim((np.min(train_loss), np.max(train_loss)))
    plt.ylabel('Loss (log scale)')
    plt.xlabel('Epoch')
    ax.set_yscale('log')
    fig.savefig(os.path.join(output_folder, 'loss_logscale.png'))

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

    # create frozen model folder
    frozen_model_folder = os.path.join(output_folder, 'frozen-model')
    if os.path.exists(frozen_model_folder):
        shutil.rmtree(frozen_model_folder)
    os.mkdir(frozen_model_folder)

    # start the input queues
    try:
        print('Starting Readers')
        train_reader.startup()
        test_reader.startup()

        print('Creating model')
        with tf.Graph().as_default(), tf.device('/' + gradient_update_location):
            is_training_placeholder = tf.placeholder(tf.bool, name='is_training')

            print('Creating Input Train Dataset')
            # wrap the input queues into a Dataset
            image_shape = tf.TensorShape((batch_size, TILE_SIZE, TILE_SIZE, 1))
            label_shape = tf.TensorShape((batch_size, TILE_SIZE, TILE_SIZE))
            train_dataset = tf.data.Dataset.from_generator(train_reader.generator, output_types=(tf.float32, tf.int32), output_shapes=(image_shape, label_shape))
            train_dataset = train_dataset.prefetch(2*NUM_GPUS) # prefetch N batches
            # train_iterator = train_dataset.make_initializable_iterator()

            print('Creating Input Test Dataset')
            test_dataset = tf.data.Dataset.from_generator(test_reader.generator, output_types=(tf.float32, tf.int32), output_shapes=(image_shape, label_shape))
            test_dataset = test_dataset.prefetch(2*NUM_GPUS)  # prefetch N batches
            # test_iterator = test_dataset.make_initializable_iterator()

            print('Converting Datasets to Iterator')
            # create a iterator of the correct shape and type
            iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
            train_init_op = iter.make_initializer(train_dataset)
            test_init_op = iter.make_initializer(test_dataset)

            optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

            # Calculate the gradients for each model tower.
            tower_grads = []
            ops_per_gpu = {}
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(NUM_GPUS):
                    print('Building tower for GPU:{}'.format(i))
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % (segnet_model.TOWER_NAME, i)) as scope:
                            # Dequeues one batch for the GPU
                            image_batch, label_batch = iter.get_next()

                            # Calculate the loss for one tower of the CIFAR model. This function
                            # constructs the entire CIFAR model but shares the variables across
                            # all towers.
                            loss_op, accuracy_op = segnet_model.tower_loss(image_batch, label_batch, train_reader.get_number_of_classes(), is_training_placeholder)
                            ops_per_gpu['gpu{}-loss'.format(i)] = loss_op
                            ops_per_gpu['gpu{}-accuracy'.format(i)] = accuracy_op

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            # Calculate the gradients for the batch of data on this tower.
                            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                                grads = optimizer.compute_gradients(loss_op)

                            # Keep track of the gradients across all towers.
                            tower_grads.append(grads)

            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            print('Setting up Average Gradient')
            grads = segnet_model.average_gradients(tower_grads)

            # create merged accuracy stats
            print('Setting up Averaged Accuracy')
            all_loss_sum = tf.constant(0, dtype=tf.float32)
            all_accuracy_sum = tf.constant(0, dtype=tf.float32)

            for i in range(NUM_GPUS):
                all_loss_sum = tf.add(all_loss_sum, ops_per_gpu['gpu{}-loss'.format(i)])
                all_accuracy_sum = tf.add(all_accuracy_sum, ops_per_gpu['gpu{}-accuracy'.format(i)])

            all_loss = tf.divide(all_loss_sum, tf.constant(NUM_GPUS, dtype=tf.float32))
            all_accuracy = tf.divide(all_accuracy_sum, tf.constant(NUM_GPUS, dtype=tf.float32))

            # Apply the gradients to adjust the shared variables.
            print('Setting up Optimizer')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.apply_gradients(grads)

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            print('Starting Session')
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            init = tf.global_variables_initializer()
            sess.run(init)

            train_loss = list()
            train_accuracy = list()
            test_loss = list()
            test_loss_sigma = list()
            test_accuracy = list()
            test_accuracy_sigma = list()
            test_loss.append(1)
            test_loss_sigma.append(0)
            test_accuracy.append(0)
            test_accuracy_sigma.append(0)

            print('Running Network')
            for epoch in range(epoch_count):
                print('---- Epoch: {} ----'.format(epoch))

                print('initializing training data iterator')
                sess.run(train_init_op)
                print('   iterator init complete')

                start_time = time.time()
                train_epoch_size = train_reader.get_epoch_size()
                adj_batch_count = int(np.ceil(train_epoch_size/NUM_GPUS))

                print('Training:')
                for step in range(adj_batch_count):
                    _, loss_val, accuracy_val = sess.run([train_op, all_loss, all_accuracy], feed_dict={is_training_placeholder: True})
                    train_loss.append(loss_val)
                    train_accuracy.append(accuracy_val)

                    if step % 100 == 0:
                        print('Epoch: {} Batch: {}/{}, Loss = {}, Accuracy = {}'.format(epoch, step*NUM_GPUS, train_epoch_size, train_loss[-1], train_accuracy[-1]))

                print('Train Accuracy = {}'.format(np.mean(train_accuracy[-train_epoch_size:])))
                print('  Batch Count= {}'.format(train_epoch_size))
                print('  took: {}'.format(time.time() - start_time))

                print('initializing test data iterator')
                sess.run(test_init_op)
                print('   iterator init complete')

                start_time = time.time()
                test_epoch_size = test_reader.get_epoch_size()
                adj_batch_count = int(np.ceil(test_epoch_size / NUM_GPUS))
                epoch_test_loss = list()
                epoch_test_accuracy = list()

                print('Testing:')
                for step in range(adj_batch_count):
                    loss_val, accuracy_val = sess.run([all_loss, all_accuracy], feed_dict={is_training_placeholder: False})
                    epoch_test_loss.append(loss_val)
                    epoch_test_accuracy.append(accuracy_val)
                    if step % 100 == 0:
                        print('Epoch: {} Batch: {}/{}, Loss = {}, Accuracy = {}'.format(epoch, step*NUM_GPUS, test_epoch_size, epoch_test_loss[-1], epoch_test_accuracy[-1]))
                test_loss.append(np.mean(epoch_test_loss))
                test_loss_sigma.append(np.std(epoch_test_loss))
                test_accuracy.append(np.mean(epoch_test_accuracy))
                test_accuracy_sigma.append(np.std(epoch_test_accuracy))

                print('Test Loss = {}'.format(test_loss[-1]))
                print('Test Loss Sigma= {}'.format(test_loss_sigma[-1]))
                print('Test Accuracy = {}'.format(test_accuracy[-1]))
                print('Test Accuracy Sigma = {}'.format(test_accuracy_sigma[-1]))
                print('  Batch Count= {}'.format(test_epoch_size))
                print('  took: {}'.format(time.time() - start_time))

                generate_plots(output_folder, train_loss, train_accuracy, test_loss, test_loss_sigma, test_accuracy, test_accuracy_sigma, train_epoch_size/NUM_GPUS)
                save_csv_file(output_folder, train_accuracy, 'train_accuracy.csv')
                save_csv_file(output_folder, train_loss, 'train_loss.csv')
                save_csv_file(output_folder, test_loss, 'test_loss.csv')
                save_csv_file(output_folder, test_loss_sigma, 'test_loss_sigma.csv')
                save_csv_file(output_folder, test_accuracy, 'test_accuracy.csv')
                save_csv_file(output_folder, test_accuracy_sigma, 'test_accuracy_sigma.csv')

                # save tf checkpoint
                saver = tf.train.Saver(tf.global_variables())
                checkpoint_filepath = os.path.join(output_folder, 'checkpoint', 'model.ckpt-{}'.format(epoch))
                saver.save(sess, checkpoint_filepath)

                # save a frozen copy of the graph
                save_frozen_model(sess, frozen_model_folder, epoch)
            sess.close()

    finally:
        train_reader.shutdown()
        test_reader.shutdown()


train_model()
