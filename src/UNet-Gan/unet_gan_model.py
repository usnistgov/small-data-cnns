# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import tensorflow as tf

NUMBER_CHANNELS = 1  # grayscale

BASELINE_FEATURE_DEPTH = 64
KERNEL_SIZE = 3
DECONV_KERNEL_SIZE = 2
POOLING_STRIDE = 2
Z_DIM = 100


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def conv_layer(layer_in, fc_out, kernel, is_training, stride=1, name=None):
    conv_output = tf.layers.conv2d(
        layer_in,
        filters=fc_out,
        kernel_size=kernel,
        padding='same',
        activation=tf.nn.leaky_relu,
        strides=stride,
        data_format='channels_first',  # translates to NCHW
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name=name)
    conv_output = tf.layers.batch_normalization(conv_output, training=is_training, axis=1, name=name)  # axis to normalize (the channels dimension)
    return conv_output


def deconv_layer(layer_in, fc_out, kernel, is_training, stride=1, name=None):
    conv_output = tf.layers.conv2d_transpose(
        layer_in,
        filters=fc_out,
        kernel_size=kernel,
        padding='same',
        activation=None,
        strides=stride,
        data_format='channels_first',  # translates to NCHW
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        name=name)
    conv_output = tf.layers.batch_normalization(conv_output, training=is_training, axis=1, name=name)  # axis to normalize (the channels dimension)
    return conv_output


def add_discriminator_ops(inputs, is_training, img_size):
    with tf.variable_scope("encoder"):
        conv_1 = conv_layer(inputs, BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)
        conv_1 = conv_layer(conv_1, BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)

        pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=POOLING_STRIDE, strides=POOLING_STRIDE, data_format='channels_first')

        conv_2 = conv_layer(pool_1, 2 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)
        conv_2 = conv_layer(conv_2, 2 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)

        pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=POOLING_STRIDE, strides=POOLING_STRIDE, data_format='channels_first')

        conv_3 = conv_layer(pool_2, 4 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)
        conv_3 = conv_layer(conv_3, 4 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)

        pool_3 = tf.layers.max_pooling2d(conv_3, pool_size=POOLING_STRIDE, strides=POOLING_STRIDE, data_format='channels_first')

        conv_4 = conv_layer(pool_3, 8 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)
        conv_4 = conv_layer(conv_4, 8 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)

        pool_4 = tf.layers.max_pooling2d(conv_4, pool_size=POOLING_STRIDE, strides=POOLING_STRIDE, data_format='channels_first')

        # bottleneck
        bottleneck = conv_layer(pool_4, 16 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)
        bottleneck = conv_layer(bottleneck, 16 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)

        N = int((16 * BASELINE_FEATURE_DEPTH) * (img_size / 16) * (img_size / 16))
        bottleneck = tf.reshape(bottleneck, [-1, N])

        logits = tf.layers.dense(bottleneck, 1, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return logits


def add_generator_ops(inputs, is_training, img_size):
    with tf.variable_scope("decoder"):

        # bottleneck must be [N, 1024, 16, 16]
        # bottleneck must be [N, (16 * BASELINE_FEATURE_DEPTH), (img_size/16), (img_size/16)]
        N = int((16 * BASELINE_FEATURE_DEPTH) * (img_size/16) * (img_size/16))
        inputs = tf.stop_gradient(inputs)

        bottleneck = tf.layers.dense(inputs, N, activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        bottleneck = tf.reshape(bottleneck, [-1, int(16 * BASELINE_FEATURE_DEPTH), int(img_size/16), int(img_size/16)])

        # up-conv which reduces the number of feature channels by 2
        deconv_4 = deconv_layer(bottleneck, 8 * BASELINE_FEATURE_DEPTH, DECONV_KERNEL_SIZE, is_training, stride=POOLING_STRIDE)
        # deconv_4 = tf.concat([conv_4, deconv_4], axis=1) # TODO to perform full ED to DE swap need to pad with zeros to match conv sizes
        deconv_4 = conv_layer(deconv_4, 8 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)
        deconv_4 = conv_layer(deconv_4, 8 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)

        deconv_3 = deconv_layer(deconv_4, 4 * BASELINE_FEATURE_DEPTH, DECONV_KERNEL_SIZE, is_training, stride=POOLING_STRIDE)
        # deconv_3 = tf.concat([conv_3, deconv_3], axis=1)
        deconv_3 = conv_layer(deconv_3, 4 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)
        deconv_3 = conv_layer(deconv_3, 4 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)

        deconv_2 = deconv_layer(deconv_3, 2 * BASELINE_FEATURE_DEPTH, DECONV_KERNEL_SIZE, is_training, stride=POOLING_STRIDE)
        # deconv_2 = tf.concat([conv_2, deconv_2], axis=1)
        deconv_2 = conv_layer(deconv_2, 2 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)
        deconv_2 = conv_layer(deconv_2, 2 * BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)

        deconv_1 = deconv_layer(deconv_2, BASELINE_FEATURE_DEPTH, DECONV_KERNEL_SIZE, is_training, stride=POOLING_STRIDE)
        # deconv_1 = tf.concat([conv_1, deconv_1], axis=1)
        deconv_1 = conv_layer(deconv_1, BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)
        deconv_1 = conv_layer(deconv_1, BASELINE_FEATURE_DEPTH, KERNEL_SIZE, is_training)

        conv_output = tf.layers.conv2d(
            deconv_1,
            filters=1,
            kernel_size=1,
            padding='same',
            activation=tf.nn.tanh,
            strides=1,
            data_format='channels_first',  # translates to NCHW
            kernel_initializer=tf.contrib.layers.xavier_initializer())

    return conv_output


def tower_loss(images, noise_z, is_training, img_size):
    with tf.variable_scope('unet', reuse=tf.AUTO_REUSE):
        # setup generator
        G = add_generator_ops(noise_z, is_training, img_size)
        D_real_logits = add_discriminator_ops(images, is_training, img_size)
    with tf.variable_scope('unet', reuse=True):
        D_fake_logits = add_discriminator_ops(G, is_training, img_size)

    with tf.variable_scope('unet'):
        with tf.variable_scope("encoder"):
            # d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.zeros_like(D_real_logits)))
            # d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))
            d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.random.uniform(tf.shape(D_real_logits), minval=0.9, maxval=1.0)))
            d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.random.uniform(tf.shape(D_real_logits), minval=0.0, maxval=0.1)))
            d_loss = d_loss_real + d_loss_fake
        with tf.variable_scope("decoder"):
            # g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
            g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.random.uniform(tf.shape(D_real_logits), minval=0.9, maxval=1.0)))

    return g_loss, d_loss, G


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def build_input_iterators(train_reader, batch_prefetch_count, tile_size):
    img_size = train_reader.get_image_size()
    batch_size = None #train_reader.get_batch_size()

    if img_size[0] != tile_size or img_size[1] != tile_size:
        print('Image Size: {}, {}'.format(img_size[0], img_size[1]))
        print('Expected Size: {}, {}'.format(tile_size, tile_size))
        raise Exception('Invalid input shape, does not match specified network tile size.')

    print('Creating Input Train Dataset')
    # wrap the input queues into a Dataset
    # this sets up the imagereader class as a Python generator
    image_shape = tf.TensorShape((batch_size, 1, img_size[0], img_size[1]))
    znoise_shape = tf.TensorShape((batch_size, Z_DIM))
    train_dataset = tf.data.Dataset.from_generator(train_reader.generator, output_types=(tf.float32, tf.float32), output_shapes=(image_shape, znoise_shape))
    train_dataset = train_dataset.prefetch(batch_prefetch_count)  # prefetch N batches

    print('Converting Datasets to Iterator')
    # create a iterator of the correct shape and type
    # the input iterator (feeding from one of the two imagereader generators) can be switched as needed by running the appropriate init_op
    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    train_init_op = iter.make_initializer(train_dataset)

    return train_init_op, iter


def build_towered_model(train_reader, gpu_ids, learning_rate):
    is_training_placeholder = tf.placeholder(tf.bool, name='is_training')

    num_gpus = len(gpu_ids)
    batch_prefetch_count = num_gpus
    tile_size = train_reader.get_image_size()[0]

    if train_reader.get_image_size()[0] != train_reader.get_image_size()[1]:
        raise Exception("Input tiles from database must be square and multiple of 16.")

    if tile_size % 16 != 0:
        raise Exception('Input Image tile size needs to be a multiple of 16 to allow integer sized downscaled feature maps')

    train_init_op, iter = build_input_iterators(train_reader, batch_prefetch_count, tile_size)

    # configure the Adam optimizer for network training with the specified learning reate
    optimizer_g = tf.train.AdamOptimizer(learning_rate)
    optimizer_d = tf.train.AdamOptimizer(learning_rate)

    # Calculate the gradients for each model tower.
    tower_grads_g = []
    tower_grads_d = []
    ops_per_gpu = {}
    with tf.variable_scope(tf.get_variable_scope()):

        for i in gpu_ids:
            print('Building tower for GPU:{}'.format(i))
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    # Dequeues one batch for the GPU
                    image_batch, z_batch = iter.get_next()

                    # Calculate the loss for one tower of the model. This function
                    # constructs the entire model but shares the variables across
                    # all towers.
                    if i == 0:
                        g_loss_op, d_loss_op, G_op = tower_loss(image_batch, z_batch, is_training_placeholder, tile_size)
                    else:
                        g_loss_op, d_loss_op, _ = tower_loss(image_batch, z_batch, is_training_placeholder, tile_size)
                    ops_per_gpu['gpu{}-g_loss'.format(i)] = g_loss_op
                    ops_per_gpu['gpu{}-d_loss'.format(i)] = d_loss_op

                    # Reuse variables for the next tower.
                    tf.get_variable_scope().reuse_variables()

                    t_vars = tf.trainable_variables()
                    d_vars = [var for var in t_vars if 'encoder' in var.name]
                    g_vars = [var for var in t_vars if 'decoder' in var.name]

                    # Calculate the gradients for the batch of data on this tower.
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        g_grads = optimizer_g.compute_gradients(g_loss_op, var_list=g_vars)
                        d_grads = optimizer_d.compute_gradients(d_loss_op, var_list=d_vars)

                    # Keep track of the gradients across all towers.
                    tower_grads_g.append(g_grads)
                    tower_grads_d.append(d_grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    print('Setting up Average Gradient')
    grads_g = average_gradients(tower_grads_g)
    grads_d = average_gradients(tower_grads_d)

    # create merged accuracy stats
    print('Setting up Averaged Accuracy')
    all_g_loss_sum = tf.constant(0, dtype=tf.float32)
    all_d_loss_sum = tf.constant(0, dtype=tf.float32)

    for i in gpu_ids:
        all_g_loss_sum = tf.add(all_g_loss_sum, ops_per_gpu['gpu{}-g_loss'.format(i)])
        all_d_loss_sum = tf.add(all_d_loss_sum, ops_per_gpu['gpu{}-d_loss'.format(i)])

    loss_g_op = tf.divide(all_g_loss_sum, tf.constant(num_gpus, dtype=tf.float32))
    loss_d_op = tf.divide(all_d_loss_sum, tf.constant(num_gpus, dtype=tf.float32))

    # Apply the gradients to adjust the shared variables.
    print('Setting up Optimizer')
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op_g = optimizer_g.apply_gradients(grads_g)
        train_op_d = optimizer_d.apply_gradients(grads_d)
        train_op = tf.group(train_op_g, train_op_d)

    return train_init_op, train_op, loss_g_op, loss_d_op, is_training_placeholder, G_op
