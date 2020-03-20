# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:45:12 2016

@author: psm; modified by pnb on 2018-10-25
"""

from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
#from IPython.display import display, Image
from scipy import ndimage
#from sklearn.linear_model import LogisticRegression
#from six.moves.urllib.request import urlretrieve
#from six.moves import cPickle as pickle
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
from skimage.transform import resize
import math
import tensorflow as tf
from six.moves import range
import scipy.misc
from skimage.transform import resize
import tensorflow.contrib.slim as slim
from  scipy.ndimage.morphology import distance_transform_edt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from ops import *
from myutils import *

import argparse 

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

#test_batch_size = test_dataset.shape[0]/num_test_steps 
#patch_size = 7
#patch_size2 = 5
#patch_size3 = 3
#depth = 64
#depth2=64
#depth3=64

#batch_size = 16
#image_size = 256
#num_labels = 2
#num_channels = 1 # grayscale

#######################################################
parser = argparse.ArgumentParser(description='U-Net + GAN inference segmentation')
parser.add_argument('--testdataset_dir', dest='testdataset_dir', default='images', help='path of the test dataset')
parser.add_argument('--testmasks_dir', dest='testmasks_dir', default='masks', help='path of the test masks')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='images', help='path of the dataset')
parser.add_argument('--masks_dir', dest='masks_dir', default='masks', help='path of the masks')
parser.add_argument('--model_to_load', dest='model_to_load', default='gan_model2', help='path to the already saved GAN model')
parser.add_argument('--save_dir', dest='save_dir', default='unet_save', help='path to save the trained model')
parser.add_argument('--epoch', dest='epoch', type=int, default=2, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--image_size', dest='image_size', type=int, default=128, help='scale images to tiles of this size')
parser.add_argument('--gf_dim', dest='gf_dim', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--df_dim', dest='df_dim', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='num_channels', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=5, help='# of output image channels')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0001, help='learning rate for Adams optimizer')
#parser.add_argument('--invert', dest='invert', type=int, default=1, help='Postprocessing: open, skeletonize and invert image')
parser.add_argument('--validation_ratio', dest='validation_ratio', type=float, default=0.05, help='percentage of images to be used for validation')
parser.add_argument('--saveim', dest='saveim', type=bool, default=True, help='save validation predictions')

args = parser.parse_args()

num_steps = args.epoch #10000
batch_size=args.batch_size
image_size = args.image_size #int(args.split_size)
num_labels = args.output_nc
num_channels = args.num_channels # grayscale

learning_rate = float(args.learning_rate)

model_to_load = args.model_to_load #'/home/pnb/gan_model2/DCGAN.model-18001'
print("model_to_load = ", model_to_load)

#postprocess = int(args.invert)#int(sys.argv[4])   
    

#############################################################################################################
#############################################################################################################


#train_seg_folders = '/home/psm/RPE/Data/RPE_z01_Train_Test/ZO1/TestLabels/' # this is equivalwnt to args.masks_dir
#train_org_folders = '/home/psm/RPE/Data/RPE_z01_Train_Test/ZO1/TestImages/' # this is equivalent to args.dataset_dir

# not used in the original code
#gan_org_folders = '/home/psm/Augmentation/data/RPEsGAN/GAN/fluorescent_gaussian_minimum/'
#gan_seg_folders = '/home/psm/Augmentation/data/RPEsGAN/GAN/mask_median/'

#test_seg_folders = '/home/psm/RPE/Data/RPE_z01_Train_Test/ZO1/TrainLabels/' # this is equivalent to args.testdataset_dir
#test_org_folders = '/home/psm/RPE/Data/RPE_z01_Train_Test/ZO1/TrainImages/' # this is equivalent to args.testmasks_dir

# print('Loading images...')
# crop_size = 256
# #image_size = 256 
# #big_size = 256
# list_of_org_test = load_images_crop(args.testmasks_dir,crop_size)
# list_of_seg_test = load_masks_crop(args.testdataset_dir,crop_size)     
 
# test_dataset = np.array(list_of_org_test)
# test_labels = np.array(list_of_seg_test)
# print('Test set', test_dataset.shape, test_labels.shape)  


# list_of_org_train = load_images_crop(args.dataset_dir,crop_size) 
# list_of_seg_train = load_masks_crop(args.masks_dir,crop_size) 
# list_of_weights = load_images_weights_distance_crop(args.masks_dir,crop_size)




# train_dataset = np.array(list_of_org_train)
# train_labels = np.array(list_of_seg_train)
# train_weights = np.array(list_of_weights)

# train_dataset, train_labels, train_weights, valid_dataset, valid_labels = separate_train_test(train_dataset, train_labels,train_weights, 0.75)

# print('Training set', train_dataset.shape, train_labels.shape, train_weights.shape)
# print('Validation set',valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)  

  
  

# train_dataset, train_labels = reformat(train_dataset, train_labels)
# train_weights = reformat_weights(train_weights)
# test_dataset, test_labels = reformat(test_dataset, test_labels)
# valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

# print('Training set', train_dataset.shape, train_labels.shape, train_weights.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)

#############################################################################################################


print('Loading images...')
 
#crop_size = 256
list_of_org_test,list_of_org_names = load_images_with_names(args.testdataset_dir)

list_of_org_test = load_images_crop(args.testmasks_dir,image_size)
list_of_seg_test = load_masks_crop(args.testdataset_dir,image_size)     
 
test_dataset = np.array(list_of_org_test)
test_labels = np.array(list_of_seg_test)
print('Test set', test_dataset.shape, test_labels.shape)  


list_of_org_train = load_images_crop(args.dataset_dir,args.image_size) 
list_of_seg_train = load_masks_crop(args.masks_dir,args.image_size) 
list_of_weights = load_images_weights_distance_crop(args.masks_dir,image_size)


train_dataset = np.array(list_of_org_train)
train_labels = np.array(list_of_seg_train)
train_weights = np.array(list_of_weights)

train_dataset, train_labels, train_weights, valid_dataset, valid_labels = separate_train_test(train_dataset, train_labels,train_weights, args.validation_ratio)

print('Training set', train_dataset.shape, train_labels.shape, train_weights.shape)
print('Validation set',valid_dataset.shape, valid_labels.shape)


train_dataset, train_labels, train_weights = no_augment_train(train_dataset, train_labels, train_weights)  
  
print('Training set', train_dataset.shape, train_labels.shape, train_weights.shape)
  

train_dataset, train_labels = reformat2(train_dataset, train_labels)
train_weights = reformat_weights(train_weights)
valid_dataset, valid_labels = reformat2(valid_dataset, valid_labels)
test_dataset, test_labels = reformat2(test_dataset, test_labels)



print('Training set', train_dataset.shape, train_labels.shape, train_weights.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)

#############################################################################################################
#############################################################################################################

    
    
def max_pool(net, stride):
  """
  Tensorflow default implementation does not provide gradient operation on max_pool_with_argmax
  Therefore, we use max_pool_with_argmax to extract mask and
  plain max_pool for, eeem... max_pooling.
  """
  with tf.name_scope('MaxPoolArgMax'):
    _, mask = tf.nn.max_pool_with_argmax(
      net,
      ksize=[1, stride, stride, 1],
      strides=[1, stride, stride, 1],
      padding='SAME')
    mask = tf.stop_gradient(mask)
    net = slim.max_pool2d(net, kernel_size=[stride, stride])
    return net, mask


# Thank you, @https://github.com/Pepslee
def max_unpool(net, mask, stride):
  assert mask is not None
  with tf.name_scope('UnPool2D'):
    ksize = [1, stride, stride, 1]
    input_shape = net.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(net)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(net, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret
    
    

graph = tf.Graph()    
    
with graph.as_default():

        
  # Input data.
  tf_train_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_labels))
  tf_train_weights = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  #tf_test_dataset = tf.constant(test_dataset)
  tf_test_dataset = tf.placeholder(
    tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  
  
  d_bn1 = batch_norm(name='d_bn1')
  d_bn2 = batch_norm(name='d_bn2')
  d_bn3 = batch_norm(name='d_bn3')

  g_bn0 = batch_norm(name='g_bn0')
  g_bn1 = batch_norm(name='g_bn1')
  g_bn2 = batch_norm(name='g_bn2')
  g_bn3 = batch_norm(name='g_bn3')
  df_dim=64
  gf_dim=64
  
  with tf.variable_scope("final_layer") as scope:
      w_extra = tf.get_variable('w_extra', [5, 5,  gf_dim, 2*gf_dim],
              initializer=tf.random_normal_initializer(stddev=0.02))
      biases_extra = tf.get_variable('biases_extra', gf_dim, initializer=tf.constant_initializer(0.0))
      
      w_final = tf.get_variable('w_final', [5, 5,  num_labels, gf_dim],
              initializer=tf.random_normal_initializer(stddev=0.02))
      biases_final = tf.get_variable('biases_final', num_labels, initializer=tf.constant_initializer(0.0))
      
  
  
       
       
  def unet_model(data, reuse, train=False):
      
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      dh0 = lrelu(conv2d(data, df_dim, name='d_h0_conv'))
      dh1 = lrelu(d_bn1(conv2d(dh0, df_dim*2, name='d_h1_conv')))
      dh2 = lrelu(d_bn2(conv2d(dh1, df_dim*4, name='d_h2_conv')))
      dh3 = lrelu(d_bn3(conv2d(dh2, df_dim*8, name='d_h3_conv')))
      
    with tf.variable_scope("generator") as scope:
        #s_h, s_w = output_height, output_width
        if reuse:
          scope.reuse_variables()

        s_h, s_w = image_size, image_size
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        #self.z_, self.h0_w, self.h0_b = linear(
        #    z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        #self.h0 = tf.reshape(
        #    self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        #h0 = tf.nn.relu(self.g_bn0(self.h0))

        h1, h1_w, h1_b = deconv2d(
            dh3, [batch_size, s_h8, s_w8, gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(g_bn1(h1))
        
        ch1 = tf.concat([h1,dh2],3)

        h2, h2_w, h2_b = deconv2d(
            ch1, [batch_size, s_h4, s_w4, gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(g_bn2(h2))
        
        ch2 = tf.concat([h2,dh1],3)

        h3, h3_w, h3_b = deconv2d(
            ch2, [batch_size, s_h2, s_w2, gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(g_bn3(h3))

        #h4, h4_w, h4_b = deconv2d(
        #   h3, [batch_size, image_size, image_size, 1], name='g_h4', with_w=True)
        
        ch3 = tf.concat([h3,dh0],3) 
        
        
        deconv = tf.nn.conv2d_transpose(ch3, w_extra, output_shape=[batch_size, image_size, image_size, gf_dim],
                strides=[1, 2, 2, 1])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases_extra), deconv.get_shape())
       
       
        deconv2 = tf.nn.conv2d_transpose(deconv, w_final, output_shape=[batch_size, image_size, image_size, num_labels],
                strides=[1, 1, 1, 1])
        deconv2 = tf.reshape(tf.nn.bias_add(deconv2, biases_final), deconv2.get_shape())
  
        if train:
          deconv2 = tf.nn.dropout(deconv2, 0.5)
       
          
      
    return deconv2
      
       
  final_layer_variables = scope_variables("final_layer")
  print (final_layer_variables)
  
 
  
  logits = unet_model(tf_train_dataset,  reuse=False,train=True)
  
  weighted_logits = tf.multiply(logits, tf_train_weights) # shape [batch_size, 2]
  loss = tf.reduce_mean(
     tf.nn.softmax_cross_entropy_with_logits(logits=weighted_logits, labels=tf_train_labels))
 
  #eps = 1e-5
  #prediction = tf.nn.softmax(logits)
  #intersection = tf.reduce_sum(prediction * tf_train_labels)
  #union =  eps + tf.reduce_sum(prediction) + tf.reduce_sum(tf_train_labels)
  #loss = -(2 * intersection/ (union))
    
  global_step = tf.Variable(0)  # count the number of steps taken.
  # learning_rate = tf.train.exponential_decay(
  #     0.5,                # Base learning rate.
  #     global_step * batch_size,  # Current index into the dataset.
  #     train_labels.shape[0],          # Decay step.
  #     0.8,                # Decay rate.
  #     staircase=False)

  #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step,var_list=[final_layer_variables,scope_variables("generator")])
  #optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss, global_step=global_step,var_list=[final_layer_variables,scope_variables("generator")])
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step,var_list=[final_layer_variables,scope_variables("generator")])
  #optimizer = tf.train.MomentumOptimizer( 0.0001, 0.5).minimize(loss, global_step=global_step,var_list=[final_layer_variables,scope_variables("generator")])
     

  train_prediction = tf.nn.softmax(logits)
  test_prediction = tf.nn.softmax(unet_model(tf_test_dataset, True,False))
  #test_prediction =seg_model(tf_test_dataset, True,False)
  
  
  discriminator_variables = scope_variables("discriminator")

# Get all the trainable parameters for the generator 
  #generator_variables = scope_variables("generator")
  #final_layer_variables = scope_variables("final_layer")

  
  
  #num_steps = 20000 # this was 10000 in the training code - TBD consistent value for training and testing 


#model_to_load = '/home/pnb/gan_model2/DCGAN.model-18001'
#model_to_load = '/home/pnb/gan_model2/'
#checkpoint_dir = '/home/psm/MyGit/DCGAN-copy/checkpoint/fluorescent_png_64_128_128/'

#checkpoint_dir = '/home/psm/Augmentation/transfer_gan/gan_model_newz01/' # this was made to be equivalent to args.save_dir + '/checkpoint/'

print_tensors_in_checkpoint_file(file_name=model_to_load, tensor_name='', all_tensors=False)   

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
 
#output_folder = '/home/psm/Augmentation/output/gan/transfer/newz01/test_unet/' # this is equivalent to args.save_dir
output_folder = args.save_dir + "/"
gt_output = args.save_dir+'/val_gt_masks/'
pred_output = args.save_dir + '/val_predicted_masks/'
model_output = args.save_dir + '/trained_model/'
pred_output_test = args.save_dir + '/test_predicted_masks/'

checkpoint_dir = model_output# + 'checkpoint/'


if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if not os.path.exists(model_output):
    os.makedirs(model_output) 

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)   

if args.saveim:    
  if not os.path.exists(gt_output):
    os.makedirs(gt_output)
  if not os.path.exists(pred_output):
    os.makedirs(pred_output)
  if not os.path.exists(pred_output_test):
    os.makedirs(pred_output_test)


num_test_steps = int(test_dataset.shape[0]/batch_size)
num_valid_steps = int(valid_dataset.shape[0]/batch_size)

tr_acc = list()
valid_acc = list()
test_acc = list()
loss_var= list()
valid_dice = list()
test_dice = list()
  

with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
  tf.initialize_all_variables().run()
  # init_top=tf.global_variables_initializer()
  saver=tf.train.Saver(scope_variables("discriminator"))
  # session.run(init_top)

  print('Initialized')
  #discriminator_variables = scope_variables("discriminator")

# Get all the trainable parameters for the generator 
  generator_variables = scope_variables("generator")
  #print (generator_variables)
  final_layer_variables = scope_variables("final_layer")
  #saver = tf.train.Saver(discriminator_variables+generator_variables+final_layer_variables)
  
  saver.restore(session, model_to_load)
  # ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  # if ckpt and ckpt.model_checkpoint_path:
  #     ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
  #     saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
  #     print(" [*] Success to read {}".format(ckpt_name))
  # else:
  #     print(" [*] Failed to find a checkpoint")

  


  for step in range(num_steps):
    offset = (step * batch_size) % (train_dataset.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]                          
    batch_weights = train_weights[offset:(offset+batch_size)]
    #print(batch_test_data.shape)                                
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, tf_train_weights:batch_weights}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1000 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      
      
      valid_pred = list()

      with tf.Graph().as_default():
        for ss in range(num_valid_steps):
          test_offset=(ss * batch_size) % (valid_dataset.shape[0] - batch_size)                       
          batch_valid_data=valid_dataset[test_offset:(test_offset + batch_size), :, :, :]                            
          #batch_test_labels = test_labels[test_offset:(test_offset + batch_size)]
          feed_dict2 = {tf_test_dataset:batch_valid_data}
          valid_pred.append(session.run(test_prediction, feed_dict=feed_dict2))    
#      session.close()    
      valid_pred = np.array(valid_pred)
      valid_pred = valid_pred.reshape((num_valid_steps*batch_size, image_size, image_size, num_labels)).astype(np.float32)
      valid_accuracy = accuracy(valid_pred, valid_labels[0:num_valid_steps*batch_size])
      print('Valid accuracy: %.1f%%' % valid_accuracy)  
      dice_v = dice_index(valid_pred, valid_labels[0:num_valid_steps*batch_size])
      print('Valid dice: %.1f%%' % dice_v)
      valid_acc.append(valid_accuracy)
      valid_dice.append(dice_v)
      
      del valid_pred
      
      nb_test_steps = int(test_labels.shape[0]/batch_size)
      test_pred = list()
      
      with tf.Graph().as_default():
          for s in range(nb_test_steps):
              test_offset=(s * batch_size) % (test_dataset.shape[0] - batch_size)                       
              batch_test_data=test_dataset[test_offset:(test_offset + batch_size), :, :, :]                            
              #batch_test_labels = test_labels[test_offset:(test_offset + batch_size)]
              feed_dict2 = {tf_test_dataset:batch_test_data}
              test_pred.append(session.run(test_prediction, feed_dict=feed_dict2))    
#      session.close()    
      test_pred = np.array(test_pred)
      test_pred = test_pred.reshape((nb_test_steps*batch_size, image_size, image_size, num_labels)).astype(np.float32)
      print(test_pred.shape)
      dice_t = dice_index(test_pred, test_labels[0:nb_test_steps*batch_size])
      nb_test_steps = test_labels.shape[0]/batch_size
      test_acc.append(accuracy(test_pred, test_labels[0:nb_test_steps*batch_size]))
      test_dice.append(dice_t)
      print('Test accuracy: %.1f%%' % accuracy(test_pred, test_labels[0:nb_test_steps*batch_size]))
      print('Test dice: %.1f%%' % dice_t)

      for ii in range(test_pred.shape[0]):
          dense_labels = np.argmax(test_pred[ii, :, :], 2)
          scipy.misc.imsave(pred_output_test + list_of_org_names[ii], dense_labels)


      #del test_pred
            #print(test_pred.shape)
      
      # make it faster 
      
      tr_acc.append(accuracy(predictions, batch_labels))
     
      loss_var.append(l)
      
#with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:            
  tr_acc = np.array(tr_acc)    
  test_acc=np.array(test_acc)
  valid_acc=np.array(valid_acc)
  loss_var=np.array(loss_var)
  valid_dice = np.array(valid_dice)
  test_dice = np.array(test_dice)
  iterations=np.arange(0,num_steps,1000)
  print(iterations.shape)
  #fig_acc,ax_acc= plt.subplots( nrows=1, ncols=1 ) 
  #ax_acc.plot(iterations, tr_acc,'b-')  
  #ax_acc.plot(iterations,test_acc,'r--')
  #ax_acc.plot(iterations,valid_acc,'g--')
  #fig_acc.savefig(output_folder+'acc_var.png')
  
  #fig_acc,ax_acc= plt.subplots( nrows=1, ncols=1 ) 
  #ax_acc.plot(iterations, tr_acc,'b-')  
  #ax_acc.plot(iterations,test_dice,'r--')
  #ax_acc.plot(iterations,valid_dice,'g--')
  #fig_acc.savefig(output_folder+'dice_var.png')
  
  #fig_loss,ax_loss= plt.subplots( nrows=1, ncols=1 ) 
  #ax_loss.plot(iterations[1:],loss_var[1:],'g-')
  #fig_loss.savefig(output_folder+'loss_var.png')
  
  # evolution_tab = np.concatenate(([iterations], [loss_var], [tr_acc], [valid_acc],[test_acc]), axis=0)  
  # np.savetxt(output_folder+"acc_meas.csv", evolution_tab.T,fmt='%.3e', delimiter=",")
  
  # evolution_tab = np.concatenate(([iterations], [loss_var], [valid_dice], [test_dice]), axis=0)  
  # np.savetxt(output_folder+"dice_meas.csv", evolution_tab.T,fmt='%.3e', delimiter=",")
  
  # for ii in range (test_pred.shape[0]):
  #     dense_labels = np.argmax(test_pred[ii,:,:], 2)
  #     scipy.misc.imsave(output_folder+'val_predicted_masks/'+'test_predicted'+str(ii)+'.png', dense_labels)
  #     gt_labels = np.argmax(test_labels[ii,:,:], 2)
  #     scipy.misc.imsave(output_folder+'val_gt_masks/'+'test_gt'+str(ii)+'.png', gt_labels)
      
  # save_path = saver.save(session, output_folder+"trained_model/"+"segmentation_gan_transfer.ckpt")
  # print("Model saved in file: %s" % save_path) 

################
  evolution_tab = np.concatenate(([iterations], [loss_var], [tr_acc], [valid_acc],[test_acc]), axis=0)  
  np.savetxt(args.save_dir+"acc_meas.csv", evolution_tab.T,fmt='%.3e', delimiter=",")
  
  evolution_tab = np.concatenate(([iterations], [loss_var], [valid_dice], [test_dice]), axis=0)  
  np.savetxt(args.save_dir+"dice_meas.csv", evolution_tab.T,fmt='%.3e', delimiter=",")
  
  for ii in range (test_pred.shape[0]):
      dense_labels = np.argmax(test_pred[ii,:,:], 2)
      scipy.misc.imsave(pred_output+list_of_org_names[ii], dense_labels)
      gt_labels = np.argmax(test_labels[ii,:,:], 2)
      scipy.misc.imsave(gt_output + list_of_org_names[ii], gt_labels)

  saver2 = tf.train.Saver()

  save_path = saver2.save(session, model_output +"segmentation_unet_gan.ckpt")
  print("Model saved in file: %s" % save_path) 
      
            
