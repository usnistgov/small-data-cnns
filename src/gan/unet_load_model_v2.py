# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:45:12 2016

@author: psm
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

parser = argparse.ArgumentParser(description='U-Net inference segmentation')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='images', help='path of the dataset')
parser.add_argument('--output_dir', dest='output_dir', default='masks', help='path of the masks')
parser.add_argument('--model_dir', dest='model_dir', default='unet_save', help='path to load the trained model')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--split_size', dest='split_size', type=int, default=128, help='split images to tiles of this size')
parser.add_argument('--gf_dim', dest='gf_dim', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--df_dim', dest='df_dim', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='num_channels', type=int, default=1, help='# of input image channels')
#parser.add_argument('--output_nc', dest='output_nc', type=int, default=5, help='# of output image channels')
parser.add_argument('--invert', dest='invert', type=int, default=1, help='Postprocessing: open, skeletonize and invert image')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#test_batch_size = test_dataset.shape[0]/num_test_steps 
#batch_size = 16
#patch_size = 7
#patch_size2 = 5
#patch_size3 = 3
#depth = 64
#depth2=64
#depth3=64

num_labels = 2 #args.output_nc
num_channels = args.num_channels # grayscale
batch_size=args.batch_size
image_size = int(args.split_size)
postprocess = int(args.invert)#int(sys.argv[4])    

df_dim = args.df_dim #64
gf_dim =  args.gf_dim #64  

#output_folder = '/home/psm/MyGit/DeepLearning/TensorFlow/SegNetForWIPP/output/'
output_folder = args.output_dir#args.output_dir#str(sys.argv[2])

#checkpoint_dir = '/home/psm/MyGit/DeepLearning/TensorFlow/SegNetForWIPP/trained_models/'
checkpoint_dir = args.model_dir #+ "/checkpoint"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    


#############################################################################################################
#############################################################################################################




#############################################################################################################
#############################################################################################################

    
 
    
    

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
  #df_dim=64
  #gf_dim=64
  
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
             
  
  test_prediction = tf.nn.softmax(unet_model(tf_test_dataset, False,False))

  
  #saver = tf.train.Saver()
  
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



#test_org_folders = '/home/psm/MyGit/DeepLearning/TensorFlow/SegNetForWIPP/z01_test/'
test_org_folders = args.dataset_dir#str(sys.argv[1])
print('Loading images...')
list_of_org_test, list_of_filenames = load_images_with_names(test_org_folders)
test_dataset = np.array(list_of_org_test)
test_dataset = reformat_weights(test_dataset)



num_test_steps = test_dataset.shape[0]
with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as session:
  #tf.global_variables_initializer().run()
  tf.initialize_all_variables().run()

  saver = tf.train.Saver()

  print('Initialized')
  print("checkpoint_dir = ", checkpoint_dir)

  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  print("ckpt file = ", ckpt)

  if ckpt and ckpt.model_checkpoint_path:

      print("model_checkpoint_path", ckpt.model_checkpoint_path)

      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      print("ckpt_name=", ckpt_name)
      saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}", ckpt_name)
  else:
      print(" [*] Failed to find a checkpoint")
  nb_test_steps = test_dataset.shape[0]/batch_size
  test_pred = list()
      
  with tf.Graph().as_default():
      for s in range(nb_test_steps):
          #test_offset=(s * batch_size) % (test_dataset.shape[0] - batch_size)                       
          #batch_test_data=test_dataset[test_offset:(test_offset + batch_size), :, :, :]
          image_test_data=test_dataset[s, :, :, :]                             
          #batch_test_labels = test_labels[test_offset:(test_offset + batch_size)]
          split_test, xshape, yshape = split_image(image_test_data, image_size)
          split_test_pred = list()
          for sp in range(len(split_test)):
              reformated_split_test  =split_test[sp].reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
              feed_dict2 = {tf_test_dataset:reformated_split_test}
              predicted_image = session.run(test_prediction, feed_dict=feed_dict2)
              split_test_pred.append(np.argmax(predicted_image[0,:,:], 2))
          unsplit_pred = unsplit_image(split_test_pred, xshape, yshape) 
          if postprocess:
              dense_labels = skel_invert(unsplit_pred)
          else:     
              dense_labels =  unsplit_pred
          print(list_of_filenames[s])
          img = scipy.misc.toimage(dense_labels, high=num_labels-1, low=0, mode='I')
          img.save(output_folder+'/'+list_of_filenames[s], dense_labels.all())
#          scipy.misc.imsave(output_folder+'/'+list_of_filenames[s]+'.tif', dense_labels)
          #test_pred.append(unsplit_pred)    
#      session.close()    
#  test_pred = np.array(test_pred)
#  test_pred = test_pred.reshape((nb_test_steps*batch_size, image_size, image_size, num_labels)).astype(np.float32)
#  print(test_pred.shape)
  
      
  
      
            
