#/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 16:48:52 2017

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


#from ops import *



batch_size = 16
patch_size = 7
patch_size2 = 5
patch_size3 = 3
depth = 64
depth2=64
depth3=64

num_labels = 2
num_channels = 1 # grayscale
image_size=256
background_value=0.0
#output_folder = '/home/psm/Code/CellSegmentation/forMike/output/experiment1/'


def check_size(source_im, target_im):
    s_width = source_im.shape[0]
    s_height = source_im.shape[1]

    t_width = target_im.shape[0]
    t_height = target_im.shape[1]
    
    min_width = np.min([s_width, t_width])
    min_height = np.min([s_height, t_height])
    
    source_im = source_im[:min_width,:min_height]
    target_im = target_im[:min_width,:min_height]
  
    return source_im, target_im
    
    
def my_crop(image, width, height):
    offset_x = (image.shape[0]-width)/2
    offset_y = (image.shape[1]-height)/2
    cropped_image = image[offset_x:offset_x+width, offset_y:offset_y+height]
    return cropped_image

def no_augment_train(dataset, labels, weights):
    aug_dataset = list()
    aug_labels = list()
    aug_weights = list()

    for ii in range(dataset.shape[0]):
     aug_dataset.append(dataset[ii,:,:])
     aug_labels.append(labels[ii])
     aug_weights.append(weights[ii])

    return randomize_with_weights(np.array(aug_dataset), np.array(aug_labels), np.array(aug_weights))


def load_images_crop(folder, crop_size):
  list_of_images = list()
  image_files = os.listdir(folder)
  image_files.sort()
  
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
        image_data = ndimage.imread(image_file).astype(np.float32)
        image_data = (image_data-np.mean(image_data))/np.std(image_data)
	#image_data = image_data/127.5 -1.0
	#image_data = image_data/np.amax(image_data)        
	orig_rows, orig_cols = image_data.shape
        if orig_rows<crop_size:
            for addition in range(0, crop_size-orig_rows):
                lst = np.array(list(int(background_value) for x in range(0, orig_cols)))
                image_data = np.vstack((image_data,lst))
                #print ('Adding rows', image_data.shape)
        orig_rows, orig_cols = image_data.shape
        if orig_cols<crop_size:
            for addition in range(0, crop_size-orig_cols):
                lst = np.array(list(int(background_value) for x in range(0, orig_rows)))
                #image_data = np.hstack((image_data,lst)) 
                image_data =np.column_stack((image_data, lst))
		#print ('Adding cols', image_data.shape)
       
        #print (image_data.shape) 
	image_data = my_crop(image_data, crop_size, crop_size)
        
        
 
        #image_data = (image_data-np.amin(image_data))/np.amax(image_data)
        #resized_image = resize(image_data, (image_size, image_size), order=1, preserve_range=True)
        
        list_of_images.append(image_data)  
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
      
    
  return list_of_images

def load_masks_crop(folder, crop_size):
  list_of_images = list()
  image_files = os.listdir(folder)
  image_files.sort()
  
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
        image_data = ndimage.imread(image_file).astype(np.int32)
        image_data = (image_data>0).astype(int)
        orig_rows, orig_cols = image_data.shape
        if orig_rows<crop_size:
            for addition in range(0, crop_size-orig_rows):
                lst = np.array(list(int(background_value) for x in range(0, orig_cols)))
                image_data = np.vstack((image_data,lst))
        orig_rows, orig_cols = image_data.shape
        if orig_cols<crop_size:
            for addition in range(0, crop_size-orig_cols):
                lst = np.array(list(int(background_value) for x in range(0, orig_rows)))
                #image_data = np.hstack((image_data,lst)) 
                image_data =np.column_stack((image_data, lst))

	image_data = my_crop(image_data, crop_size, crop_size)
        #resized_image = resize(image_data, (image_size, image_size), order=1, preserve_range=True)
        list_of_images.append(image_data)  
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
      
    
  return list_of_images


def load_int_images(folder):
  """Load the data for a single letter label."""
  list_of_images = list()
  image_files = os.listdir(folder)
  image_files.sort()
  
  #ii = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
        image_data = ndimage.imread(image_file).astype(np.int32)
        #image_data = image_data/np.amax(image_data)
        image_data = (image_data>0).astype(int)
        #image_data = resize(image_data,(image_size,image_size),order=1,preserve_range=True )
        image_data = image_data[0:image_size, 0:image_size]
        #scipy.misc.imsave('test_gt'+str(ii)+'.png', image_data)
        #ii=ii+1 
        list_of_images.append(image_data)  
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
   
  return list_of_images   

def load_images_weights_distance(folder):
  """Load the data for a single letter label."""
  list_of_images = list()
  image_files = os.listdir(folder)
  image_files.sort()
  
  #ii = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
        image_data = ndimage.imread(image_file).astype(np.float32)
        bw_data = (image_data>0).astype(np.float32)
        image_weight =  np.exp(-0.5*(distance_transform_edt(1-bw_data)))
        image_weight = image_weight[0:image_size, 0:image_size]
        #ii=ii+1 
        list_of_images.append(image_weight)  
     
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
   
  return list_of_images     

def load_images_weights_distance_crop(folder, crop_size):
  """Load the data for a single letter label."""
  list_of_images = list()
  image_files = os.listdir(folder)
  image_files.sort()
  
  #ii = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
        image_data = ndimage.imread(image_file).astype(np.float32)
        image_data = my_crop(image_data, crop_size, crop_size)
        #bw_data = (image_data>0).astype(np.float32)
        #image_wei =  1.0/(1+(distance_transform_edt(1-bw_data)**2))
        #image_weight =  1.0/(1+(distance_transform_edt(1-bw_data)))
        #image_weight = image_weight[0:image_size, 0:image_size]
        #ii=ii+1
        orig_rows, orig_cols = image_data.shape
        if orig_rows<crop_size:
            for addition in range(0, crop_size-orig_rows):
                lst = np.array(list(int(background_value) for x in range(0, orig_cols)))
                image_data = np.vstack((image_data,lst))
        orig_rows, orig_cols = image_data.shape
        if orig_cols<crop_size:
            for addition in range(0, crop_size-orig_cols):
                lst = np.array(list(int(background_value) for x in range(0, orig_rows)))
                #image_data = np.hstack((image_data,lst)) 
                image_data =np.column_stack((image_data, lst))

        image_data = my_crop(image_data, crop_size, crop_size)
        bw_data = (image_data>0).astype(np.float32)
        image_weight =  1.0/(1+2*(distance_transform_edt(1-bw_data)))
	#image_weight[bw_data<1]=image_weight[bw_data<1]*(-1)
        list_of_images.append(image_weight)  
     
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
   
  return list_of_images    
        
def load_images(folder):
  """Load the data for a single letter label."""
  list_of_images = list()
  image_files = os.listdir(folder)
  image_files.sort()
  
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
        image_data = ndimage.imread(image_file).astype(float)
        image_data = (image_data-np.mean(image_data))/np.std(image_data)
        #image_data = resize(image_data,(image_size,image_size),order=1,preserve_range=True )
        image_data = image_data[0:image_size, 0:image_size]
        #print(image_data.shape)
        list_of_images.append(image_data)  
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
      
    
  return list_of_images   


def load_images_with_names(folder):
  """Load the data for a single letter label."""
  list_of_images = list()
  list_of_filenames = list()
  image_files = os.listdir(folder)
  image_files.sort()
  
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
        image_data = ndimage.imread(image_file).astype(float)
        image_data = (image_data-np.mean(image_data))/np.std(image_data)
        #image_data = resize(image_data,(image_size,image_size),order=1,preserve_range=True )
        image_data = image_data[0:image_size, 0:image_size]
        #print(image_data.shape)
        list_of_images.append(image_data)
        list_of_filenames.append(image)
    except IOError as e:
        print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
      
    
  return list_of_images, list_of_filenames  


def randomize_with_weights(dataset, labels,weights):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  shuffled_weights = weights[permutation]
  return shuffled_dataset, shuffled_labels, shuffled_weights  


  
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
#train_dataset, train_labels = randomize(train_dataset, train_labels)



def reformat(dataset, labels):

  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  
  
  r_labels = np.zeros(shape=(labels.shape[0], labels.shape[1], labels.shape[2], num_labels), dtype=np.float32)
  
  for ii in range(labels.shape[0]):
      lim = labels[ii]
      rweight = np.zeros(shape=(lim.shape[0],lim.shape[1],2), dtype=np.float32)
      rweight[lim==1,1] = 1.0#wim[lim==1]
      rweight[lim==1,0] = 0.0#1.0-wim[lim==1]
      rweight[lim==0,0] = 1.0#wim[lim==0]
      rweight[lim==0,1] = 0.0#1.0-wim[lim==0]
      r_labels[ii]=rweight
  
  
  
  return dataset, r_labels
  
def reformat2(dataset, labels):
  
  num_labels = 2
  num_channels = 1  
  dataset = dataset.reshape(
    (-1, dataset.shape[1], dataset.shape[2], num_channels)).astype(np.float32)
  
  
  r_labels = np.zeros(shape=(labels.shape[0], labels.shape[1], labels.shape[2], num_labels), dtype=np.float32)
  
  for ii in range(labels.shape[0]):
      lim = labels[ii]
      rweight = np.zeros(shape=(lim.shape[0],lim.shape[1],2), dtype=np.float32)
      rweight[lim==1,1] = 1.0#wim[lim==1]
      rweight[lim==1,0] = 0.0#1.0-wim[lim==1]
      rweight[lim==0,0] = 1.0#wim[lim==0]
      rweight[lim==0,1] = 0.0#1.0-wim[lim==0]
      r_labels[ii]=rweight
  
  
  
  return dataset, r_labels 
  
def reformat_weights(weights):
  weights = weights.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)   
  
  return weights

def reformat_weights2(weights, labels):
  
  r_weights = np.zeros(shape=(weights.shape[0], weights.shape[1], weights.shape[2], 2), dtype=np.float32)
  
  for ii in range(labels.shape[0]):
      lim = labels[ii]
      wim = weights[ii]
      rweight = np.zeros(shape=(wim.shape[0],wim.shape[1],2), dtype=np.float32)
      rweight[lim==1,1] = 0.8#wim[lim==1]
      rweight[lim==1,0] = 0.2#1.0-wim[lim==1]
      rweight[lim==0,0] = 0.2#wim[lim==0]
      rweight[lim==0,1] = 0.8#1.0-wim[lim==0]
      r_weights[ii]=rweight
  

  return r_weights  
  
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)))
          / (predictions.shape[0]*predictions.shape[1]*predictions.shape[2])) 
  
  
  
def dice_index (predictions, labels):
  eps = 1e-5
  #prediction = tf.nn.softmax(logits)
  #intersection = tf.reduce_sum(predictions * labels)
  bw_pred = np.argmax(predictions, 3) 
  bw_labels = np.argmax(labels, 3)
  intersection = np.sum(bw_pred * bw_labels)
  #union =  eps + tf.reduce_sum(predictions) + tf.reduce_sum(labels)
  union = eps + np.sum(bw_pred) + np.sum(bw_labels)
  return (2 * intersection/ (union))



def separate_train_test(dataset, labels, weights, ratio):
    #dataset, labels, weights = randomize_with_weights(dataset, labels, weights)
    num_images = labels.shape[0]
    nb_test_size = int(math.floor(num_images*ratio))
    test_dataset = dataset[0:nb_test_size,:,:]
    test_labels = labels[0:nb_test_size]
    
    train_dataset = dataset[nb_test_size:num_images,:,:]
    train_labels = labels[nb_test_size:num_images]
    train_weights = weights[nb_test_size:num_images]

    return train_dataset, train_labels, train_weights, test_dataset, test_labels

def split_image(input_image, split_size):
    xshape = float(input_image.shape[0])
    yshape = float(input_image.shape[1]) 
    x_nb_split = int(np.ceil(xshape/split_size))
    y_nb_split = int(np.ceil(yshape/split_size))
    xshape = int(xshape)
    yshape = int(yshape)
    list_of_images = list()
    #print (x_nb_split,y_nb_split)
    for ii in range(x_nb_split-1):
        for jj in range(y_nb_split-1):
            small_image = input_image[ii*split_size:(ii+1)*split_size,jj*split_size:(jj+1)*split_size]
            list_of_images.append(small_image)    
            #print(ii*split_size,jj*split_size)                      
        #last_small_y_image = input_image[ii*split_size:(ii+1)*split_size, yshape-split_size:yshape]
        last_small_y_image = input_image[ii*split_size:(ii+1)*split_size, yshape-split_size:yshape]                                 
        list_of_images.append(last_small_y_image)                               

    for jj in range(y_nb_split-1):
            #small_image = input_image[xshape-split_size-1:xshape-1,jj*split_size:(jj+1)*split_size]
            small_image = input_image[xshape-split_size:xshape,jj*split_size:(jj+1)*split_size]
            list_of_images.append(small_image)                          
    #last_small_y_image = input_image[xshape-split_size-1:xshape-1, yshape-split_size-1:yshape-1]
    last_small_y_image = input_image[xshape-split_size:xshape, yshape-split_size:yshape]
 
    list_of_images.append(last_small_y_image)   

    return list_of_images, xshape, yshape
    
def unsplit_image(list_of_splits, xshape, yshape):
    split_size = list_of_splits[0].shape[0]
    x_nb_split = int(np.ceil(float(xshape)/split_size))
    y_nb_split = int(np.ceil(float(yshape)/split_size))
    #input_image= np.zeros(shape=(xshape,yshape,list_of_splits[1].shape[2]))
    input_image= np.zeros(shape=(xshape,yshape))
                            
    ll=0
    #print (x_nb_split,y_nb_split)
    for ii in range(x_nb_split-1):
        for jj in range(y_nb_split-1):
            input_image[ii*split_size:(ii+1)*split_size,jj*split_size:(jj+1)*split_size] = list_of_splits[ll]
            ll=ll+1            
            #print(ii*split_size,jj*split_size)   
        input_image[ii*split_size:(ii+1)*split_size, int(yshape)-split_size:int(yshape)]=list_of_splits[ll]
        ll=ll+1
        
    for jj in range(y_nb_split-1):
            input_image[int(xshape)-split_size:int(xshape),jj*split_size:(jj+1)*split_size] = list_of_splits[ll]
            ll=ll+1            
    input_image[int(xshape)-split_size:int(xshape), int(yshape)-split_size:int(yshape)]= list_of_splits[ll]

    return input_image
    