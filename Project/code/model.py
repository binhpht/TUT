# -*- coding: utf-8 -*-

#Pham Huu Thanh Binh
#Tampere University of Technology 
import numpy as np
from keras import layers
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imshow
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Subtract
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
import glob
from PIL import Image
# for current forlder
import os 
from skimage import color
from skimage import io
import fnmatch

#Pham Huu Thanh Binh

def model_architecture(traning_data,learner_params):
    """
    Design the architecture for model
    
    Arguments:
    traning_data  -- the training set of noising images 
    learner_params -- the paramaters that need to config the model

    Returns:
    the model
    """
    np.random.seed(learner_params['general']['seed'])
    # Read general model parameters.
    batchnorm_flag = learner_params['general']['batchnorm_flag']
    dropout_flag = learner_params['general']['dropout_flag']
    dropout_position = learner_params['general']['dropout_position']

    dropout_rate = learner_params['general']['dropout_rate']
    print(dropout_rate)
    output_act = learner_params['general']['output_act']
    
    # CNN general model parameters
    input_shape = traning_data.shape
    nb_conv_filters = learner_params['conv_params']['nb_conv_filters']
    kernel_size = learner_params['conv_params']['kernel_size']
    conv_act = learner_params['conv_params']['conv_act']
    deep_level = learner_params['conv_params']['deep_level']
    conv_border_mode = learner_params['conv_params']['conv_border_mode']
    conv_stride = learner_params['conv_params']['conv_stride']



    X_input = Input(traning_data[1,:].shape)
    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(nb_conv_filters, kernel_size, strides = conv_stride, padding=conv_border_mode, name = 'conv0')(X_input)
    #X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation(conv_act)(X)
     # 15 layers, Conv+BN+relu
    for i in range(deep_level):
        X = Conv2D(nb_conv_filters, kernel_size , strides=conv_stride, padding=conv_border_mode)(X)
        if (dropout_flag == True and  i % dropout_position == 0):
            X = (Dropout(float(dropout_rate)))(X)
        X = BatchNormalization(axis=-1, epsilon=1e-3)(X)
        X = Activation(conv_act)(X)   
    X = Conv2D(1,kernel_size, strides=conv_stride, padding=conv_border_mode)(X)
    X = Subtract()([X_input, X])   # input - noise
    model = Model(inputs=X_input, outputs=X)
    return model
    
