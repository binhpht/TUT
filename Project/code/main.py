#Pham Huu Thanh Binh
#Tampere University of Technology 
#Import library

import yaml
import os
import numpy as np
import glob
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

from keras.optimizers import *
from keras.losses import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from keras.models import load_model
from keras.utils import to_categorical
from model import model_architecture
from process_data import process_data

from IPython import embed
from sklearn.metrics import confusion_matrix
import itertools# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from common_utils import plot_loss

if __name__ == "__main__":
    np.random.seed(1234)
    # Read YAML file
    with open('/Users/binhpht/Dropbox/TUT/Master/Thesis/Project/code/config.yaml','r') as f:
        learner_params = yaml.load(f)
        
    data_path = '/Users/binhpht/Dropbox/TUT/Master/Thesis/Project/code/data/raw_data/*.png'
    w = learner_params['image_parames']['w']
    h = learner_params['image_parames']['h']
    seed_data = learner_params['general']['seed']
    return_data = process_data(data_path,w,h,seed_data)
    
    image_list_grey = return_data[0]
    image_list_grey_noise = return_data[1]
    image_list_residual_error = return_data[2]
    
    X_train, X_test, y_train, y_test = train_test_split(image_list_grey_noise, image_list_grey, test_size=0.01, random_state=123)
   
    # Reshape as the format of Keras
    X_train = X_train.reshape(X_train.shape[0], h, w, 1)
    y_train = y_train.reshape(y_train.shape[0], h, w, 1)
    X_test = X_test.reshape(X_test.shape[0], h, w, 1)
    
    # standardize our dataset
    X_train = X_train/255
    y_train = y_train/255
    
    model = model_architecture(X_train, y_train, learner_params)
    loss_param = learner_params['general']['loss']
    optimize_param = learner_params['general']['optimizer']
    nb_epoch = learner_params['general']['nb_epoch']
    batch_size = learner_params['general']['batch_size']
    model.compile(loss = loss_param ,optimizer= optimize_param)
    model.fit(x = X_train ,y = y_train, epochs = nb_epoch, batch_size = batch_size)
    result = mode.predict(X_test)

    result_path = '/Users/binhpht/Dropbox/TUT/Master/Thesis/Project/code/result/' +str(learner_params['general']['id'])
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(os.path.join(result_path, 'loss_plot.png'))
    plt.close()
    
    ## add later for saving the model 
#    checkpoint = ModelCheckpoint(os.path.join(result_path, 'best_model.h5'), monitor='val_loss', save_best_only=True, mode='min')
#    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, mode='min')
#    callbacks_list = [checkpoint, early_stopping]