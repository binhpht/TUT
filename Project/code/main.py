# Pham Huu Thanh Binh
# Tampere University of Technology
# Import library

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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.optimizers import *
from keras.losses import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler

from IPython import embed
import itertools  # -*- coding: utf-8 -*-

from model import model_architecture
from process_data import process_data, train_datagen


# define loss
def sum_squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true)) / 2

if __name__ == "__main__":
    runing_path  = '/home/pham/projects/denosing/'
    np.random.seed(1234)
    # Read YAML file
    with open(runing_path + 'code/config.yaml', 'r') as f:
        learner_params = yaml.load(f)

    # Configurating the Image Path
    # data_path = runing_path + 'data/DIV2K_train_LR_bicubic/X4/*.png'
    data_path = runing_path + 'data/test/*.png'

    # Set the size of each block of images
    w = learner_params['image_parames']['w']
    h = learner_params['image_parames']['h']

    seed_data = learner_params['general']['seed']

    # Return the block of training images from data.
    return_data = process_data(data_path, h, w)

    # Seperating train, validation data. 
    train_data, val_data = train_test_split(return_data, test_size=0.01, random_state=123)
    print('Traing Data Shape : ' +str(train_data.shape))
    print('Validation Data Shape : ' +str(val_data.shape))

    # Set the path to save the results.
    result_path = runing_path + 'result' + str(learner_params['general']['id'])
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model = model_architecture(learner_params, 1)

    # configurating loss funtion.
    loss_param = learner_params['general']['loss']
    optimize_param = learner_params['general']['optimizer']
    nb_epoch = learner_params['general']['nb_epoch']
    batch_size = learner_params['general']['batch_size']

    # Show the model architecture. 
    model.summary()
    plot_model(model,to_file = os.path.join(result_path, 'model_arc.png'), show_shapes = True)

    model.compile(loss=sum_squared_error, optimizer=optimize_param)
    
    print('Training Loading model')

    checkpoint = ModelCheckpoint(os.path.join(result_path, 'best_model_{epoch:02d}.h5'), save_weights_only=False,
                                 period=1)
    csv_logger = CSVLogger(os.path.join(result_path, 'log.csv'), append=True, separator=',')
    callbacks_list = [checkpoint, csv_logger]

    history = model.fit_generator(train_datagen(train_data, epoch_num=5000), validation_data=train_datagen(val_data, process = 'Validaition'),
                                  steps_per_epoch=5000, validation_steps = 50, epochs=100, verbose=1,
                                  callbacks=callbacks_list)
    print('Completing Training model')

    # summarize history for loss and accuracy

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(os.path.join(result_path, 'accuracy_plot.png'))

    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    plt.savefig(os.path.join(result_path, 'loss_plot.png'))
    plt.close()
