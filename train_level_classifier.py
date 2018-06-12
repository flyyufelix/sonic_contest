"""
Train Sonic level classifier with ResNet 50.
Implementation reference: https://github.com/flyyufelix/cnn_finetune/blob/master/resnet_50.py
"""

import numpy as np
import os
import glob
import cv2
import copy
import time

import pandas as pd
from timeit import default_timer

import tensorflow as tf

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold, StratifiedKFold
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.optimizers import SGD
from keras.optimizers import Adadelta, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, Lambda, GlobalAveragePooling2D, Merge
from keras.models import Model
from keras import backend as K
from keras import regularizers

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json
from numpy.random import permutation
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

random_seed = 3
np.random.seed(random_seed)

def get_im(path, img_rows, img_cols, color_type=1):
    if color_type == 1:
        # Load as grayscale
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif color_type == 3:
        img = cv2.imread(path)
    # Reduce size
    try:
      resized = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
      resized = cv2.resize(resized, (img_cols, img_rows))
    except Exception as e:
      print('err resizing image: ' + str(e))
      resized = None

    return resized

def load_train(img_rows, img_cols, color_type=1):
    X_train = []
    y_train = []
    X_train_id = []
    start_time = time.time()

    print('Read train images')

    # 27 Classes
    dataset_dir = "./dataset"
    folders = [name for name in os.listdir(dataset_dir)]
    print(folders)

    for fld in folders:
        index = folders.index(fld) # Class ID
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join(dataset_dir, fld, '*.png')
        files = glob.glob(path)

        for fl in files:
            flbase = os.path.basename(fl)
            try:
                img = get_im(fl, img_rows, img_cols, color_type) # Read and Resize Image
                img = np.expand_dims(img, axis=-1)
                if img is not None:
                    X_train.append(img)
                    X_train_id.append(flbase)
                    y_train.append(index)
            except Exception as e:
                print('err reading image: ' + str(e))

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id

def read_and_normalize_and_shuffle_train_data(img_rows, img_cols, color_type, random_seed, subtract_mean=True):

    np.random.seed(random_seed)

    train_data, train_target, train_id = load_train(img_rows, img_cols, color_type)

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    train_target_vec = copy.deepcopy(train_target)
    train_target = np_utils.to_categorical(train_target, 27) # Transform Categorical Vector to One-Hot-Encoded Vector
    train_id = np.array(train_id)

    print('Convert to float...')
    train_data = train_data.astype('float32')
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    train_target_vec = train_target_vec[perm]
    train_id = train_id[perm]
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_target_vec, train_id

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

def resnet50_model(img_rows, img_cols, color_type=1, num_class=None):
    '''Instantiate the ResNet50 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
            to use as image input for the model.
    # Returns
        A Keras model instance.
    '''
    bn_axis = -1
    img_input = Input(shape=(img_rows, img_cols, color_type))
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    # load weights
    #model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_class, activation='softmax', name='fc10')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  

    return model

def save_model(model, desc='', save_weights=True):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + desc + '.json'
    open(json_name, 'w').write(json_string)

    # Might use ModelCheckpoint callbacks to store weights instead
    if save_weights:
      weight_name = 'model_weights_' + desc + '.h5'
      model.save_weights(weight_name, overwrite=True)

def train_covnet(nb_epoch=3, size=(224,224)):

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101

    img_rows, img_cols = size
    batch_size = 16
    random_state = random_seed
    num_class = 27
    color_type = 1

    train_data, train_target, train_target_vec, train_id = read_and_normalize_and_shuffle_train_data(img_rows, img_cols, color_type,random_seed)

    X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=0.1)

    model = resnet50_model(img_rows, img_cols, color_type, num_class)

    model.fit(
        X_train,
        Y_train,
        epochs=nb_epoch,
        batch_size=batch_size,
        validation_data=(X_valid, Y_valid)
    )

    save_model(model, 'level_classifier', save_weights=True)


if __name__ == '__main__':
  train_covnet()
