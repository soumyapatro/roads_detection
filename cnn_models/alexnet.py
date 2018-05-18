import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_normal
import scipy.misc

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def alexnet(input_shape = (256,256,3), labels = 2):
    
    """ AlextNet architecture (without bactchnorm) = [ CONV2D->RELU-> MAXPOOL] -> [CONV2D (same)->RELU-> 
    MAXPOOL] -> [CONV2D (same)->RELU] -> [CONV2D (same)->RELU] -> [CONV2D (same)->RELU -> MAXPOOL] -> 
    FC1 -> FC2 ->FC3 -> Logistic Unit """
    
    """ Even with input size 256*256*3, the dimensions start aligining in the 4th stage"""
    #Load input
    I_input = Input(input_shape)
    
    #Stage 1
    I = Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), padding='valid', use_bias=True,
               kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros', name = 'conv1' )(I_input)
    I = Activation('relu')(I)
    I = MaxPooling2D((3, 3), strides=(2, 2))(I)
    
    #Stage 2
    I = Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same', use_bias=True,
               kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros', name = 'conv2' )(I)
    I = Activation('relu')(I)
    I = MaxPooling2D((3, 3), strides=(2, 2))(I)
    
    #Stage 3
    I = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=True,
               kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros', name = 'conv3' )(I)
    I = Activation('relu')(I)
    
    #Stage 4
    I = Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=True,
               kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros', name = 'conv4' )(I)
    I = Activation('relu')(I)
    
    #Stage 5
    I = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=True,
               kernel_initializer=glorot_normal(seed=1), bias_initializer='zeros', name = 'conv5' )(I)
    I = Activation('relu')(I)
    I = MaxPooling2D((3, 3), strides=(2, 2))(I)
    
    
    #Flatten & Fully Connected layers
    I = Flatten()(I)
    I = Dense(units=4096, name='fc1', activation= 'relu', kernel_initializer = glorot_normal(seed=1))(I)
    I = Dense(units=4096, name='fc2', activation= 'relu', kernel_initializer = glorot_normal(seed=1))(I)
    I = Dense(labels, name='fc3' + str(labels), kernel_initializer = glorot_normal(seed=1))(I)
    I = Activation('sigmoid')(I)
    
    model = Model(inputs = I_input, outputs = I, name='AlexNet')
    
    return model