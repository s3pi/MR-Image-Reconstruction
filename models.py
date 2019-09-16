import random
from keras.layers.core import *
import sys
import cv2
import nibabel as nib
import _pickle as cPickle
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
import numpy as np
import scipy.misc
import numpy.random as rng
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Input, concatenate, Add
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D,Convolution2D
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import model_from_json,model_from_config,load_model
from keras.optimizers import SGD,RMSprop,adam,Adam
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.preprocessing import image
from keras import backend as K
from keras.initializers import random_uniform, RandomNormal
from sklearn.utils import shuffle
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from collections import OrderedDict as od
import tensorflow as tf
import os
import math
from contextlib import redirect_stdout
import csv
import pylab as plt

def net2(input_shape):
	input_tensor = Input(input_shape)
	conv1 = Conv2D(128, (1, 1), activation='relu', padding='same')(input_tensor)
	conv1 = BatchNormalization()(conv1)
	conv2 = Conv2D(64, (1,1), activation='relu', padding='same')(conv1)
	conv2 = BatchNormalization()(conv2)
	conv3 = Conv2D(32, (1,1), activation='relu', padding='same')(conv2)
	conv3 = BatchNormalization()(conv3)
	conv4 = Conv2D(3, (1,1), activation='sigmoid',padding='same')(conv3)
	model = Model(input_tensor, conv4)
	return model

def hourglass(input_shape):
	input_tensor = Input(input_shape)
	conv0 = Conv2D(256, (1,1), activation='relu', padding='same')(input_tensor)
	conv0 = BatchNormalization()(conv0)

	conv1 = Conv2D(64, (1, 1), activation='relu', padding='same')(conv0)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	residual1 = Add()([conv0,conv1])

	pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual1) #56

	branch1 = Conv2D(64, (1, 1), activation='relu', padding='same')(residual1)
	branch1 = BatchNormalization()(branch1)
	branch1 = Conv2D(128, (3, 3), activation='relu', padding='same')(branch1)
	branch1 = BatchNormalization()(branch1)
	branch1 = Conv2D(256, (1, 1), activation='relu', padding='same')(branch1)
	branch1 = BatchNormalization()(branch1)
	bresidual1 = Add()([residual1,branch1])

	conv2 = Conv2D(64, (1, 1), activation='relu', padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	residual2 = Add()([pool1,conv2])

	pool2 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual2) #28

	branch2 = Conv2D(64, (1, 1), activation='relu', padding='same')(residual2)
	branch2 = BatchNormalization()(branch2)
	branch2 = Conv2D(128, (3, 3), activation='relu', padding='same')(branch2)
	branch2 = BatchNormalization()(branch2)
	branch2 = Conv2D(256, (1, 1), activation='relu', padding='same')(branch2)
	branch2 = BatchNormalization()(branch2)
	bresidual2 = Add()([residual2,branch2])

	#conv3 = Conv2D(128, (1, 1), activation='relu', padding='same')(pool2)
	#conv3 = BatchNormalization()(conv3)
	#conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	#conv3 = BatchNormalization()(conv3)
	#conv3 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv3)
	#conv3 = BatchNormalization()(conv3)
	#residual3 = Add()([pool2,conv3])

	#pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual3) #14

	#branch3 = Conv2D(128, (1, 1), activation='relu', padding='same')(residual3)
	#branch3 = BatchNormalization()(branch3)
	#branch3 = Conv2D(128, (3, 3), activation='relu', padding='same')(branch3)
	#branch3 = BatchNormalization()(branch3)
	#branch3 = Conv2D(256, (1, 1), activation='relu', padding='same')(branch3)
	#branch3 = BatchNormalization()(x`branch3)
	#bresidual3 = Add()([residual3,branch3])

	###########################BOTLLENECK######################################

	conv4 = Conv2D(64, (1, 1), activation='relu', padding='same')(pool2)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	residual4 = Add()([pool2,conv4])
	#pool4 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual4)

	#conv5 = Conv2D(128, (1, 1), activation='relu', padding='same')(residual4)
	#conv5 = BatchNormalization()(conv5)
	#conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
	#conv5 = BatchNormalization()(conv5)
	#conv5 = Conv2D(256, (1, 1), activation='relu', padding='same')(conv5)
	#conv5 = BatchNormalization()(conv5)
	#residual5 = Add()([residual4,conv5])
	#pool5 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(residual4)

	#############################################################################

	#up1 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same')(residual4)
	#up1 = BatchNormalization()(up1) #28
	#add1 = Add()([up1,bresidual3])

	#uconv1 = Conv2D(128, (1, 1), activation='relu', padding='same')(add1)
	#uconv1 = BatchNormalization()(uconv1)
	#uconv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(uconv1)
	#uconv1 = BatchNormalization()(uconv1)
	#uconv1 = Conv2D(256, (1, 1), activation='relu', padding='same')(uconv1)
	#uconv1 = BatchNormalization()(uconv1)
	#uresidual1 = Add()([add1,uconv1])

	up2 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same')(residual4)#######(conv5)
	up2 = BatchNormalization()(up2) #56
	add2 = Add()([up2,bresidual2])

	uconv2 = Conv2D(64, (1, 1), activation='relu', padding='same')(add2)
	uconv2 = BatchNormalization()(uconv2)
	uconv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(uconv2)
	uconv2 = BatchNormalization()(uconv2)
	uconv2 = Conv2D(256, (1, 1), activation='relu', padding='same')(uconv2)
	uconv2 = BatchNormalization()(uconv2)
	uresidual2 = Add()([add2,uconv2])

	up3 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same')(uresidual2)#######(conv5)
	up3 = BatchNormalization()(up3) #112
	add3 = Add()([up3,bresidual1])

	uconv3 = Conv2D(64, (1, 1), activation='relu', padding='same')(add3)
	uconv3 = BatchNormalization()(uconv3)
	uconv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(uconv3)
	uconv3 = BatchNormalization()(uconv3)
	uconv3 = Conv2D(256, (1, 1), activation='relu', padding='same')(uconv3)
	uconv3 = BatchNormalization()(uconv3)
	uresidual3 = Add()([add3,uconv3])
	model = Model(input_tensor, uresidual3)

	return model

def encoder(input_shape):
	input_tensor = Input(input_shape)#tensor called input_shape is done
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
	conv1 = BatchNormalization()(conv1)

	conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	conv2 = BatchNormalization()(conv2)
	#pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(conv1)
	  
	conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)#(pool1)
	conv3 = BatchNormalization()(conv3)

	residual1 =Add()([Conv2D(64, (1, 1), activation='relu', padding='same')(conv1),conv3])
	
	conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(residual1)
	conv4 = BatchNormalization()(conv4)

	pool2 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same')(conv4)
		
	conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv5 = BatchNormalization()(conv5)

	conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
	conv6 = BatchNormalization()(conv6)
	#pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	
	residual2 =Add()([Conv2D(128, (1, 1), activation='relu', padding='same', name = 'resd2')(pool2),conv6])

	pool = MaxPooling2D(pool_size=(2, 2),padding='same')(residual2) #112
	
	conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool)#(pool3)
	conv7 = BatchNormalization()(conv7)

	conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv7)
	conv8 = BatchNormalization()(conv8)
	#pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	
	#residual3 = Add(name='BEFORE_HG')([pool,conv8])
	model = Model(input_tensor, conv8)
	return model

	
def decoder(input_shape):
	input_tensor = Input(input_shape)	
	up1 = Conv2DTranspose(512, (2,2), strides = (2,2), activation = 'relu', padding = 'same', name='AFTER_HG')(input_tensor)
	up1 = BatchNormalization()(up1)

	uconv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
	uconv1 = BatchNormalization()(uconv1)

	uconv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(uconv1)
	uconv2 = BatchNormalization()(uconv2)

	residual4 =Add()([Conv2D(128, (1, 1), activation='relu', padding='same')(up1),uconv2])

	up2 = Conv2DTranspose(128,(2,2),strides = (2,2), activation = 'relu', padding = 'same')(residual4)
	up2 = BatchNormalization()(up2)

	uconv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
	uconv3 = BatchNormalization()(uconv3)

	uconv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(uconv3)
	uconv4 = BatchNormalization()(uconv4)

	residual5 =Add()([Conv2D(64, (1, 1), activation='relu', padding='same')(up2),uconv4])

	uconv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(residual5)
	uconv5 = BatchNormalization()(uconv5)

	uconv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(uconv5)
	uconv6 = BatchNormalization()(uconv6)

	generator = Conv2D(3, (3, 3),activation='sigmoid', padding='same')(uconv6)
	
	#generator = Model(input_img, output = ref)
	model = Model(input_tensor, generator)

	return model

def autoencoder(input_img):
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(32, (3,3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(512, (3, 3), activation='sigmoid', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)
	up6 = concatenate([conv5, conv4],axis=3)
	# up6 = merge([conv5, conv4], mode='concat', concat_axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
	conv6 = BatchNormalization()(conv6)
	up7 = UpSampling2D((2,2))(conv6)
	up7 = concatenate([up7, conv3],axis=3)
	# up7 = merge([up7, conv3], mode='concat', concat_axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()()
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
	conv7 = BatchNormalization()(conv7)
	up8 = UpSampling2D((2,2))(conv7)
	up8 = concatenate([up8, conv2],axis=3)
	# up8 = merge([up8, conv2], mode='concat', concat_axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
	conv8 = BatchNormalization()(conv8)
	up9 = UpSampling2D((2,2))(conv8)
	up9 = concatenate([up9, conv1],axis=3)
	# up9 = merge([up9, conv1], mode='concat', concat_axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = BatchNormalization()(conv9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
	conv9 = BatchNormalization()(conv9)	
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)
	conv9 = BatchNormalization()(conv9)
	decoded_2 = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(conv9)
	model = Model(input_img, decoded_2)
	return model
