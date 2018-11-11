#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, SpatialDropout2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

import numpy as np
import random

import os
import sys

if( not os.path.exists( 'WiG/keras/activation.py' ) ):
	import subprocess
	cmd = 'git clone https://github.com/mastnk/WiG'
	subprocess.call(cmd.split())
	
sys.path.append('WiG/keras')
import activation

##### model #####
def build_model( nb_classes=10, Wl20=0, dr0=0, nb_features0=1024, act='relu' ):
	model = Sequential()
	L = 0
	
	##### 32x32
	nb_features = nb_features0//16
	dropout = dr0/8.0
	Wl2 = Wl20/8.0
	
	#
	name = 'Conv2D_{L:02d}'.format(L=L); L+=1
	model.add(Conv2D(nb_features, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Wl2), name=name, input_shape=(32,32,3)))
	model.add( activation.afterConv2D(act, nb_features) )
	
	#
	if( dropout > 0 ):
		model.add(SpatialDropout2D(dropout))
	name = 'Conv2D_{L:02d}'.format(L=L); L+=1
	model.add(Conv2D(nb_features, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Wl2), name=name))
	model.add( activation.afterConv2D(act, nb_features) )
	
	##### 32x32 -> 16x16
	nb_features = nb_features0//8
	dropout = dr0/4.0
	Wl2 = Wl20/4.0
	
	#
	if( dropout > 0 ):
		model.add(SpatialDropout2D(dropout))
	name = 'Conv2DPooling_{L:02d}'.format(L=L); L+=1
	model.add(Conv2D( nb_features, (2,2), strides=(2,2), kernel_initializer='he_normal', padding='valid', kernel_regularizer=l2(Wl2), name=name ))
	model.add( activation.afterConv2D(act, nb_features) )

	##### 16x16
	for i in range(2):
		#
		if( dropout > 0 ):
			model.add(SpatialDropout2D(dropout))
		name = 'Conv2D_{L:02d}'.format(L=L); L+=1
		model.add(Conv2D(nb_features, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Wl2), name=name))
		model.add( activation.afterConv2D(act, nb_features) )
	
	##### 16x16 -> 8x8
	nb_features = nb_features0//4
	dropout = dr0/2.0
	Wl2 = Wl20/2.0

	#
	if( dropout > 0 ):
		model.add(SpatialDropout2D(dropout))
	name = 'Conv2DPooling_{L:02d}'.format(L=L); L+=1
	model.add(Conv2D( nb_features, (2,2), strides=(2,2), kernel_initializer='he_normal', padding='valid', kernel_regularizer=l2(Wl2), name=name ))
	model.add( activation.afterConv2D(act, nb_features) )
	
	##### 8x8
	for i in range(2):
		#
		if( dropout > 0 ):
			model.add(SpatialDropout2D(dropout))
		name = 'Conv2D_{L:02d}'.format(L=L); L+=1
		model.add(Conv2D(nb_features, (3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(Wl2), name=name))
		model.add( activation.afterConv2D(act, nb_features) )

	#
	model.add(Flatten())
	nb_features = nb_features0
	dropout = dr0
	Wl2 = Wl20
	
	for i in range(2):
		if( dropout > 0 ):
			model.add(Dropout(dropout))
		name = 'Dense_{L:02d}'.format(L=L); L+=1
		model.add(Dense(nb_features, kernel_initializer='he_normal', kernel_regularizer=l2(Wl2), name=name))
		model.add( activation.afterDense(act, nb_features) )
	
	if( dropout > 0 ):
		model.add(Dropout(dropout))
	name = 'Dense_{L:02d}'.format(L=L); L+=1
	model.add(Dense(nb_classes, kernel_initializer='he_normal', kernel_regularizer=l2(Wl2), name=name, activation='softmax'))
	
	return model


##### generator #####
def build_generator( X_train, Y_train, batch_size, gen = None ):
	if( gen == None ):
		gen = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True, rotation_range=5.0, zoom_range=[0.99, 1.05], shear_range=3.14/180)
	gen.fit(X_train)
	flow = gen.flow(X_train, Y_train, batch_size=batch_size)
	while(True):
		X, Y = flow.__next__()
		if( X.shape[0] == batch_size ):
			for i in range(X.shape[0]):
				a = random.uniform( 0.95, 1.05 )
				X[i,:,:,0] *= a
				a = random.uniform( 0.95, 1.05 )
				X[i,:,:,2] *= a
				
				a = random.uniform( 0.95, 1.05 )
				b = random.uniform( -0.005, +0.005 )
				X[i,:,:,:] = X[i,:,:,:] * a + b - (a-1.0)/2.0
			yield X, Y
			

if( __name__ == '__main__' ):
	from keras.callbacks import CSVLogger, ModelCheckpoint
	from keras.optimizers import Adam
	from keras.models import load_model
	import train1000
	import os
	import sys

	title, ext = os.path.splitext(sys.argv[0]);
	
	epochs = 100
	steps_per_epoch = 100
	
	batch_size = 100
	
	(X_train, Y_train), (X_test, Y_test) = train1000.cifar10()
	nb_classes = 10
	
	model = build_model( nb_classes=nb_classes, Wl20=1E-6, dr0=0.5, nb_features0=1024 )
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_crossentropy', 'accuracy'])

	if( not os.path.exists( title + '.hdf5' ) ):
		gen = build_generator( X_train, Y_train, batch_size )

		callbacks = [ModelCheckpoint(title + '.hdf5', monitor='val_categorical_crossentropy', verbose=1, save_best_only=True, mode='min'), CSVLogger(title+'.csv')]
		model.fit_generator( gen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, callbacks=callbacks, validation_data=(X_train, Y_train) )
	
	model.load_weights( title + '.hdf5' )
	
	print( 'train data:' )
	eva = model.evaluate( X_train, Y_train, verbose=0 )
	for i in range(1,len(model.metrics_names)):
		print( model.metrics_names[i] + ' : ', eva[i] )

	print()

	print( 'test data:' )
	eva = model.evaluate( X_test, Y_test, verbose=0 )
	for i in range(1,len(model.metrics_names)):
		print( model.metrics_names[i] + ' : ', eva[i] )
