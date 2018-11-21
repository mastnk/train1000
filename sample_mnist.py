#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2

import numpy as np

import os
import sys

if( not os.path.exists( 'WiG/keras/activation.py' ) ):
	import subprocess
	cmd = 'git clone https://github.com/mastnk/WiG'
	subprocess.call(cmd.split())
	
sys.path.append('WiG/keras')
import activation


##### model #####
def build_model( nb_layers = 3, dropout = 0, nb_features=256, Wl2=0, act_func='relu', nb_classes = 10, input_shape = (28,28,1) ):
	model = Sequential()
	
	model.add( Flatten( input_shape=input_shape ) )
	for i in range(nb_layers-1):
		model.add( Dense( nb_features, kernel_regularizer=l2(Wl2) ) )
		model.add( activation.afterDense( act_func, nb_features ) )
		
	if( dropout > 0 ):
		model.add( Dropout(dropout) )

	model.add( Dense( nb_classes, activation='softmax', kernel_regularizer=l2(Wl2) ) )
	return model

##### generator #####
def build_generator( X_train, Y_train, batch_size, gen = None ):
	if( gen == None ):
		gen = ImageDataGenerator(width_shift_range=2, height_shift_range=2, zoom_range=[0.9, 1.1], shear_range=3.14/180*5)
	gen.fit(X_train)
	flow = gen.flow(X_train, Y_train, batch_size=batch_size)
	while(True):
		X, Y = flow.__next__()
		if( X.shape[0] == batch_size ):
			yield X, Y

if( __name__ == '__main__' ):
	from keras.callbacks import CSVLogger, ModelCheckpoint
	from keras.optimizers import Adam
	from keras.models import load_model
	import train1000
	import os
	import sys

	title, ext = os.path.splitext(sys.argv[0])
		

	epochs = 100
	steps_per_epoch = 100
	
	batch_size = 100
	
	nb_layers = 5
	nb_features = 128
	dropout = 0.5
	Wl2 = 1E-6

	if( len(sys.argv) > 2 and sys.argv[1] == 'fashion' ):
		title += '_fashion'
		(X_train, Y_train), (X_test, Y_test) = train1000.fashion_mnist()
	else:
		(X_train, Y_train), (X_test, Y_test) = train1000.mnist()
	
	model = build_model( nb_layers = nb_layers, dropout = dropout, nb_features = nb_features, Wl2=Wl2 )
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_crossentropy', 'accuracy'])

	if( not os.path.exists( title+'.hdf5' ) ):

		gen = build_generator( X_train, Y_train, batch_size )

		callbacks = [ModelCheckpoint(title+'.hdf5', monitor='val_categorical_crossentropy', verbose=1, save_best_only=True, mode='min'), CSVLogger(title+'.csv')]
		model.fit_generator( gen, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=1, callbacks=callbacks, validation_data=(X_train, Y_train) )
	
	model.load_weights( title+'.hdf5' )
	
	print( 'train data:' )
	eva = model.evaluate( X_train, Y_train, verbose=0 )
	for i in range(1,len(model.metrics_names)):
		print( model.metrics_names[i] + ' : ', eva[i] )

	print()

	print( 'test data:' )
	eva = model.evaluate( X_test, Y_test, verbose=0 )
	for i in range(1,len(model.metrics_names)):
		print( model.metrics_names[i] + ' : ', eva[i] )
