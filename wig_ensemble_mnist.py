#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Add, Average, Multiply, Input, Activation, Lambda
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
def build_model( nb_layers = 3, dropout = 0, nb_features=256, Wl2=0, nb_classes = 10, input_shape = (28,28,1), nb_layers_weight = 0, nb_features_weight=None):
	inp = Input(shape=input_shape)
	x0 = Flatten() (inp)
	x = x0
	
	yy = []
	y = Dense( nb_classes, kernel_initializer='zeros', kernel_regularizer=l2(Wl2), use_bias=False ) (x)
	m = Dense( nb_classes, kernel_initializer='zeros', activation='sigmoid' ) (x)
	y = Multiply() ([m,y])

	if( dropout > 0 ):
		y = Dropout(dropout) (y)
	yy.append(y)
	
	for i in range(nb_layers):
		y = Dense( nb_features, kernel_initializer='he_normal', kernel_regularizer=l2(Wl2) ) (x)
		m = Dense( nb_features, kernel_initializer='zeros', activation='sigmoid' ) (x)
		x = Multiply() ([m,y])
		
		y = Dense( nb_classes, kernel_initializer='zeros', kernel_regularizer=l2(Wl2), use_bias=False ) (x)
		m = Dense( nb_classes, kernel_initializer='zeros', activation='sigmoid' ) (x)
		y = Multiply() ([m,y])

		if( dropout > 0 ):
			y = Dropout(dropout) (y)
		yy.append(y)
	
	if( nb_layers_weight > 0 ):
		if( nb_features_weight == None ):
			nb_features_weight = nb_features
		
		x = x0
		for i in range(nb_layers_weight-1):
			y = Dense( nb_features_weight, kernel_initializer='he_normal', kernel_regularizer=l2(Wl2) ) (x)
			m = Dense( nb_features_weight, kernel_initializer='zeros', activation='sigmoid' ) (x)
			x = Multiply() ([m,y])
		
		y = Dense( nb_layers+1, kernel_initializer='zeros', kernel_regularizer=l2(Wl2), use_bias=False ) (x)
		m = Dense( nb_layers+1, kernel_initializer='zeros', activation='sigmoid' ) (x)
		y = Multiply() ([m,y])
		w = Activation('softmax')(y)
		
		for i in range(nb_layers+1):
			wy = Lambda( lambda w: w[:,i:i+1], lambda s: (s[0], nb_classes) ) (w)
			yy[i] = Multiply() ( [wy, yy[i]] )
		y = Add() (yy)

	else:
		y = Average() (yy)
	
	y = Activation('softmax')(y)
	return Model(inputs=inp, outputs=y)

##### generator #####
def build_generator( X_train, Y_train, batch_size, gen = None ):
	if( gen == None ):
		gen = ImageDataGenerator(width_shift_range=0.5, height_shift_range=0.5, zoom_range=[0.98, 1.02], shear_range=3.14/180*0.1)
	gen.fit(X_train)
	flow_batch_size = batch_size
	if( flow_batch_size > X_train.shape[0] ):
		flow_batch_size = X_train.shape[0]
	flow = gen.flow(X_train, Y_train, batch_size=flow_batch_size)
	Xque, Yque = flow.__next__()
	
	while(True):
		while( Xque.shape[0] < batch_size ):
			_X, _Y = flow.__next__()
			Xque = np.concatenate( (Xque, _X), axis=0 )
			Yque = np.concatenate( (Yque, _Y), axis=0 )
		
		X = Xque[:batch_size, :]
		Y = Yque[:batch_size, :]
		
		Xque = Xque[batch_size:, :]
		Yque = Yque[batch_size:, :]
		
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
	
	batch_size = 1100
	
	nb_layers = 3
	nb_layers_weight = 2
	nb_features = 128
	dropout = 0.5
	Wl2 = 1E-6

	(X_train, Y_train), (X_test, Y_test) = train1000.mnist()
	
	model = build_model( nb_layers = nb_layers, dropout = dropout, nb_features = nb_features, Wl2=Wl2, nb_layers_weight=nb_layers_weight )
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_crossentropy', 'accuracy'])
	
	if( not os.path.exists( title+'.hdf5' ) ):
		gen = build_generator( X_train, Y_train, batch_size )

		flog = open( title+'.csv', 'w' )

		line = 'epoch, categorical_crossentropy, accuracy, val_categorical_crossentropy, val_accuracy, save'
		print( line )
		flog.write( line+'\n' )
		flog.flush()		
	
		min_crossentropy = None
		for epoch in range(epochs):
			for step in range(steps_per_epoch):
				x, y = gen.__next__()
				model.train_on_batch( x, y )
			
			eva = model.evaluate( X_train, Y_train, verbose=0 )
			crossentropy = eva[1]
			accuracy = eva[2]
			
			eva = model.evaluate( X_test, Y_test, verbose=0 )
			val_crossentropy = eva[1]
			val_accuracy = eva[2]
			
			if( min_crossentropy == None or min_crossentropy > crossentropy ):
				min_crossentropy = crossentropy
				model.save( title+'.hdf5' )
				save = '*'
			else:
				save = '-'
			
			line = '{epoch}, {crossentropy}, {accuracy}, {val_crossentropy}, {val_accuracy}, {save}'.format(epoch=epoch, crossentropy=crossentropy, accuracy=accuracy, val_crossentropy=val_crossentropy, val_accuracy=val_accuracy, save=save)
			print( line )
			flog.write( line+'\n' )
			flog.flush()
		
		flog.close()
		
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
