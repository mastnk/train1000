#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

import sample_cifar10 as cifar
import os
import sys

if( not os.path.exists( 'WiG/keras/activation.py' ) ):
	import subprocess
	cmd = 'git clone https://github.com/mastnk/WiG'
	subprocess.call(cmd.split())
	
sys.path.append('WiG/keras')
import activation

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
	
	(X_train, Y_train), (X_test, Y_test) = train1000.cifar100()
	nb_classes = 100
	
	model = cifar.build_model( nb_classes=nb_classes, Wl20=1E-6, dr0=0.5, nb_features0=1024 )
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['categorical_crossentropy', 'accuracy'])

	if( not os.path.exists( title + '.hdf5' ) ):
		gen = cifar.build_generator( X_train, Y_train, batch_size )

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
