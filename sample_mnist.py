#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

import numpy as np



i = 0
title, ext = os.path.splitext(sys.argv[i]); i+=1
act = sys.argv[i]; i+=1
batch_size = int(sys.argv[i]); i+=1
soft_positive = float(sys.argv[i]); i+=1
truncated_th = float(sys.argv[i]); i+=1

filetitle = '{title}_{act}_{batch_size:04d}_{soft_positive:04.2f}_{truncated_th:04.2f}'.format(title=title, act=act, batch_size=batch_size, soft_positive=soft_positive, truncated_th=truncated_th)

nb_per_class = 100
nb_updates = 1000


##### data #####
(x_train, y_train), (x_test, y_test) = data.mnist()
(x_train, y_train) = data.extract( x_train, y_train, nb_per_class )

x_train = x_train.reshape( ( x_train.shape[0], x_train.shape[1]*x_train.shape[2] ) )
x_test = x_test.reshape( ( x_test.shape[0], x_test.shape[1]*x_test.shape[2] ) )


##### model #####
nb_classes = 10
Wl2 = 0
dropout = 0
nb_features = 512
model = mnist_model.shallow( nb_classes=nb_classes, Wl2=Wl2, dropout=dropout, nb_features=nb_features, activation=act )

tsce = soft.truncated_soft_crossentropy( truncated_th = truncated_th, soft_positive = soft_positive, nb_classes = nb_classes )
loss = tsce.K_truncated_soft_crossentropy

'''
if( truncated_th == 1.0 and soft_positive == 1.0 ):
	loss = 'categorical_crossentropy'
else:
	loss = tsce.K_truncated_soft_crossentropy
'''

#opt = SGD(lr=1, momentum=0.5)
opt = SGD(lr=1, momentum=0.9)
#opt = SGD(lr=1, momentum=0.0)
model.compile(loss=loss, optimizer=opt, metrics=['categorical_crossentropy', 'accuracy'])

##### train #####
gen = data.gen_sample( x_train, y_train, batch_size )
#gen = data.gen_choices( x_train, y_train, batch_size )
#gen = data.gen( x_train, y_train, batch_size )

flog = open( filetitle+'.csv', 'w' )

line = 'i, hce, acc, val_hce, val_acc'
print( line )
flog.write( line+'\n' )
flog.flush()

for i in range(nb_updates):


	x, y = gen.__next__()

	y_train_pred = model.predict( x_train )
	y_test_pred = model.predict( x_test )

	hce = data.np_crossentropy( y_train, y_train_pred )
	acc = data.np_accuracy( y_train, y_train_pred )
	val_hce = data.np_crossentropy( y_test, y_test_pred )
	val_acc = data.np_accuracy( y_test, y_test_pred )
	
	'''
	eva = model.evaluate( x=x_train, y=y_train, verbose=0 )
	hce = eva[1]
	acc = eva[2]

	eva = model.evaluate( x=x_test, y=y_test, verbose=0 )
	val_hce = eva[1]
	val_acc = eva[2]
	'''
	
	line = '{i}, {hce}, {acc}, {val_hce}, {val_acc}'.format(i=i, hce=hce, acc=acc, val_hce=val_hce, val_acc=val_acc)
	print( line )
	flog.write( line+'\n' )
	flog.flush()
	
	model.train_on_batch( x, y )

flog.close()
model.save_weights(filetitle+'.hdf5')

