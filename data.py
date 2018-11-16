#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

from keras import datasets
from keras.utils import np_utils

import numpy as np

def mnist():
	(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	y_train = np_utils.to_categorical(y_train, 10).astype('float32')
	y_test = np_utils.to_categorical(y_test, 10).astype('float32')

	x_train = x_train.reshape( x_train.shape+(1,) )
	x_test = x_test.reshape( x_test.shape+(1,) )

	return (x_train, y_train), (x_test, y_test)

def fashion_mnist():
	(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	y_train = np_utils.to_categorical(y_train, 10).astype('float32')
	y_test = np_utils.to_categorical(y_test, 10).astype('float32')

	x_train = x_train.reshape( x_train.shape+(1,) )
	x_test = x_test.reshape( x_test.shape+(1,) )

	return (x_train, y_train), (x_test, y_test)


def cifar10():
	(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
	Y_train = np_utils.to_categorical(y_train, 10)
	Y_train = np.single( Y_train )
	Y_test  = np_utils.to_categorical(y_test, 10)
	Y_test = np.single( Y_test )
	
	X_train = X_train.astype('float32')/255.0
	X_test = X_test.astype('float32')/255.0

	return (X_train, Y_train), (X_test, Y_test)

def cifar100():
	(X_train, y_train), (X_test, y_test) = datasets.cifar100.load_data()
	Y_train = np_utils.to_categorical(y_train, 100)
	Y_train = np.single( Y_train )
	Y_test  = np_utils.to_categorical(y_test, 100)
	Y_test = np.single( Y_test )
	
	X_train = X_train.astype('float32')/255.0
	X_test = X_test.astype('float32')/255.0

	return (X_train, Y_train), (X_test, Y_test)

if( __name__ == '__main__' ):
	print( 'mnist' )
	(x_train, y_train), (x_test, y_test) = mnist()
	print( 'dtype:' )
	print( 'x_train:', x_train.dtype )
	print( 'y_train:', y_train.dtype )
	print( 'x_test:', x_test.dtype )
	print( 'y_test:', y_test.dtype )
	print()

	print( 'shape:' )
	print( 'x_train:', x_train.shape )
	print( 'y_train:', y_train.shape )
	print( 'x_test:', x_test.shape )
	print( 'y_test:', y_test.shape )
	print()
	
	print( 'fashion_mnist' )
	(x_train, y_train), (x_test, y_test) = fashion_mnist()
	print( 'dtype:' )
	print( 'x_train:', x_train.dtype )
	print( 'y_train:', y_train.dtype )
	print( 'x_test:', x_test.dtype )
	print( 'y_test:', y_test.dtype )
	print()

	print( 'shape:' )
	print( 'x_train:', x_train.shape )
	print( 'y_train:', y_train.shape )
	print( 'x_test:', x_test.shape )
	print( 'y_test:', y_test.shape )
	print()
	
	print( 'cifar10' )
	(x_train, y_train), (x_test, y_test) = cifar10()
	print( 'dtype:' )
	print( 'x_train:', x_train.dtype )
	print( 'y_train:', y_train.dtype )
	print( 'x_test:', x_test.dtype )
	print( 'y_test:', y_test.dtype )
	print()

	print( 'shape:' )
	print( 'x_train:', x_train.shape )
	print( 'y_train:', y_train.shape )
	print( 'x_test:', x_test.shape )
	print( 'y_test:', y_test.shape )
	print()
	
	print( 'cifar100' )
	(x_train, y_train), (x_test, y_test) = cifar100()
	print( 'dtype:' )
	print( 'x_train:', x_train.dtype )
	print( 'y_train:', y_train.dtype )
	print( 'x_test:', x_test.dtype )
	print( 'y_test:', y_test.dtype )
	print()

	print( 'shape:' )
	print( 'x_train:', x_train.shape )
	print( 'y_train:', y_train.shape )
	print( 'x_test:', x_test.shape )
	print( 'y_test:', y_test.shape )
	print()
	
