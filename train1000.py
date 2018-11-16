#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

import numpy as np

import data

def extract( x_src, y_src, nb_per_class ):
	nb_classes = y_src.shape[1]
	nb_data = nb_classes * nb_per_class
	
	x_dst = np.zeros( ( nb_data,) + x_src.shape[1:], dtype=np.float32 )
	y_dst = np.zeros( ( nb_data,) + y_src.shape[1:], dtype=np.float32 )
	
	nb = [0] * nb_classes
	
	k = 0
	for i in range(y_src.shape[0]):
		ind = y_src[i,:].tolist().index(1.0)
		if( nb[ind] < nb_per_class ):
			x_dst[k,:] = x_src[i,:]
			y_dst[k,:] = y_src[i,:]
			nb[ind] += 1
			k += 1
			if( k >= nb_data ):
				break
	
	return ( x_dst, y_dst )

def mnist():
	nb_per_class = 100
	(X_train, Y_train), (X_test, Y_test) = data.mnist()
	(X_train, Y_train) = extract( X_train, Y_train, nb_per_class )
	return (X_train, Y_train), (X_test, Y_test)

def fashion_mnist():
	nb_per_class = 100
	(X_train, Y_train), (X_test, Y_test) = data.fashion_mnist()
	(X_train, Y_train) = extract( X_train, Y_train, nb_per_class )
	return (X_train, Y_train), (X_test, Y_test)

def cifar10():
	nb_per_class = 100
	(X_train, Y_train), (X_test, Y_test) = data.cifar10()
	(X_train, Y_train) = extract( X_train, Y_train, nb_per_class )
	return (X_train, Y_train), (X_test, Y_test)

def cifar100():
	nb_per_class = 10
	(X_train, Y_train), (X_test, Y_test) = data.cifar100()
	(X_train, Y_train) = extract( X_train, Y_train, nb_per_class )
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
	
