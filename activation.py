#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
K.set_image_data_format('channels_last')

from keras.layers import Layer, Activation, LeakyReLU, PReLU, ELU, ThresholdedReLU

import os

if( not os.path.exists( 'WiG/WiG_keras.py' ) ):
	import subprocess
	cmd = 'git clone https://github.com/mastnk/WiG'
	subprocess.call(cmd.split())	

from WiG.WiG_keras import WiG_Dense, WiG_Conv2D


class Swish(Layer):
	def __init__(self, **kwargs):
		super(Swish, self).__init__(**kwargs)
		
	def build(self, input_shape):
		self.beta = self.add_weight(name='beta', shape=(1,1,1,input_shape[-1]), initializer='ones', trainable=True)
		super(Swish, self).build(input_shape)
	
	def call(self, x):
		return x * K.sigmoid( self.beta * x )

	def compute_output_shape(self, input_shape):
		return input_shape

class SiL(Layer):
	def __init__(self, **kwargs):
		super(SiL, self).__init__(**kwargs)
		
	def build(self, input_shape):
		super(SiL, self).build(input_shape)
	
	def call(self, x):
		return x * K.sigmoid( x )

	def compute_output_shape(self, input_shape):
		return input_shape
		
def get( activation ):
	if( activation == 'LeakyReLU' ):
		act = LeakyReLU()

	elif( activation == 'PReLU' ):
		act = PReLU()
	
	elif( activation == 'ELU' ):
		act = ELU()
	
	elif( activation == 'ThresholdedReLU' ):
		act = ThresholdedReLU()

	elif( activation == 'Swish' ):
		act = Swish()

	elif( activation == 'SiL' ):
		act = SiL()

	else:
		act = Activation(activation)
	return act

def afterConv2D( activation, nb_features = None ):
	if( activation == 'WiG' ):
		act = WiG_Conv2D(nb_features, (3,3), padding='same', kernel_initializer='zeros')
	else:
		act = get( activation )
	return act

def afterDense( activation, nb_features = None ):
	if( activation == 'WiG' ):
		act = WiG_Dense(nb_features, kernel_initializer='zeros')
	else:
		act = get( activation )
	return act

custom_objects = {'WiG_Conv2D':WiG_Conv2D, 'WiG_Dense':WiG_Dense, 'SiL':SiL, 'Swish': Swish}

if( __name__ == '__main__' ):
	import os
	from keras.models import Sequential, load_model
	from keras.layers import Dense
	
	nb_features = 256
	temp = '__activation__.hdf5'
	activations = [ 'softmax', 'elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'LeakyReLU', 'PReLU', 'ThresholdedReLU', 'SiL', 'Swish', 'WiG' ]
	for act in activations:
		print( act )
		model = Sequential()
		model.add( Dense( nb_features, input_shape=(128,) ) )
		model.add( afterDense( act, 256 ) )
		model.save( temp )
		model = load_model( temp, custom_objects = custom_objects )
	os.remove(temp)
