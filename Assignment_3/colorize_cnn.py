from PIL import Image, ImageCms
from skimage.color import rgb2lab, lab2rgb 
import numpy as np
import glob
import math
from time import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_data(path):
	
	with h5py.File(path,'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

	return X_train, X_test, Y_train, Y_test

def initialize_model(image_size):

	model = Sequential()

	model.add(Conv2D(16, (5,5), strides=3, activation='relu', input_shape=(imsize,imsize,1)))
	
	model.add(Conv2D(32, (5,5), strides=3, activation='relu'))
	
	model.add(Conv2D(64, (5,5), strides=3, activation='relu'))

	model.add(UpSampling2D((2, 2)))
	
	model.add(Conv2D(32, (5,5), strides=3, activation='relu'))

	model.add(UpSampling2D((2, 2)))
	
	model.add(Conv2D(16, (5,5), strides=3, activation='tanh'))

	model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error')

	model.summary()
	return model

def main():
	start = time()

	path = 'colorize_image_data.h5'

	#Load in data
	X_train, X_test, Y_train, Y_test = load_data(path)
	
	#Initialize model 
	#model = initialize_model(image_size=imsize)

	end = time()-start
	print 'Time taken: {:.1f} s'.format(end)

if __name__ == '__main__':
	main()