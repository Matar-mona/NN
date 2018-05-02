from PIL import Image, ImageCms
from skimage.color import rgb2lab, lab2rgb 
import numpy as np
import glob
import math
from time import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split

def load_data(image_size):
	imagelist = glob.glob('Images/*.jpg')

	X = np.zeros((len(imagelist),image_size,image_size,1))
	Y = np.zeros((len(imagelist),image_size,image_size,3))

	for i, impath in enumerate(imagelist):
		print i
		image = Image.open(impath)

		max_dims = math.floor(np.min(image.size)/1000)*1000 
		square_box = (image.size[0]/2-max_dims/2,
					  image.size[1]/2-max_dims/2,
					  image.size[0]/2+max_dims/2,
					  image.size[1]/2+max_dims/2)
		image = image.crop(box=square_box)
		img = image.resize(size=(image_size,image_size))

		img_arr = np.array(img)/255.
		X[i,:,:,:] = rgb2lab(img_arr)[:,:,0]
		Y[i,:,:,:] = rgb2lab(img_arr)[:,:,1:]
		image.close()
	Y /= 128.
	return X, Y

def initialize_model(image_size=imsize):

	model = Sequential()
	model.add(ZeroPadding2D(1,1), input_shape=(imsize,imsize,1))
	model.add(Conv2D(16, (5,5), strides=3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(32, (5,5), strides=3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(64, (5,5), strides=3, activation='relu'))
	model.add(UpSampling2D((2, 2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(32, (5,5), strides=3, activation='relu'))
	model.add(UpSampling2D((2, 2)))
	model.add(ZeroPadding2D((1,1)))
	model.add(Conv2D(16, (5,5), strides=3, activation='relu'))

def main():
	start = time()

	imsize = 1000

	#Load in data
	X, Y = load_data(image_size=imsize)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

	#Initialize model 
	model = initialize_model(image_size=imsize)

	end = time()-start
	print 'Time taken: {:.1f} s'.format(end)

if __name__ == '__main__':
	main()