from PIL import Image
import numpy as np
import glob
import math
from time import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
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

		X[i,:,:,:] = np.array(img.convert('L'))[:,:,np.newaxis]
		Y[i,:,:,:] = np.array(img)
		image.close()
	return X, Y

def initialize_model(image_size=imsize):
	model = Sequential()

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