from skimage.color import rgb2lab, lab2rgb 
import numpy as np
import glob
import math
import h5py
from time import time
import matplotlib.pyplot as plt

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


class OutputObserver(Callback):
	""""
	Callback to observe the output of the network
	"""

	def __init__(self, n, nn, x_train, y_train):
		self.out_log = []
		self.nn = nn
		self.n = n # number of epochs before output
		self.x_train = x_train
		self.y_train = y_train
	
	def on_epoch_end(self, epoch, logs={}):
		if epoch % self.n == 0:
			prediction = self.nn.predict(self.x_train)
			show_image(self.x_train[0], self.y_train[0], prediction[0], Save=True, epoch=epoch)

def show_image(L_image, ab_image, ab_image_pred, Save=False, epoch=None):

	prediction = np.zeros((256,256,3))
	prediction[:,:,0] = L_image[:,:,0]
	prediction[:,:,1:] = ab_image_pred*128

	ground_truth = np.zeros((256,256,3))
	ground_truth[:,:,0] = L_image[:,:,0]
	ground_truth[:,:,1:] = ab_image*128

	fig = plt.figure()

	ax = fig.add_subplot(221)
	ax.imshow(L_image[:,:,0], cmap='gray')
	ax.axis('off')
	ax.set_title('Input')

	ax = fig.add_subplot(222)
	ax.imshow(lab2rgb(ground_truth))
	ax.axis('off')
	ax.set_title('True image')
	
	ax = fig.add_subplot(223)
	ax.imshow(lab2rgb(prediction))
	ax.axis('off')
	ax.set_title('Predicted image')

	ax = fig.add_subplot(224)
	ax.set_title('Histogram')
	ax.hist(prediction[:,:,1].flatten(), bins=40, label='green-red')
	ax.hist(prediction[:,:,2].flatten(), bins=40, label='blue-yellow')
	ax.set_yscale('log') 

	if Save:
		plt.savefig('./predictions/prediction_{}.png'.format(epoch))
		plt.close()
	else:
		plt.show()


def load_data(path):
	print('Loading in data...\n')
	with h5py.File(path,'r') as hf:
		X = hf['X'][:]
		Y = hf['Y'][:]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

	return X_train, X_test, Y_train, Y_test

def initialize_model(kernel_size, learning_rate):

	model = Sequential()

	model.add(Conv2D(16, kernel_size, strides=2, padding='same', activation='relu', input_shape=(256,256,1)))

	model.add(BatchNormalization())
	
	model.add(Conv2D(32, kernel_size, strides=2, padding='same', activation='relu'))

	model.add(BatchNormalization())
	
	model.add(Conv2D(64, kernel_size, strides=2, padding='same', activation='relu'))

	model.add(BatchNormalization())

	model.add(UpSampling2D((2, 2)))
	
	model.add(Conv2D(32, kernel_size, padding='same', activation='relu'))

	model.add(BatchNormalization())

	model.add(UpSampling2D((2, 2)))
	
	model.add(Conv2D(16, kernel_size, padding='same', activation='relu'))

	model.add(BatchNormalization())

	model.add(UpSampling2D((2, 2)))

	model.add(Conv2D(2, kernel_size, padding='same', activation='tanh'))

	model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error')

	model.summary()

	return model

def train_model(model, X_train, Y_train, X_test, Y_test, num_batches, num_epochs):

	show_output = OutputObserver(50,model,X_train,Y_train)

	model.fit(X_train, Y_train, validation_data=(X_test,Y_test), 
			  batch_size=num_batches, epochs=num_epochs, callbacks=[show_output])


def main():
	start = time()

	path = '../../colorize_image_data.h5'

	#Load in data
	X_train, X_test, Y_train, Y_test = load_data(path)
	
	#Initialize model 
	model = initialize_model(kernel_size=4, learning_rate=0.01)
	train_model(model, X_train, Y_train, X_test, Y_test, num_batches=5, num_epochs=500)

	end = time()-start
	print('Time taken: {:.1f} s'.format(end))

if __name__ == '__main__':
	main()
