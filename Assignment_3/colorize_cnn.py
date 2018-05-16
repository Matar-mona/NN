from skimage.color import rgb2lab, lab2rgb 
import numpy as np
import glob
import math
import h5py
from time import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from keras.callbacks import Callback, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam, RMSprop
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
	"""
	Show an image containing input image, ground truth, prediction and a histogram of ab-values of the prediction

	L_image: Input black and white image
	ab_image: True ab-values of the image
	ab_image_pred: Predicted ab-values of the image
	Save: Whether to save the image
	epoch: Current epoch
	"""
	prediction = np.zeros((256,256,3))
	prediction[:,:,0] = L_image[:,:,0]
	prediction[:,:,1:] = ab_image_pred*128

	ground_truth = np.zeros((256,256,3))
	ground_truth[:,:,0] = L_image[:,:,0]
	ground_truth[:,:,1:] = ab_image*128

	fig = plt.figure(figsize=(10,10))

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
	ax.set_title('AB values of prediction')
	ax.hist(prediction[:,:,1].flatten(), bins=40, alpha=0.4, label='green-red')
	ax.hist(prediction[:,:,2].flatten(), bins=40, alpha=0.4, label='blue-yellow')
	ax.legend()
	ax.set_yscale('log') 

	if Save:
		plt.savefig('./predictions/prediction_{}.png'.format(epoch), dpi=300)
		plt.close()
	else:
		plt.show()


def load_data(path):
	"""
	Load in the data from from the file and split the data into a training and a test set

	path: Path to .h5 file containing all the data
	"""
	print('Loading in data...\n')
	with h5py.File(path,'r') as hf:
		X = hf['X'][:50]
		Y = hf['Y'][:50]

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

	return X_train, X_test, Y_train, Y_test

def initialize_model(kernel_size, learning_rate):
	"""
	Initialize the neural network and create the model

	kernel_size: Kernel size of the convolutional layers
	learning_rate: Learning rate of the network
	"""

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

	#Specify optimizer, learning rate and loss function
	model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')

	model.summary()

	return model

def train_model(model, X_train, Y_train, X_test, Y_test, num_batches, num_epochs):
	"""
	Train the model

	model: Compiled model
	X_train, Y_train: Training data
	X_test, Y_test: Validation data
	num_batches: The batch size for training the network
	num_epochs: The number of epochs to train the network

	Return: Training and validation loss
	"""

	#Create a callback to output an image every 50 epochs
	show_output = OutputObserver(50,model,X_train,Y_train)

	history = model.fit(X_train, Y_train, validation_data=(X_test,Y_test), 
			  batch_size=num_batches, epochs=num_epochs, callbacks=[show_output, TensorBoard(log_dir='./Graph')])

	#Save the model
	model.save('color_cnn.h5')

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	return loss, val_loss

def main():
	start = time()

	#Path to data
	path = '../../colorize_image_data.h5'

	#Load in data
	X_train, X_test, Y_train, Y_test = load_data(path)
	
	#Initialize model 
	model = initialize_model(kernel_size=4, learning_rate=0.001)
	loss, val_loss = train_model(model, X_train, Y_train, X_test, Y_test, num_batches=5, num_epochs=1501)

	#Plot the losses
	plt.plot(np.arange(len(loss)), loss, label='Training loss')
	plt.plot(np.arange(len(val_loss)), val_loss, label='Validation loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.yscale('log')
	plt.legend()
	plt.savefig('losses.png', dpi=300)
	plt.close()

	end = time()-start
	print('Time taken: {:.1f} s'.format(end))

if __name__ == '__main__':
	main()
