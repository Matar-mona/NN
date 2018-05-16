from PIL import Image, ImageCms
from skimage.color import rgb2lab, lab2rgb 
import numpy as np
import glob
import math
import sys
import h5py

def main(image_size):
	imagelist = glob.glob('*.jpg')

	X = np.zeros((len(imagelist),image_size,image_size,1))
	print X.itemsize
	Y = np.zeros((len(imagelist),image_size,image_size,2))

	for i, impath in enumerate(imagelist):
		sys.stdout.write('\rImage {}'.format(i))
		sys.stdout.flush()
		
		image = Image.open(impath)

		max_dims = math.floor(np.min(image.size)/image_size)*image_size 
		square_box = (image.size[0]/2-max_dims/2,
					  image.size[1]/2-max_dims/2,
					  image.size[0]/2+max_dims/2,
					  image.size[1]/2+max_dims/2)
		img = image.crop(box=square_box)
		img = img.resize(size=(image_size,image_size))

		img_arr = np.array(img)/255.
		X[i,:,:,:] = rgb2lab(img_arr)[:,:,0].reshape(256,256,1)
		Y[i,:,:,:] = rgb2lab(img_arr)[:,:,1:]
		image.close()
	Y /= 128.
	
	with h5py.File('colorize_image_data.h5', 'w') as hf:
		hf.create_dataset('X',data=X)
		hf.create_dataset('Y',data=Y)

if __name__ == '__main__':
	imsize = 256
	main(imsize)