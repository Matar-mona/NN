import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
sns.set_style('ticks')
plt.rc('text', usetex=True)

#w = np.random.randn(2)
#b = np.random.randn()
def sigmoid(X)
	return 1/(1+np.exp(-X))

def heaviside(X, boundary=0):
	return 1*(X>boundary)

def Q1_3():
	w = np.array([.457029, .2394019])
	b = -.4672
		
	xx, yy = np.meshgrid(np.linspace(0,1,100),
						 np.linspace(0,1,100))
								
	X = np.stack((np.ravel(xx),np.ravel(yy)))

	y_h = heaviside(np.dot(w.T,X) + b)
	y_h = y_h.reshape(xx.shape)

	plt.contour(xx,yy,y_h, levels=[0.5])
	plt.title('Decision boundary for AND')
	plt.xlabel('X1')
	plt.ylabel('X2')
	plt.show()

def Q1_4():
	w = np.random.randn(2,3)

	a11 = sigmoid(np.dot(w[:,0].T,X))
	a12 = sigmoid(np.dot(w[:,1].T,X))

	a21 = sigmoid(np.dot(w[:,2].T,np.array([a11,a12])))

	y_h = heaviside(a21,0.5)

def main():
	X1 = np.array([0,0,1,1])
	X2 = np.array([0,1,0,1])
	X = np.stack((X1,X2))
	

if __name__ == '__main__':
	main()