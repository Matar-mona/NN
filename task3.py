import numpy as np
import matplotlib.pyplot as plt

def calc_feature(number):
    number_feature = np.zeros(len(number))
    ones_number = 1*(number > 0.0)
    for i in range(len(number)):
        temp = ones_number[i,:].reshape(16,16)
        number_feature[i] = np.sum(temp[8:,:])
    return number_feature

def main():
    train_x = np.genfromtxt('data/train_in.csv', delimiter=',')
    train_y = np.genfromtxt('data/train_out.csv', delimiter=',')

    test_x = np.genfromtxt('data/test_in.csv', delimiter=',')
    test_y = np.genfromtxt('data/test_out.csv', delimiter=',')

    class1_train = train_x[train_y == 5,:]
    class2_train = train_x[train_y == 7,:]
        
    class1_feature_train = calc_feature(class1_train)  
    class2_feature_train = calc_feature(class2_train)

    binbounds = np.linspace(class2_feature_train.min(),class2_feature_train.max(), 20)
    
    n_class1, bins_class1, patches = plt.hist(class1_feature_train, bins=binbounds, alpha=0.5, normed=True)
    n_class2, bins_class2, patches = plt.hist(class2_feature_train, bins=binbounds, alpha=0.5, normed=True)
    plt.show()
    
    class1 = test_x[test_y == 5,:]
    class2 = test_x[test_y == 7,:]
    
    print 'Number of 5s: ', len(class1)
    print 'Number of 7s: ', len(class2)
    
    prior_class1 = float(len(class1_train))/(len(class1_train)+len(class2_train))
    prior_class2 = float(len(class2_train))/(len(class1_train)+len(class2_train))
    
    class1_feature = calc_feature(class1)  
    class2_feature = calc_feature(class2)
    
    all_n = np.concatenate((class1_feature,class2_feature))
    
    binned_class1 = np.digitize(all_n, bins_class1)
    binned_class1[binned_class1 >= len(n_class1)] = len(n_class1)-1
    binned_class2 = np.digitize(all_n, bins_class2)
    binned_class2[binned_class2 >= len(n_class2)] = len(n_class2)-1
    
    classified_class1 = np.zeros(len(all_n))
    for i in range(len(all_n)):
         if (n_class1[binned_class1[i]-1]*prior_class1 > n_class2[binned_class2[i]-1]*prior_class2):
            classified_class1[i] = 1
      
    accuracy = (np.sum(1*classified_class1[:len(class1)]==1)+
    				 np.sum(1*classified_class1[len(class1):]==0))/float(len(classified_class1))
    print 'Accuracy : {}%'.format(accuracy*100)

if __name__ == '__main__':
    main()