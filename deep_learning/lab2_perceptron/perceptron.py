import numpy as np
import pandas
import matplotlib.pyplot as plt

def calc_classification_accuracy(w, x, t):
    z = np.sign(np.dot(w.T,x)).T #perceptron classification output for all samples
    z = np.where(z < 0, 0, z) #change -1 to 0
    diff = t - z
    #print diff
    #print sum(num == 0 for num in diff)
    num_right = float(sum(num == 0 for num in diff))
    return (num_right/float(np.size(diff)))*100.0

#Load Iris Data Set-----------------------------------
data = pandas.read_csv( 'Fisher.csv' )
m = data.as_matrix()
labels = m[:,0]
labels[ labels==2 ] = 1  #label everything that is class 2 as class 1 so it is binary
labels = np.atleast_2d( labels ).T
features = m[:,1:5]

# Initialize weights
num_outputs = 1 #since it's a binary problem (1 or 0)
num_examples, num_features = np.shape(features)
num_iterations = 100
learning_rates = np.array([0.01, 0.1, 1.0]) #learning rate
accuracy_array = np.zeros((3,num_iterations))
weight_norm_array = np.zeros((3,num_iterations))
#dimension of x is 5x150
x = np.concatenate((features.T,np.ones([1,num_examples])),axis=0) #stack  ones on to multiply the bias by
#dimension of t is 150x1
t = labels
#dimension of w is 5x1
w0 = np.random.rand(num_features + 1, num_outputs) #the +1 is because I'm including the bias in the weights
# for c in xrange(0,len(learning_rates)):
plt.figure()
for j in range(len(learning_rates)):
    w = w0
    c = learning_rates[j]
    for i in range(num_iterations):
        accuracy_array[j,i] = calc_classification_accuracy(w, x, t)
        weight_norm_array[j,i] = np.sqrt(np.dot(w.T,w))
        zi = np.sign(np.dot(w.T,x[:,i])) #perceptron classification output for sample i
        zi = np.where(zi < 0, 0, zi) #change -1 to 0
        # update weights
        del_w = np.atleast_2d(c*np.dot((t[i,0]-zi[0]).T,x[:,i].T))
        w = w + del_w.T
    plt.subplot(121)    
    plt.plot(accuracy_array[j,:])
    plt.subplot(122)
    plt.plot(weight_norm_array[j,:])
#plt.plot(weight_norm_array)
plt.subplot(121)
plt.legend(('0.01','0.1','1.0'))
plt.xlabel('Iterations')
plt.ylabel('Classification Accuracy %')
plt.subplot(122)
plt.legend(('0.01','0.1','1.0'))
plt.xlabel('Iterations')
plt.ylabel('L2 norm of weight vector')
plt.yscale('log')
plt.show()

#
# plt.figure()
# plt.plot(weight_norm_array)
# plt.show()

#------------------------------------------------------
# #Load CIFAR-10 Data Set--------------------------------
# def unpickle( file ):
#     import cPickle
#     fo = open(file, 'rb')
#     dict = cPickle.load(fo)
#     fo.close()
#     return dict
# #------------------------------------------------------
# data = unpickle( 'cifar-10-batches-py/data_batch_1' )

# features = data['data']
# labels = data['labels']
# labels = np.atleast_2d( labels ).T

# # squash classes 0-4 into class 0, and squash classes 5-9 into class 1
# labels[ labels < 5 ] = 0
# labels[ labels >= 5 ] = 1