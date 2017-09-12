import numpy as np
import pandas
import matplotlib.pyplot as plt

def calc_classification_accuracy(w, z, t):
    diff = t - z
    num_right = float(sum(num == 0 for num in diff))
    return (num_right/float(np.size(diff)))*100.0

#Load Iris Data Set-----------------------------------
data = pandas.read_csv( 'Fisher.csv' )
m = data.as_matrix()
labels = m[:,0]
labels[ labels==2 ] = 1  #label everything that is class 2 as class 1 so it is binary
labels = np.atleast_2d( labels ).T
features = m[:,1:5]

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

accuracy_array = np.array([])
weight_norm_array = np.array([])

# Initialize weights
num_outputs = 1 #since it's a binary problem (1 or 0)
num_examples, num_features = np.shape(features)
w = np.zeros([num_features + 1, num_outputs]) #the +1 is because I'm including the bias in the weights
print type(w)
x = np.concatenate((features.T,np.ones([1,num_examples])),axis=0) #stack  ones on to multiply the bias by
z = np.sign(np.dot(w.T,x)).T #current perceptron classification output
z = np.where(z < 0, 0, z) #change all the -1 to 0
learning_rates = [0.01, 0.1, 1.0] #learning rate
t = labels
accuracy_array = np.append(accuracy_array, calc_classification_accuracy(w, z, t))

for c in xrange(0,len(learning_rates)):
    for i in range(100):
        del_w = c*np.dot((t-z).T,x.T)
        w = w + del_w.T
        z = np.sign(np.dot(w.T,x)).T #current perceptron classification output
        z = np.where(z < 0, 0, z) #change all the -1 to 0
        accuracy_array = np.append(accuracy_array,calc_classification_accuracy(w, z, t))
        weight_norm_array = np.append(weight_norm_array,np.sqrt(np.dot(w.T,w)))

# plt.figure()
# plt.plot(accuracy_array)
# plt.plot(weight_norm_array)
# plt.show()
#
# plt.figure()
# plt.plot(weight_norm_array)
# plt.show()
