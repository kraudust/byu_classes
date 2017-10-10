import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage import transform
import random
from pdb import set_trace as stop
#----------------------------------CONVOLUTION FUNCTION-------------------------------------------------------
def conv( x, filter_size=3, stride=2, num_filters=64, is_output=False, name="conv" ):
        '''
        x is an input tensor (4 dimensional)
        Declare a name scope using the "name" parameter
        Within that scope:
            Create a W filter variable with the proper size
            Create a B bias variable with the proper size
            Convolve x with W by calling the tf.nn.conv2d function
            Add the bias
            If is_output is False,
                Call the tf.nn.relu function
            Return the final op
        '''
        x_shape = x.get_shape().as_list() #returns a list of the shape of the input tensor
        with tf.name_scope(name):
            w_filter = tf.get_variable(name+"_filter",shape=[filter_size,filter_size,x_shape[3],num_filters], initializer=tf.contrib.layers.variance_scaling_initializer())
            # w_filter = tf.Variable(tf.random_normal([filter_size,filter_size,x_shape[3],num_filters]),name=name+"_filter")
            # bias = tf.get_variable(name+"_bias",shape=[num_filters], initializer=tf.contrib.layers.variance_scaling_initializer())
            bias = tf.Variable(tf.random_normal([num_filters]),name=name+"_bias")
            conv_bias = tf.nn.bias_add(tf.nn.conv2d(x, w_filter, strides = [1, stride, stride, 1], padding="SAME"), bias)
            if not is_output:
                return tf.nn.relu(conv_bias)
            else:
                return conv_bias
#---------------------------------Fully Connected Function ---------------------------------------------------
def fc(x, out_size=50, is_output=False, name="fc"):
        '''
        x is an input tensor
        Declare a name scope using the "name" parameter
        Within that scope:
            Create a W filter variable with the proper size
            Create a B bias variable with the proper size
            Multiply x by W and add b
            If is_output is False,
                Call the tf.nn.relu function
            Return the final op
        '''
        x_shape = x.get_shape().as_list()
        with tf.name_scope(name):
            W = tf.get_variable(name+"_weights",shape=[x_shape[1], out_size], initializer=tf.contrib.layers.variance_scaling_initializer())
            # W = tf.Variable(tf.random_normal([x_shape[1], out_size]), name = name+"_weights")
            # b = tf.get_variable(name+"_b",shape=[out_size], initializer=tf.contrib.layers.variance_scaling_initializer())
            b = tf.Variable(tf.random_normal([out_size]), name = name+"_b")
            y = tf.nn.bias_add(tf.matmul(x,W),b)
            if not is_output:
                return tf.nn.relu(y)
            else:
                return

#------------------------------------------------Load Data ---------------------------------------------------
pos_train_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/train/pos')
pos_train_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/train/pos')
neg_train_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/train/neg')
neg_train_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/train/neg')
pos_test_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/test/pos')
pos_test_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/test/pos')
neg_test_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/test/neg')
neg_test_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/test/neg')

n = 300 #number of training images to use
train_index = random.sample(range(len(pos_train_filenames)),n)
im_size = 512
train_ims = np.zeros((2*n,im_size,im_size,3)).astype(np.float32)
train_labs = np.zeros((2*n,im_size,im_size,1)).astype(np.float32)
test_ims = np.zeros((175,im_size,im_size,3)).astype(np.float32)
test_labs = np.zeros((175, im_size, im_size, 1)).astype(np.float32)
data_dir = '/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/'

#load training data into matrices
for i in xrange(n):
    im_train_pos = skio.imread(data_dir + 'inputs/train/pos/' +   pos_train_filenames[train_index[i]])
    im_train_neg = skio.imread(data_dir + 'inputs/train/neg/' +   neg_train_filenames[train_index[i]])
    im_lab_pos = skio.imread(data_dir + 'outputs/train/pos/' +   pos_train_labels[train_index[i]])
    im_lab_neg = skio.imread(data_dir + 'outputs/train/neg/' +   neg_train_labels[train_index[i]])
    train_ims[2*i,:,:,:] = transform.resize(im_train_pos,(im_size,im_size,3))
    train_ims[2*i+1,:,:,:] = transform.resize(im_train_neg,(im_size,im_size,3))
    train_labs[2*i,:,:,:] = transform.resize(im_lab_pos, (im_size, im_size, 1))
    train_labs[2*i+1,:,:,:] = transform.resize(im_lab_neg, (im_size, im_size, 1))
    print i

#load test data into matrices
for i in xrange(len(neg_test_filenames)):
    if i < 75:
        test_im_pos = skio.imread(data_dir + 'inputs/test/pos/' + pos_test_filenames[i])
        test_im_neg = skio.imread(data_dir + 'inputs/test/neg/' + neg_test_filenames[i])
        test_im_lab_pos = skio.imread(data_dir + 'outputs/test/pos/' + pos_test_labels[i])
        test_im_lab_neg = skio.imread(data_dir + 'outputs/test/neg/' + neg_test_labels[i])
        test_ims[2*i,:,:,:] = transform.resize(test_im_pos,(im_size,im_size,3))
        test_ims[2*i+1,:,:,:] = transform.resize(test_im_neg,(im_size,im_size,3))
        test_labs[2*i,:,:,:] = transform.resize(test_im_lab_pos, (im_size, im_size, 1))
        test_labs[2*i+1,:,:,:] = transform.resize(test_im_lab_pos, (im_size, im_size, 1))
        print i
    else:
        test_im_neg = skio.imread(data_dir + 'inputs/test/neg/' + neg_test_filenames[i])
        test_im_lab_neg = skio.imread(data_dir + 'outputs/test/neg/' + neg_test_labels[i])
        test_ims[i+75,:,:,:] = transform.resize(test_im_neg,(im_size,im_size,3))
        test_labs[i+75,:,:,:] = transform.resize(test_im_lab_pos, (im_size, im_size, 1))
        print i

# whiten data
train_ims = (train_ims - np.mean(train_ims,0))/(np.std(train_ims,0))
test_ims = (test_ims - np.mean(test_ims,0))/(np.std(test_ims,0))
#-------------------------------------------------------------------------------------------------------------




















stop()
