import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage import transform
import random
from pdb import set_trace as stop
import pickle
#----------------------------------CONVOLUTION FUNCTION-------------------------------------------------------
def conv( x, filter_size=3, stride=1, num_filters=64, is_output=False, name="conv" ):
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

#-----------------------------------------------Max Pool -----------------------------------------------------
def max_pool(x, k_size = 2, stride = 2, name = "max_pool"):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x,ksize = [1,k_size,k_size,1], strides = [1,stride,stride,1],padding = 'VALID')
        return out

#-----------------------------------------------Up-Convolution Function---------------------------------------
def upconv(x,filter_size=2, stride = 2, output_channels =64, name="upconv"):
    x_shape = x.get_shape().as_list() #returns a list of the shape of the input tensor
    # batch_size = tf.shape(x)[0]
    output_shape = [x_shape[0], stride*x_shape[1], stride*x_shape[2], output_channels]
    strides = [1, stride, stride, 1]
    with tf.name_scope(name):
        w_filter = tf.get_variable(name+"_filter",shape=[x_shape[1],x_shape[2],output_channels,x_shape[3]], initializer=tf.contrib.layers.variance_scaling_initializer())
        out = tf.nn.conv2d_transpose(x,w_filter, output_shape, strides, padding='SAME')
        return out

#------------------------------------------------Load Data ---------------------------------------------------
pos_train_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/train/pos')
pos_train_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/train/pos')
neg_train_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/train/neg')
neg_train_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/train/neg')
pos_test_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/test/pos')
pos_test_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/test/pos')
neg_test_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/test/neg')
neg_test_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/test/neg')

# n = 300 #n*2 = number of training images to use
n = 1
train_index = random.sample(range(len(pos_train_filenames)),n)
# im_size = 512
im_size = 256
train_ims = np.zeros((2*n,im_size,im_size,3)).astype(np.float32)
train_labs = np.zeros((2*n,im_size,im_size,1)).astype(np.float32)
test_ims = np.zeros((175,im_size,im_size,3)).astype(np.float32)
test_labs = np.zeros((175, im_size, im_size, 1)).astype(np.float32)
data_dir = '/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/'
# train_ims[:,:,:,:] = pickle.load(open('whitened_training_images','rb'))
#load training data into matrices
for i in xrange(n):
    im_train_pos = skio.imread(data_dir + 'inputs/train/pos/' +   pos_train_filenames[train_index[i]])
    # im_train_neg = skio.imread(data_dir + 'inputs/train/neg/' +   neg_train_filenames[train_index[i]])
    im_lab_pos = skio.imread(data_dir + 'outputs/train/pos/' +   pos_train_labels[train_index[i]])
    # im_lab_neg = skio.imread(data_dir + 'outputs/train/neg/' +   neg_train_labels[train_index[i]])
    # train_ims[2*i,:,:,:] = transform.resize(im_train_pos,(im_size,im_size,3))
    train_ims[i,:,:,:] = transform.resize(im_train_pos,(im_size, im_size,3))
    im_train_pos = transform.resize(im_train_pos, (im_size, im_size, 3))
    # train_ims[2*i+1,:,:,:] = transform.resize(im_train_neg,(im_size,im_size,3))
    # train_labs[2*i,:,:,:] = transform.resize(im_lab_pos, (im_size, im_size, 1))
    train_labs[i,:,:,:] = transform.resize(im_lab_pos, (im_size, im_size, 1))
    # train_labs[2*i+1,:,:,:] = transform.resize(im_lab_neg, (im_size, im_size, 1))
    print i

#load test data into matrices
# for i in xrange(len(neg_test_filenames)):
for i in xrange(n):
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
print "Finished whitening data..."
#--------------------------------------------Design Neural Net---------------------------------------------
batch_size = 1
input_images = tf.placeholder(tf.float32,[batch_size,im_size,im_size,3],name='image')
label_images = tf.placeholder(tf.int64,[batch_size, im_size, im_size],name = 'label')
#define the neural net
l0 = conv(input_images, name='conv0', num_filters = 8)
# l1 = conv(l0, name = 'conv1', num_filters = 8)
score = conv(l0, name = 'conv1', num_filters = 2, is_output=True)
# l2 = tf.nn.max_pool(l1,ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'VALID', name = 'l2')
# l2 = max_pool(l1, name = 'pool2')
# l3 = conv(l2, name = 'conv3', num_filters = 16)
# l4 = conv(l3, name = 'conv4', num_filters = 16)
# l5 = upconv(l4, name="upconv5", output_channels = 8)
# l6 = tf.concat([l1, l5], 3, name='concat')
# l7 = conv(l6, name = 'conv6', num_filters = 8)
# # l8 = conv(l7,name = 'conv8', num_filters = 8)
# score = conv(l7,name = 'conv9', num_filters = 2, is_output = True)
output_image = tf.argmax(score,3)
# score = conv(l9,name = 'conv10', num_filters = 1,filter_size = 2 , is_output = True)

#calculate loss
with tf.name_scope('softmax_cross_entropy_loss'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_images, logits = score))

#calculate accuracy
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(score,3),label_images),tf.float32))

#Optimizer
with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
print "finished building neural net..."
#------------------------------------------------Run Neural Net--------------------------------------------
# train_im = np.reshape(train_ims[0:batch_size,:,:,:],[batch_size,im_size,im_size,3])
train_im = np.reshape(im_train_pos,[batch_size,im_size,im_size,3])
train_lab = np.reshape(train_labs[0:batch_size,:,:,:],[batch_size,im_size,im_size]).astype(np.int64)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
writer = tf.summary.FileWriter("./tf_logs",sess.graph)
tf.summary.image('im_prediction', tf.cast(tf.reshape(output_image, [batch_size, im_size, im_size, 1]), tf.float32))
tf.summary.image('im_label', tf.cast(tf.reshape(train_lab, [batch_size, im_size, im_size, 1]), tf.float32))
tf.summary.image('im_image', tf.cast(train_im, tf.float32))
merged = tf.summary.merge_all()

print "finished initializing neural net..."
# for i in range(1):
    # print "running train step ", i
score_, ss = sess.run([score, merged], feed_dict = {input_images: train_im, label_images: train_lab})
writer.add_summary(ss)
# print sess.run(accuracy, feed_dict = {input_images: train_im, label_images: train_lab})
    # print "loss ", sess.run(loss, feed_dict = {input_images: train_im, label_images: train_lab})
writer.close()

