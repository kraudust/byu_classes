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
        # w_filter = tf.get_variable(name+"_filter",shape=[x_shape[1],x_shape[2],output_channels,x_shape[3]], initializer=tf.contrib.layers.variance_scaling_initializer())
        w_filter = tf.get_variable(name+"_filter",shape=[filter_size,filter_size,output_channels,x_shape[3]], initializer=tf.contrib.layers.variance_scaling_initializer())
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

# number of training images to use
n = 150
# number of test images to use
t = 50
train_index = random.sample(range(len(pos_train_filenames)),n)
test_index = random.sample(range(len(pos_test_filenames)), t)
print len(test_index)
# im_size = 512
# im_size = 256
# im_size = 128
im_size = 64
train_ims = np.zeros((n,im_size,im_size,3)).astype(np.float32)
train_labs = np.zeros((n,im_size,im_size,1)).astype(np.float32)
test_ims = np.zeros((t,im_size,im_size,3)).astype(np.float32)
test_labs = np.zeros((t, im_size, im_size, 1)).astype(np.float32)
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
# pos_test_000072.png
test_im_pos = skio.imread(data_dir + 'inputs/test/pos/pos_test_000072.png')
test_im_lab_pos = skio.imread(data_dir + 'outputs/test/pos/pos_test_000072.png')
test_ims[0,:,:,:] = transform.resize(test_im_pos,(im_size,im_size,3))
test_labs[0,:,:,:] = transform.resize(test_im_lab_pos, (im_size, im_size, 1))

for i in xrange(1, t):
    if i < 75:
        print test_index[i]
        test_im_pos = skio.imread(data_dir + 'inputs/test/pos/' + pos_test_filenames[test_index[i]])
        # test_im_neg = skio.imread(data_dir + 'inputs/test/neg/' + neg_test_filenames[i])
        test_im_lab_pos = skio.imread(data_dir + 'outputs/test/pos/' + pos_test_labels[test_index[i]])
        # test_im_lab_neg = skio.imread(data_dir + 'outputs/test/neg/' + neg_test_labels[i])
        test_ims[i,:,:,:] = transform.resize(test_im_pos,(im_size,im_size,3))
        # test_ims[2*i+1,:,:,:] = transform.resize(test_im_neg,(im_size,im_size,3))
        test_labs[i,:,:,:] = transform.resize(test_im_lab_pos, (im_size, im_size, 1))
        # test_labs[2*i+1,:,:,:] = transform.resize(test_im_lab_neg, (im_size, im_size, 1))
        print i
    # else:
    #     test_im_neg = skio.imread(data_dir + 'inputs/test/neg/' + neg_test_filenames[i])
    #     test_im_lab_neg = skio.imread(data_dir + 'outputs/test/neg/' + neg_test_labels[i])
    #     test_ims[i+75,:,:,:] = transform.resize(test_im_neg,(im_size,im_size,3))
    #     test_labs[i+75,:,:,:] = transform.resize(test_im_lab_pos, (im_size, im_size, 1))
    #     print i

# whiten data
train_ims = (train_ims - np.mean(train_ims,0))/(np.std(train_ims,0))
test_ims = (test_ims - np.mean(test_ims,0))/(np.std(test_ims,0))
print "Finished whitening data..."
#--------------------------------------------Design Neural Net---------------------------------------------
batch_size = 3
input_images = tf.placeholder(tf.float32,[batch_size,im_size,im_size,3],name='image')
label_images = tf.placeholder(tf.int64,[batch_size, im_size, im_size],name = 'label')
#define the neural net
l0 = conv(input_images, name='conv0', num_filters = 64)
l1 = conv(l0, name = 'conv1', num_filters = 64)
# l2 = conv(l1, name = 'conv2', num_filters = 8)
# l3 = conv(l2, name = 'conv3', num_filters = 8)
# score = conv(l3, name = 'conv4', num_filters = 2, is_output=True)
# l2 = tf.nn.max_pool(l1,ksize = [1,2,2,1], strides = [1,2,2,1],padding = 'VALID', name = 'l2')
l2 = max_pool(l1, name = 'pool2')
l3 = conv(l2, name = 'conv3', num_filters = 128)
l4 = conv(l3, name = 'conv4', num_filters = 128)
l5 = upconv(l4, name="upconv5", output_channels = 64)
l6 = tf.concat([l1, l5], 3, name='concat')
l7 = conv(l6, name = 'conv7', num_filters = 64)
l8 = conv(l7,name = 'conv8', num_filters = 64)
score = conv(l8,name = 'conv9', num_filters = 2, is_output = True)
output_image = tf.argmax(score,3)
# score = conv(l9,name = 'conv10', num_filters = 1,filter_size = 2 , is_output = True)

#calculate loss
with tf.name_scope('softmax_cross_entropy_loss'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_images, logits = score))

#calculate accuracy
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(output_image,label_images),tf.float32))

#Optimizer
with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
print "finished building neural net..."

#------------------------------------------------Run Neural Net-----------------------------------------------
train_im = np.zeros((batch_size, im_size, im_size, 3))
test_im = np.zeros((batch_size, im_size, im_size, 3))
train_lab = np.zeros((batch_size, im_size, im_size))
test_lab = np.zeros((batch_size, im_size, im_size))
# train_im = np.reshape(train_ims[0:batch_size,:,:,:],[batch_size,im_size,im_size,3])
# test_im = np.reshape(test_ims[0:batch_size,:,:,:],[batch_size,im_size,im_size,3])
# train_lab = np.reshape(train_labs[0:batch_size,:,:,:],[batch_size,im_size,im_size]).astype(np.int64)
# test_lab = np.reshape(test_labs[0:batch_size,:,:,:],[batch_size,im_size,im_size]).astype(np.int64)

sess = tf.Session()
saver = tf.train.Saver()
writer = tf.summary.FileWriter("./tf_logs",sess.graph)
#uncomment the line below to start from previous weights
# saver.restore(sess, "tmp/epoch_80.ckpt")
#uncomment the 2 lines below to start from scratch
init = tf.global_variables_initializer()
sess.run(init)

pred = tf.summary.image('im_prediction', tf.cast(tf.reshape(output_image, [batch_size, im_size, im_size, 1]), tf.float32), max_outputs = batch_size)
test_pred = tf.summary.image('test_im_prediction', tf.cast(tf.reshape(output_image, [batch_size, im_size, im_size, 1]), tf.float32), max_outputs = batch_size)
lab = tf.summary.image('im_label', tf.cast(tf.reshape(label_images, [batch_size, im_size, im_size, 1]), tf.float32), max_outputs = batch_size)
test_lab_sum = tf.summary.image('test_im_label', tf.cast(tf.reshape(label_images, [batch_size, im_size, im_size, 1]), tf.float32), max_outputs = batch_size)
im = tf.summary.image('im_image', tf.cast(tf.reshape(input_images, [batch_size, im_size, im_size, 3]), tf.float32), max_outputs = batch_size)
test_im_sum = tf.summary.image('test_im_image', tf.cast(tf.reshape(input_images, [batch_size, im_size, im_size, 3]), tf.float32), max_outputs = batch_size)

loss_plot = tf.summary.scalar('im_loss_train', loss)
acc_plot = tf.summary.scalar('im_accuracy_train', accuracy)
acc_plot_test = tf.summary.scalar('im_accuracy_test', accuracy)

print "finished initializing neural net..."
num_steps = 1000
for i in range(num_steps):
    train_index = random.sample(range(n),batch_size)
    test_index = random.sample(range(t), batch_size)
    test_im[0,:,:,:] = test_ims[0,:,:,:]
    test_lab[0,:,:] = tf.reshape(test_labs[0,:,:,:],[im_size, im_size, 1])
    for i in xrange(batch_size):
        train_im[i,:,:,:] = train_ims[train_index[i], :,:,:]
        train_lab[i,:,:] = tf.reshape(train_labs[train_index[i], :,:,:], [im_size, im_size,1])
        if i > 0:
            test_im[i,:,:,:] = test_ims[train_index[i], :,:,:]
            test_lab[i,:,:] = tf.reshape(test_labs[test_index[i], :,:,:], [im_size, im_size,1])

    # test_index = random.sample(range(len(pos_test_filenames)), 1)
    train_im = np.reshape(train_ims[0:batch_size,:,:,:],[batch_size,im_size,im_size,3])
    test_im = np.reshape(test_ims[0:batch_size,:,:,:],[batch_size,im_size,im_size,3])
    train_lab = np.reshape(train_labs[0:batch_size,:,:,:],[batch_size,im_size,im_size]).astype(np.int64)
    test_lab = np.reshape(test_labs[0:batch_size,:,:,:],[batch_size,im_size,im_size]).astype(np.int64)

    sess.run(train_step, feed_dict = {input_images: train_im, label_images: train_lab})
    loss_plot_ = sess.run(loss_plot, feed_dict = {input_images: train_im, label_images: train_lab})
    acc_plot_ = sess.run(acc_plot, feed_dict = {input_images: train_im, label_images: train_lab})
    acc_plot_test_ = sess.run(acc_plot_test, feed_dict = {input_images: test_im, label_images: test_lab})
    writer.add_summary(loss_plot_,i)
    writer.add_summary(acc_plot_,i)
    writer.add_summary(acc_plot_test_,i)
    print sess.run(accuracy, feed_dict = {input_images: train_im, label_images: train_lab})
    print sess.run(loss, feed_dict = {input_images: train_im, label_images: train_lab})
    print i, '\n'

    if i % 100.0 == 0.0:
        save_path = saver.save(sess, "tmp/epoch_"+str(i)+".ckpt")
        print("Model saved in file: %s" % save_path)
        images1 = sess.run(test_pred,  feed_dict = {input_images: test_im, label_images: test_lab})
        # images = sess.run([pred, lab, im],  feed_dict = {input_images: train_im, label_images: train_lab})
        writer.add_summary(images1,i)
    if i == num_steps-1:
        images2 = sess.run(test_lab_sum,  feed_dict = {input_images: test_im, label_images: test_lab})
        images3 = sess.run(test_im_sum,  feed_dict = {input_images: test_im, label_images: test_lab})
        # images2_t = sess.run(test_lab,  feed_dict = {input_images: test_im[0,:,:,:], label_images: test_lab})
        # images3_t = sess.run(test_im,  feed_dict = {input_images: test_im[0,:,:,:], label_images: test_lab})
        writer.add_summary(images2)
        writer.add_summary(images3)
        # writer.add_summary(images2_t)
        # writer.add_summary(images3_t)

writer.close()

