import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage import transform
import random
from pdb import set_trace as stop

#------------------------------------------FUNCTIONS----------------------------------------------------------
# Convolution Function
def conv( x, filter_size=5, stride=2, num_filters=64, is_output=False, name="conv" ):
        x_shape = x.get_shape().as_list() #returns a list of the shape of the input tensor
        fshape = [filter_size, filter_size, x_shape[3], num_filters]
        finit = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope('weights_' + name):
            w_filter = tf.get_variable('W', shape=fshape, initializer=finit)
            bias = tf.get_variable('b', shape=[num_filters], initializer=finit)
        with tf.name_scope(name):
            conv_bias = tf.nn.bias_add(tf.nn.conv2d(x, w_filter, strides = [1, stride, stride, 1], padding="SAME"), bias)
            # conv_bias = tf.nn.bias_add(conv, bias)
            if not is_output:
                return tf.nn.relu(conv_bias) #I should use tf.nn.leaky_relu here, what function is this??
            else:
                return conv_bias

# Up Convolution Function
def upconv(x,filter_size=5, stride = 2, num_filters = 64, name="upconv", is_output=False):
    x_shape = x.get_shape().as_list() #returns a list of the shape of the input tensor
    w_shape = [filter_size, filter_size, num_filters, x_shape[3]]
    output_shape = tf.stack([tf.shape(x)[0], stride*x_shape[1], stride*x_shape[2], num_filters])
    finit = tf.contrib.layers.variance_scaling_initializer()
    # output_shape = [x_shape[0], stride*x_shape[1], stride*x_shape[2], output_channels]
    with tf.variable_scope('weights_'+name):
        w_filter = tf.get_variable('W', shape = w_shape, initializer=finit)
    strides_ = [1, stride, stride, 1]
    with tf.name_scope(name):
        if not is_output:
            x = tf.contrib.layers.layer_norm(x)
        out = tf.nn.conv2d_transpose(x,w_filter, output_shape, strides = strides_, padding='SAME')
        out = tf.reshape(out,output_shape)
        if not is_output:
            out = tf.nn.relu(out)
        else:
            out = tf.nn.tanh(out)
        return out

# Fully Connected Function
def fc(x, out_size, name="fc"):
        x_shape = x.get_shape().as_list()
        x_len = x_shape[1] * x_shape[2] * x_shape[3]
        finit = tf.contrib.layers.variance_scaling_initializer()
        with tf.variable_scope('weights_' + name):
            W = tf.get_variable('W',shape=[x_len, out_size], initializer=finit)
            b = tf.get_variable('b',shape=[out_size], initializer=finit)
        with tf.name_scope(name):
            # y = tf.nn.bias_add(tf.matmul(x,W),b)
            y = tf.matmul(tf.reshape(x,[-1,x_len]),W) + b
        return y

# Project and Reshape Function (transforms from z space)
def proj_resh(z, out_shape = [-1, 4, 4, 1024], name = 'proj_resh'):
    num_entries = out_shape[1] * out_shape[2] * out_shape[3]
    z_shape = z.get_shape().as_list()
    w_shape = [z_shape[1],num_entries]
    finit = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope('weights_' + name):
        W = tf.get_variable('W',shape=w_shape, initializer=finit)
        b = tf.get_variable('b',shape=[num_entries], initializer=finit)
    with tf.name_scope(name):
        out = tf.reshape(tf.matmul(z,W) + b, out_shape)
        out = tf.nn.relu(out)
    return out

# Generator Function
def gen(z,out_shape = [-1,4,4,1024], name = 'generator', reuse = False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope(name):
        l0 = proj_resh(z,out_shape,name='G_proj_resh')
        l1 = upconv(l0,num_filters = out_shape[3]/2, name = 'gen_upconv1')
        l2 = upconv(l1,num_filters = out_shape[3]/4, name = 'gen_upconv2')
        l3 = upconv(l2,num_filters = out_shape[3]/8, name = 'gen_upconv3')
        l4 = upconv(l3,num_filters = 3, name = 'gen_upconv4', is_output = True)
    return l4

# Discriminator Function
def disc(images, num_filters_, reuse = False, name = 'discriminator'):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope(name):
        l0 = conv(images, num_filters = num_filters_, name = 'disc_conv0')
        l1 = conv(l0, num_filters = num_filters_*2, name = 'disc_conv1')
        l2 = conv(l1, num_filters = num_filters_*4, name = 'disc_conv2')
        l3 = conv(l2, num_filters = num_filters_*8, name = 'disc_conv3')
        l4 = fc(l3, out_size = 1, name = 'disc_fc1')
    return l4

# Load Data Batch Function
def get_data_batch(filenames, batch_size, im_size, data_path):
    idx = random.sample(range(len(filenames)), batch_size)
    im_batch = np.zeros((batch_size, im_size, im_size, 3))
    for i in range(batch_size):
        image = skio.imread(data_path+filenames[idx[i]])
        im_batch[i,:,:,:] = transform.resize(image, (im_size, im_size, 3))
    return im_batch

def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0/np.linalg.norm(p0), p1/np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1

#---------------------------------------Set up the GAN--------------------------------------------------------
# Tensorflow parameters
batch_size = 6
num_iterations = 10000
im_size = 64
lamda = 10.0
n_critic = 5
alpha = 0.0001
beta_1 = 0.0
beta_2 = 0.9

# Set up placeholders
z = tf.placeholder(tf.float32, [None, 100], name='z')
x = tf.placeholder(tf.float32, [None,im_size, im_size, 3], name = 'x')
eps = tf.placeholder(tf.float32, [None,1,1,1], name = 'eps')

with tf.variable_scope('GAN'):
    # Make generator
    x_tild = gen(z)

    # Make discriminator
    Dx = disc(x, im_size*2, name = 'disc_x')
    x_hat = eps*x + (1 - eps)*x_tild
    Dx_tild = disc(x_tild, im_size*2, name = 'disc_xtild', reuse = True)
    Dx_hat = disc(x_hat, im_size*2, name = 'disc_xhat', reuse = True)

    # Loss function
    with tf.name_scope('Loss'):
        L = Dx_tild - Dx + lamda*(tf.norm(tf.gradients(Dx_hat, x_hat), 2) - 1)**2

    # Set up optimizers
    vars = tf.trainable_variables()
    Dvars = [v for v in vars if 'disc' in v.name]
    Gvars = [v for v in vars if 'gen' in v.name]
with tf.name_scope('Optimizers'):
    wopt = tf.train.AdamOptimizer(alpha,beta_1,beta_2).minimize(tf.reduce_mean(L),var_list=[Dvars])
    thopt = tf.train.AdamOptimizer(alpha,beta_1,beta_2).minimize(tf.reduce_mean(-Dx_tild),var_list=[Gvars])

#----------------------------------------Run the GAN----------------------------------------------------------
# Get filenames and data path
data_path = '/home/kraudust/git/personal_git/byu_classes/deep_learning/lab7_wasserstein_GAN/img_align_celeba/'
filenames = os.listdir(data_path)
sess = tf.Session()
saver = tf.train.Saver()
load_prev_weights = True
if load_prev_weights:
    saver.restore(sess, "/home/kraudust/git/personal_git/byu_classes/deep_learning/lab7_wasserstein_GAN/train2/ckpt/GAN.ckpt")
else:
    init = tf.global_variables_initializer()
    sess.run(init)

# Summaries
writer = tf.summary.FileWriter("./tf_logs",sess.graph)
gen_im = tf.summary.image('generated_image',x_tild,max_outputs = batch_size)
real_im = tf.summary.image('real_image',x,max_outputs = batch_size)
merged = tf.summary.merge_all()

# Train the GAN!!
for i in xrange(num_iterations):
    for t in xrange(n_critic):
        batch = get_data_batch(filenames, batch_size, im_size, data_path)
        zin = np.random.uniform(size=(batch_size,100))
        epsin = np.random.uniform(size = (batch_size,1,1,1))
        disc_opt, ss = sess.run([wopt,merged], feed_dict={z:zin, x:batch, eps:epsin})
    zin = np.random.uniform(size=(batch_size,100))
    gen_opt, ss = sess.run([thopt, merged], feed_dict = {z:zin, x:batch})
    writer.add_summary(ss,i)
    saver.save(sess, "GAN.ckpt")
    print i*n_critic*batch_size, ' images trained on'
    print i, '\n'

# Generate a single image
# batch_size = 2
# zin1 = np.random.uniform(size=(100))
# zin2 = np.random.uniform(size=(100))
# num_steps = 10
# z_array = np.array([slerp(zin1,zin2,t) for t in np.arange(0.0,1.0,1.0/num_steps)])
# im_interp = tf.summary.image('image_interpolation',x_tild,max_outputs = num_steps)
# im_array, write = sess.run([x_tild, im_interp], feed_dict = {z:z_array})
# clip_neg = im_array < 0
# im_array[clip_neg] = 0.0
# fig = plt.figure()
# for i in xrange(num_steps):
#     fig.add_subplot(1,num_steps,i+1)
#     plt.imshow(im_array[i,:,:,:])
#     plt.axis('off')
#     print i
# plt.show()
# writer.add_summary(write)

writer.close()

