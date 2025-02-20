import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
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
            w_filter = tf.get_variable(name+"_filter",shape=[filter_size,filter_size,x_shape[3],num_filters],dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            # bias = tf.get_variable(name+"_bias",shape=[num_filters],dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
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
            W = tf.get_variable(name+"_weights",shape=[out_size,x_shape[0]], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            # b = tf.get_variable(name+"_b",shape=[out_size,1], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
            b = tf.Variable(tf.random_normal([out_size, 1]), name = name+"_b")
            y = tf.add(tf.matmul(W,x),b,name='y')
            if not is_output:
                return tf.nn.relu(y,name='relu')
            else:
                return y
#---------------------------------------Reshape Function------------------------------------------------------
def reshape_cifar_image(im):
    im_new = np.zeros((1,32,32,3))
    l = 0
    for k in range(3):
        for j in range(32):
            for i in range(32):
                im_new[0,j,i,k] = im[l]
                l = l + 1
    return im_new

#-------------------------------------Load CIFAR-10 Data Set--------------------------------------------------
def unpickle( file ):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
data1 = unpickle( 'cifar-10-batches-py/data_batch_1' )
data2 = unpickle( 'cifar-10-batches-py/data_batch_2' )
data3 = unpickle( 'cifar-10-batches-py/data_batch_3' )
data4 = unpickle( 'cifar-10-batches-py/data_batch_4' )
data5 = unpickle( 'cifar-10-batches-py/data_batch_5' )
data_test = unpickle( 'cifar-10-batches-py/test_batch' )
features1 = data1['data'] #size is #samples(10000) x #features(3072)
features2 = data2['data'] #size is #samples(10000) x #features(3072)
features3 = data3['data'] #size is #samples(10000) x #features(3072)
features4 = data4['data'] #size is #samples(10000) x #features(3072)
features5 = data5['data'] #size is #samples(10000) x #features(3072)
features_test = data_test['data'] #size is #samples(10000) x #features(3072)
labels1 = data1['labels']
labels2 = data2['labels']
labels3 = data3['labels']
labels4 = data4['labels']
labels5 = data5['labels']
labels_test = data_test['labels']
# pdb.set_trace()
# labels1 = np.atleast_2d(labels1).T #size is number of samples x 1
# labels2 = np.atleast_2d(labels2).T #size is number of samples x 1
# labels3 = np.atleast_2d(labels3).T #size is number of samples x 1
# labels4 = np.atleast_2d(labels4).T #size is number of samples x 1
# labels5 = np.atleast_2d(labels5).T #size is number of samples x 1
# labels_test = np.atleast_2d(labels_test).T #size is number of samples x 1
features_train = np.concatenate((features1, features2, features3, features4, features5),axis=0)
labels_train = np.concatenate((labels1, labels2, labels3, labels4, labels5), axis=0)

#-----------------------------------Build Neural Net ---------------------------------------------------------
#define placeholder for data and labels
input_image = tf.placeholder(tf.float32,shape=[1,32,32,3],name='image')
lab = tf.placeholder(tf.int64,[1], name = 'label')

#define the neural net which outputs scores
h0 = conv(input_image, num_filters=128, name='conv1')
h1 = conv(h0,filter_size = 5, name='conv2')
h2 = conv(h1, filter_size=4,name='conv3')
h3 = conv(h2,name='conv4',stride=1)
h4 = conv(h3,name='conv5',num_filters=32)
h4_shape = h4.get_shape().as_list()
h4_flat = tf.reshape(h4, [h4_shape[1]*h4_shape[2]*h4_shape[3], 1])
fc0 = fc(h4_flat,out_size = 256,name='fc1')
fc1 = fc(fc0, out_size =128, name='fc2')
score = fc(fc1, out_size = 10,is_output = True, name = 'fc3')

#determine loss
with tf.name_scope('softmax_cross_entropy_loss'):
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = lab,logits = score[:,0])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = lab,logits = tf.transpose(score))

#to calculate accuracy
with tf.name_scope('accuracy'):
    classification_correct = tf.equal(tf.argmax(score,0), lab)
    # acc = tf.equal(classification, lab)  #returns a 1 if we classified the image correctly or a 0 if we didn't
    acc = tf.cast(classification_correct,tf.float32)

#Optimizer
with tf.name_scope('optimizer'):
    train_weights = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
#-------------------------------------------------------------------------------------------------------------

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
writer = tf.summary.FileWriter("./tf_logs",sess.graph)
tf.summary.scalar('loss', loss[0])
merged = tf.summary.merge_all()
accuracy_array = np.array([])
i_array = np.array([])
# for i in range(len(labels_train)):
for i in range(20000):
    im = reshape_cifar_image(features_train[i,:])
    # im = np.reshape(features_train[i,:], [-1, 32, 32, 3])
    label_ = [labels_train[i]]
    # pdb.set_trace()
    sess.run(train_weights, feed_dict = {input_image:im, lab:label_})
    if i % 100.0 == 0: #check accuracy every 100 training images
        [loss_, ss] =  sess.run([loss, merged],feed_dict = {input_image:im, lab:label_})
        writer.add_summary(ss,i)
        acc_sum = 0.0
        test_indices = np.random.choice(len(labels_test),size=200)
        for j in range(200): #test accuracy on 200 random images from the test set
            acc_i = sess.run(acc, feed_dict = {input_image: reshape_cifar_image(features_test[test_indices[j],:]), lab: [labels_test[test_indices[j]]]})
            # acc_i = sess.run(acc, feed_dict = {input_image: np.reshape(features_test[test_indices[j],:],[-1,32,32,3]), lab: [labels_test[test_indices[j]]]})
            acc_sum = acc_sum + acc_i
        accuracy = (acc_sum/200.0)*100.0
        print 'Loss: ', loss_
        print 'Accuracy: ', accuracy
        print 'i: ', i, 'out of ', len(labels_train)
        print '\n'
        accuracy_array = np.append(accuracy_array,accuracy)
        i_array = np.append(i_array, i)
writer.close()

plt.figure()
plt.plot(i_array, accuracy_array)
plt.xlabel('Train_step')
plt.ylabel('Classification Accuracy %')
plt.title('Accuracy')
plt.show()

