
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 784] )
y_true = tf.placeholder(tf.float32, [None, 10] )

NN1 = 200
NN2 = 100

with tf.name_scope( "first_layer" ) as scope:
    W1 = tf.Variable( 1e-3*np.random.randn( 784, NN1 ).astype(np.float32), name="W1" )
    b1 = tf.Variable( 1e-3*np.random.randn( NN1 ).astype(np.float32), name="b1"  )
    h0 = tf.nn.relu( tf.matmul(x, W1) + b1 )

with tf.name_scope( "second_layer" ) as scope:
    W2 = tf.Variable( 1e-3*np.random.randn( NN1, NN2 ).astype(np.float32), name="W2"  )
    b2 = tf.Variable( 1e-3*np.random.randn( NN2 ).astype(np.float32), name="b2"  )
    h1 = tf.nn.relu( tf.matmul(h0, W2) + b2 )

with tf.name_scope( "third_layer" ) as scope:
    W3 = tf.Variable( 1e-3*np.random.randn( NN2, 10 ).astype(np.float32), name="W3"  )
    b3 = tf.Variable( 1e-3*np.random.randn( 10 ).astype(np.float32), name="b3"  )
    h2 = tf.matmul(h1, W3) + b3

y_hat = tf.nn.softmax( h2 )

# =============================

with tf.name_scope( "loss_function" ) as scope:
    xent = -tf.reduce_sum( y_true * tf.log(y_hat), reduction_indices=[1] )
    cross_entropy = tf.reduce_mean( xent )

with tf.name_scope( "accuracy" ) as scope:
    correct_prediction = tf.equal( tf.argmax(y_true,1), tf.argmax(y_hat,1) )
    accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

# =============================

#train_step = tf.train.GradientDescentOptimizer( 0.1 ).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer( 0.001 ).minimize( cross_entropy )

#saver = tf.train.Saver()

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run( init )

train_writer = tf.summary.FileWriter( "./tf_logs", sess.graph )

tf.summary.scalar('cross_entropy', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()

for i in range( 100 ):
    batch_xs, batch_ys = mnist.train.next_batch( 100 )
    sess.run( train_step, feed_dict={x: batch_xs, y_true: batch_ys} )
    acc, ss = sess.run( [accuracy, merged], feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
    train_writer.add_summary( ss, i )
    print( "%d %.2f" % ( i, acc ) )

#saver.save( sess, './tf_logs/model.ckpt' )
train_writer.close()
