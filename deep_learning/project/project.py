import numpy as np
from pdb import set_trace as pause
import tensorflow as tf


# ---------------------------------CONVOLUTION FUNCTION-------------------------------------------------------
def conv(x, filter_size=3, stride=1, num_filters=64, is_output=False, name="conv"):
        x_shape = x.get_shape().as_list()  # returns a list of the shape of the input tensor
        with tf.name_scope(name):
            w_filter = tf.get_variable(name + "_filter",
                                       shape=[filter_size, filter_size, x_shape[3], num_filters],
                                       initializer=tf.contrib.layers.variance_scaling_initializer())
            bias = tf.Variable(tf.random_normal([num_filters]), name=name+"_bias")
            conv_bias = tf.nn.bias_add(tf.nn.conv2d(x, w_filter,
                                                    strides=[1, stride, stride, 1], padding="SAME"), bias)
            if not is_output:
                return tf.nn.relu(conv_bias)
            else:
                return conv_bias


# --------------------------------Fully Connected Function ---------------------------------------------------
def fc(x, out_size=50, is_output=False, name="fc"):
        x_shape = x.get_shape().as_list()
        with tf.name_scope(name):
            W = tf.get_variable(name+"_weights", shape=[x_shape[1], out_size],
                                initializer=tf.contrib.layers.variance_scaling_initializer())
            b = tf.Variable(tf.random_normal([out_size]), name=name+"_b")
            y = tf.nn.bias_add(tf.matmul(x, W), b)
            if not is_output:
                return tf.nn.relu(y)
            else:
                return y


# -------------------------------------------LOAD DATA--------------------------------------------------------
# puzzles = np.zeros((1000000, 81), np.int32)
# solutions = np.zeros((1000000, 81), np.int32)
data_size = 100  # number of sudoku puzzles to load into RAM
puzzles = np.zeros((data_size, 81), np.int32)
solutions = np.zeros((data_size, 81), np.int32)
for i, line in enumerate(open('sudoku.csv', 'r').read().splitlines()[1:]):
    quiz, solution = line.split(",")
    for j, q_s in enumerate(zip(quiz, solution)):
        q, s = q_s
        puzzles[i, j] = q
        solutions[i, j] = s
    if i == data_size-1:
        break
    print 'loading ', i, 'out of ', data_size
puzzles = puzzles.reshape((-1, 9, 9))
solutions = solutions.reshape((-1, 9, 9))
# --------------------------------------------BUILD NEURAL NET------------------------------------------------
# define placeholder for data and labels
batch_size = 50
puzzle = tf.placeholder(tf.float32, [batch_size, 9, 9, 1], name='puz')
# solution = tf.placeholder(tf.float32, [batch_size, 9, 9, 1], name='sol')
solution = tf.placeholder(tf.int32, [batch_size, 9, 9], name='sol')

# define structure of neural net
h0 = conv(puzzle, num_filters=128, name='conv1')
h1 = conv(h0, name='conv2')
h2 = conv(h1, name='conv3')
h3 = conv(h2, name='conv4')
h4 = conv(h3, name='conv5')
# h5 = conv(h4, name='conv6')
# h6 = conv(h5, name='conv7')
# h7 = conv(h6, name='conv8')
# h8 = conv(h7, name='conv9')
# h9 = conv(h8, name='conv10')
# h10 = conv(h9, name='conv11')
# h11 = conv(h10, name='conv12')
# h12 = conv(h11, name='conv13')
# h13 = conv(h12, name='conv14')
# h14 = conv(h13, name='conv15')
score = conv(h4, num_filters=9, is_output=True, name='score')

# istarget = tf.to_float(tf.equal(puzzle, tf.zeros_like(puzzle))) # 0: blanks
# for i in xrange(9):
#     for j in xrange(9):
#         if puzzle[0,i,j,0] != 0:
#             index = puzzle[0,i,j,0]
#             # score[0,i,j,index] = 1000.
#             # score = tf.scatter_update(score,[0, i, j, index], [1000])
#             values = [1000.]
#             shape = [1,9,9,9]
#             delta = tf.SparseTensor(index, values, shape)
#             score = score_raw + tf.sparse_tensor_to_dense(delta)
output_sudoku = tf.argmax(score, 3)


# determine loss
with tf.name_scope('softmax_cross_entropy_loss'):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=solution, logits=score))
    # ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=solution, logits=score)
    # loss = tf.reduce_sum(ce * istarget) / (tf.reduce_sum(istarget))

#calculate accuracy
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(output_sudoku, tf.int32),solution),tf.float32))

# Optimizer
with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

# --------------------------------------------RUN NEURAL NET--------------------------------------------------
sess = tf.Session()
saver = tf.train.Saver()
load_weights = False
if load_weights == False:
    init = tf.global_variables_initializer()
    sess.run(init)
else:
    checkpoint_path = 'tmp/pass_2/epoch_19990.ckpt'
    saver.restore(sess, checkpoint_path)
i = 1

puzzle_easy = np.array([[0,0,3,0,0,5,2,0,4],
              [4,9,0,0,0,0,6,8,7],
              [0,0,0,0,0,0,9,0,5],
              [0,4,2,0,9,6,0,0,0],
              [0,0,9,3,0,8,4,0,0],
              [0,0,0,7,1,0,8,2,0],
              [3,0,8,0,0,0,0,0,0],
              [9,1,7,0,0,0,0,4,8],
              [2,0,4,8,0,0,5,0,0]]).astype(np.int32)
puzzle_easy_soln = np.array([[7,8,3,9,6,5,2,1,4],
              [4,9,5,2,3,1,6,8,7],
              [6,2,1,4,8,7,9,3,5],
              [8,4,2,5,9,6,1,7,3],
              [1,7,9,3,2,8,4,5,6],
              [5,3,6,7,1,4,8,2,9],
              [3,5,8,1,4,9,7,6,2],
              [9,1,7,6,5,2,3,4,8],
              [2,6,4,8,7,3,5,9,1]]).astype(np.int32)
# acc_print = sess.run(accuracy, feed_dict={puzzle: np.reshape(puzzle_easy,
#                                                     [batch_size, 9, 9, 1]).astype(np.float32) - 1,
#                                                     solution: np.reshape(puzzle_easy_soln,
#                                                     [batch_size, 9, 9]).astype(np.int32) - 1})

# acc_print_easy = sess.run(accuracy, feed_dict={puzzle: np.reshape(puzzles[batch_size*i:batch_size*(i+1), :, :],
#                                                     [batch_size, 9, 9, 1]).astype(np.float32) - 1,
#                                                     solution: np.reshape(solutions[batch_size*i:batch_size*(i+1), :, :],
#                                                     [batch_size, 9, 9]).astype(np.int32) - 1})
# print acc_print_easy

# pause()

# Summaries
writer = tf.summary.FileWriter("./tf_logs", sess.graph)
loss_summary = tf.summary.scalar('loss', loss)
acc_summary = tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
num_batches = int(data_size/batch_size)
# for i in xrange(num_batches*batch_size):
# puzzle_easy = np.array([[0,0,3,0,0,5,2,0,4],
#               [4,9,0,0,0,0,6,8,7],
#               [0,0,0,0,0,0,9,0,5],
#               [0,4,2,0,9,6,0,0,0],
#               [0,0,9,3,0,8,4,0,0],
#               [0,0,0,7,1,0,8,2,0],
#               [3,0,8,0,0,0,0,0,0],
#               [9,1,7,0,0,0,0,4,8],
#               [2,0,4,8,0,0,5,0,0]]).astype(np.int32)
# puzzle_easy_soln = np.array([[7,8,3,9,6,5,2,1,4],
#               [4,9,5,2,3,1,6,8,7],
#               [6,2,1,4,8,7,9,3,5],
#               [8,4,2,5,9,6,1,7,3],
#               [1,7,9,3,2,8,4,5,6],
#               [5,3,6,7,1,4,8,2,9],
#               [3,5,8,1,4,9,7,6,2],
#               [9,1,7,6,5,2,3,4,8],
#               [2,6,4,8,7,3,5,9,1]]).astype(np.int32)
for i in xrange(num_batches):
# for i in xrange(10000):
    sess.run(train_step, feed_dict={puzzle: np.reshape(puzzles[batch_size*i:batch_size*(i+1), :, :],
                                                          [batch_size, 9, 9, 1]).astype(np.float32) - 1,
                                                          solution: np.reshape(solutions[batch_size*i:batch_size*(i+1), :, :],
                                                          [batch_size, 9, 9]).astype(np.int32) - 1})
    # sess.run(train_step, feed_dict={puzzle: np.reshape(puzzles[batch_size-1, :, :],
    #                                                       [batch_size, 9, 9, 1]).astype(np.float32) - 1,
    #                                                       solution: np.reshape(solutions[batch_size-1, :, :],
    #                                                       [batch_size, 9, 9]).astype(np.int32) - 1})
    # sess.run(train_step, feed_dict={puzzle: np.reshape(puzzle_easy,
    #                                                       [batch_size, 9, 9, 1]).astype(np.float32) - 1,
    #                                                       solution: np.reshape(puzzle_easy_soln,
    #                                                       [batch_size, 9, 9]).astype(np.int32) - 1})

    ss = sess.run(merged, feed_dict={puzzle: np.reshape(puzzles[batch_size*i:batch_size*(i+1), :, :],
                                                          [batch_size, 9, 9, 1]).astype(np.float32) - 1,
                                                          solution: np.reshape(solutions[batch_size*i:batch_size*(i+1), :, :],
                                                          [batch_size, 9, 9]).astype(np.int32) - 1})
    # ss = sess.run(merged, feed_dict={puzzle: np.reshape(puzzles[batch_size-1, :, :],
    #                                                       [batch_size, 9, 9, 1]).astype(np.float32) - 1,
    #                                                       solution: np.reshape(solutions[batch_size-1, :, :],
    #                                                       [batch_size, 9, 9]).astype(np.int32) - 1})
    # ss = sess.run(merged, feed_dict={puzzle: np.reshape(puzzle_easy,
    #                                                       [batch_size, 9, 9, 1]).astype(np.float32) - 1,
    #                                                       solution: np.reshape(puzzle_easy_soln,
    #                                                       [batch_size, 9, 9]).astype(np.int32) - 1})
    writer.add_summary(ss,i)
    if i % 10 == 0:

        # output = sess.run(output_sudoku, feed_dict={puzzle: np.reshape(puzzle_easy,
        #                                                     [batch_size, 9, 9, 1]).astype(np.float32) - 1,
        #                                                     solution: np.reshape(puzzle_easy_soln,
        #                                                     [batch_size, 9, 9]).astype(np.int32) - 1})
        output = sess.run(output_sudoku, feed_dict={puzzle: np.reshape(puzzles[batch_size*i:batch_size*(i+1), :, :],
                                                            [batch_size, 9, 9, 1]).astype(np.float32) - 1,
                                                            solution: np.reshape(solutions[batch_size*i:batch_size*(i+1), :, :],
                                                            [batch_size, 9, 9]).astype(np.int32) - 1})
        # output = sess.run(output_sudoku, feed_dict={puzzle: np.reshape(puzzles[batch_size-1, :, :],
        #                                                     [batch_size, 9, 9, 1]).astype(np.float32) - 1,
        #                                                     solution: np.reshape(solutions[batch_size-1, :, :],
        #                                                     [batch_size, 9, 9]).astype(np.int32) - 1})
        # acc_print = sess.run(accuracy, feed_dict={puzzle: np.reshape(puzzle_easy,
        #                                                     [batch_size, 9, 9, 1]).astype(np.float32) - 1,
        #                                                     solution: np.reshape(puzzle_easy_soln,
        #                                                     [batch_size, 9, 9]).astype(np.int32) - 1})
        acc_print = sess.run(accuracy, feed_dict={puzzle: np.reshape(puzzles[batch_size*i:batch_size*(i+1), :, :],
                                                            [batch_size, 9, 9, 1]).astype(np.float32) - 1,
                                                            solution: np.reshape(solutions[batch_size*i:batch_size*(i+1), :, :],
                                                            [batch_size, 9, 9]).astype(np.int32) - 1})
        # acc_print = sess.run(accuracy, feed_dict={puzzle: np.reshape(puzzles[batch_size-1, :, :],
        #                                                     [batch_size, 9, 9, 1]).astype(np.float32) - 1,
        #                                                     solution: np.reshape(solutions[batch_size-1, :, :],
        #                                                     [batch_size, 9, 9]).astype(np.int32) - 1})

        print i,'out of ', num_batches, 'accuracy: ',  acc_print
        # print_puzzle = puzzle_easy
        print_puzzle = puzzles[batch_size*i,:,:] - 1
        # print_puzzle = puzzles[batch_size-1,:,:] - 1
        for j in range(9):
            for k in range(9):
                if print_puzzle[j,k] == -1:
                    print_puzzle[j,k] = 9
        # print puzzles[batch_size-1, :, :] - 1
        print print_puzzle
        print output[0,:,:]
        # print puzzle_easy_soln-1
        print solutions[batch_size*i, :,:]
        # print solutions[batch_size-1, :,:] - 1
        save_path = saver.save(sess, "tmp/epoch_"+str(i)+".ckpt")
        print("Model saved in file: %s" % save_path)

writer.close()

