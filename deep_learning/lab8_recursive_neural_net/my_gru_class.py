from tensorflow.python.ops.rnn_cell import RNNCell
import tensorflow as tf

class mygru( RNNCell ):
    def __init__( self, state_dim):
        self.state_dim = state_dim
        self.scope = None
        
 
    @property
    def state_size(self):
        return self.state_dim 

    @property
    def output_size(self):
        return self.state_dim
 
    def __call__( self, inputs, state):
        input_shape = inputs.get_shape().as_list()
        with tf.variable_scope('gru') as scope:
            if self.scope == None:
                wx_shape = [input_shape[1], self.state_dim]
                wh_shape = [self.state_dim, self.state_dim]
                b_shape = [self.state_dim]
                self.Wxr = tf.get_variable('wxr', shape = wx_shape, initializer = tf.contrib.layers.variance_scaling_initializer())
                self.Wxz = tf.get_variable('wxz', shape = wx_shape, initializer = tf.contrib.layers.variance_scaling_initializer())
                self.Wxh = tf.get_variable('wxh', shape = wx_shape, initializer = tf.contrib.layers.variance_scaling_initializer())
                self.Whr = tf.get_variable('whr', shape = wh_shape, initializer = tf.contrib.layers.variance_scaling_initializer())
                self.Whz = tf.get_variable('whz', shape = wh_shape, initializer = tf.contrib.layers.variance_scaling_initializer())
                self.Whh = tf.get_variable('whh', shape = wh_shape, initializer = tf.contrib.layers.variance_scaling_initializer())
                self.br = tf.get_variable('br', shape = b_shape, initializer = tf.contrib.layers.variance_scaling_initializer())
                self.bz = tf.get_variable('bz', shape = b_shape, initializer = tf.contrib.layers.variance_scaling_initializer())
                self.bh = tf.get_variable('bh', shape = b_shape, initializer = tf.contrib.layers.variance_scaling_initializer())
                self.scope = 'gru' 
            else:
                scope.reuse_variables()
                self.Wxr = tf.get_variable('wxr')
                self.Wxz = tf.get_variable('wxz')
                self.Wxh = tf.get_variable('wxh')
                self.Whr = tf.get_variable('whr')
                self.Whz = tf.get_variable('whz')
                self.Whh = tf.get_variable('whh')
                self.br = tf.get_variable('br')
                self.bz = tf.get_variable('bz')
                self.bh = tf.get_variable('bh')
            r = tf.nn.sigmoid(tf.matmul(inputs, self.Wxr) + tf.matmul(state, self.Whr) + self.br)
            z = tf.nn.sigmoid(tf.matmul(inputs, self.Wxz) + tf.matmul(state, self.Whz) + self.bz)
            htild = tf.nn.tanh(tf.matmul(inputs, self.Wxh) + tf.matmul(tf.multiply(r,state), self.Whh) + self.bh)
            h = tf.multiply(z,state) + tf.multiply((1-z), htild)

        return h,h

