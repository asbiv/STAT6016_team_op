import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import sys
sys.path.append('..')
import time
import numpy as np

import tensorflow as tf

#READ IN LOCAL DATA
data = np.load('/Users/Ab/Desktop/SYS6016_Local/team_op/phase_2/data/preprocessed_X.npy')
y_raw = np.load('/Users/Ab/Desktop/SYS6016_Local/team_op/phase_2/data/preprocessed_y.npy')


# reset_graph()

#TOP PARAMETERS
height = 165
width = 102
channels = 1
n_inputs = height * width

#EXISTING PARAMETERS
#n_steps = 28
#n_inputs = 28
n_neurons = 150
n_outputs = 7 #Changed for TOP data

#EXISTING
learning_rate = 0.001

#X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#TOP X PLACEHOLDER
X = tf.placeholder(tf.float32, [None, width, height])
y = tf.placeholder(tf.int32, [None])

#These four lines are the bits that change for the RNN
basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)

loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

#PREP DATA FOR CONSUMPTION
'''
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
'''
(X_train, y_train), (X_test, y_test) = (data[:5500],y_raw[:5500]), (data[5000:],y_raw[5000:])
X_train = X_train.astype(np.float32).reshape(-1, height*width)
X_test = X_test.astype(np.float32).reshape(-1, height*width)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[4500:], X_train[:4500]
y_valid, y_train = y_train[4500:], y_train[:4500]


#EXISTING OPS
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

##IN EXISTING: Converts (10000, 784) --> (10000, 28, 28)
#X_test = X_test.reshape((-1, n_steps, n_inputs))
##IN TOP: Converts (1521, 16830) --> (1521, 102, 165)
X_test = X_test.reshape((-1, width, height))

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            #X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            #TOP
            X_batch = X_batch.reshape((-1, width, height))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Test accuracy:", acc_test)