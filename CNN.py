'''
Ready to code up some CNN? Well here you go...

We will:
    
    1. Use MNIST data again;
    
    2. Follow CNN structure:
    
        Input --> Convo 1 --> Convo 2 --> Pooling --> Fully Connected --> Output;

    3. Implement Early Stopping 

        A. Every 100 training iterations, evaluate the model on the validation set;
        B. If the model performs better than the best model found so far, save the model to RAM;
        C. If there is no progress for 100 evaluations in a row, terminate training;
        D. After training, store the best model found.

Tuesday, 3/5/2019
'''

import os
import numpy as np
import tensorflow as tf
from Preprocessing import *

# Step 1: Define parameters for the CNN

# Input
height = 165
width = 102
channels = 1
n_inputs = height * width

# Parameters for TWO convolutional layers: 
conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

# Define a pooling layer
pool3_dropout_rate = 0.25
pool3_fmaps = conv2_fmaps

# Define a fully connected layer 
n_fc1 = 64
fc1_dropout_rate = 0

# Output
n_outputs = 7

tf.reset_default_graph()

# Step 2: Set up placeholders for input data
X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
y = tf.placeholder(tf.int32, shape=[None], name="y")
training = tf.placeholder_with_default(False, shape=[], name='training')
    
# Step 3: Set up the two convolutional layers using tf.layers.conv2d
conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name="conv1")
conv1_flat = tf.reshape(conv1, shape=[-1, conv1_fmaps * height * width])
# conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
#                          strides=conv2_stride, padding=conv2_pad,
#                          activation=tf.nn.relu, name="conv2")
# conv2_flat = tf.reshape(conv2, shape=[-1, conv2_fmaps * height * width])

# Step 4: Set up the pooling layer with dropout using tf.nn.max_pool 
#     pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#     pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 14 * 14])
#     pool3_flat_drop = tf.layers.dropout(pool3_flat, pool3_dropout_rate, training=training)

# Step 5: Set up the fully connected layer using tf.layers.dense
fc1 = tf.layers.dense(conv1_flat, n_fc1, activation=tf.nn.relu, name="fc1")
fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

# Step 6: Calculate final output from the output of the fully connected layer
logits = tf.layers.dense(fc1, n_outputs, name="output")
Y_proba = tf.nn.softmax(logits, name="Y_proba")

# Step 5: Define the optimizer; taking as input (learning_rate) and (loss)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)

# Step 6: Define the evaluation metric
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Step 7: Initiate    
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Step 8: Read in data
(X_train, y_train), (X_test, y_test) = (data[:5500],y_raw[:5500]), (data[5000:],y_raw[5000:])
X_train = X_train.astype(np.float32).reshape(-1, height*width)
X_test = X_test.astype(np.float32).reshape(-1, height*width)
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[4500:], X_train[:4500]
y_valid, y_train = y_train[4500:], y_train[:4500]

# Step 9: Define some necessary functions
def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

# Step 10: Define training and evaluation parameters
n_epochs = 3
batch_size = 50
iteration = 0

best_loss_val = np.infty
check_interval = 30
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None 
batch_time = list()

# Step 11: Train and evaluate CNN with Early Stopping procedure defined at the very top
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        epoch_time = time.time()
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            print(iteration)
            iteration += 1
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
            batch_time.append(time.time()-start_time)
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print("Epoch {}, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}, time:{:.2f}".format(epoch, acc_val * 100, best_loss_val, time.time()-epoch_time))
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break
    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final accuracy on test set:", acc_test)
    save_path = saver.save(sess, "./my_mnist_model")