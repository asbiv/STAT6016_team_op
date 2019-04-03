#FFNN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.chdir('/Users/Ab/Desktop/SYS6016_Local/team_op/phase_2')

import tensorflow as tf
import numpy      as np
import pandas as pd
import time

input_mat = pd.read_csv('data/input_mat.csv')

#UDF
def make_new_labels(df, num):
    labels = pd.concat([df.iloc[:,0].astype('category').cat.codes.astype(np.int8),
                         df.iloc[:,0]], axis=1)
    label_lookup = labels.drop_duplicates()
    new_labels = np.zeros((num, 7))
    new_labels[np.arange(num), labels[0]] = 1
    return(new_labels, label_lookup)


def df_to_tensor_lists(df, dataset='train'):
    '''Takes train or test data, outputs tf-friendly set'''
    X = df.iloc[:,1:].values.astype(np.float32)
    num = len(X)
    new_labels, label_lookup = make_new_labels(df, num)
    data = (X, new_labels)
    return(data, label_lookup)

#DEFINE PARAMETERS
learning_rate = 0.01
batch_size = 100
n_epochs = 30

#CONVERT TO TENSOR LIST
train, label_lookup = df_to_tensor_lists(input_mat)
train_data_raw = tf.data.Dataset.from_tensor_slices(train)
train_data = train_data_raw.batch(batch_size)

#
iterator = tf.data.Iterator.from_structure(train_data.output_types, 
                                           train_data.output_shapes)
X, label = iterator.get_next()

train_init = iterator.make_initializer(train_data) # initializer for train_data

#Create weights and biases
'''
#TWO LAYERS
w1 = tf.get_variable(name='w1', shape=(140, 90),
                    initializer=tf.random_normal_initializer(0, 0.01))
w2 = tf.get_variable(name='w2', shape=(90, 50),
                    initializer=tf.random_normal_initializer(0, 0.01))
w3 = tf.get_variable(name='w3', shape=(50, 7),
                    initializer=tf.random_normal_initializer(0, 0.01))

#b is the bias, sort of like an intercept term
b1 = tf.get_variable(name='b1', shape=(1, 90), initializer=tf.zeros_initializer())
b2 = tf.get_variable(name='b2', shape=(1, 50), initializer=tf.zeros_initializer())
b3 = tf.get_variable(name='b3', shape=(1, 7), initializer=tf.zeros_initializer())
'''
#THREE LAYER
w1 = tf.get_variable(name='w1', shape=(140, 100),
                    initializer=tf.random_normal_initializer(0, 0.01))
w2 = tf.get_variable(name='w2', shape=(100, 60),
                    initializer=tf.random_normal_initializer(0, 0.01))
w3 = tf.get_variable(name='w3', shape=(60, 20),
                    initializer=tf.random_normal_initializer(0, 0.01))
w4 = tf.get_variable(name='w4', shape=(20, 7),
                    initializer=tf.random_normal_initializer(0, 0.01))

#b is the bias, sort of like an intercept term
b1 = tf.get_variable(name='b1', shape=(1, 100), initializer=tf.zeros_initializer())
b2 = tf.get_variable(name='b2', shape=(1, 60), initializer=tf.zeros_initializer())
b3 = tf.get_variable(name='b3', shape=(1, 20), initializer=tf.zeros_initializer())
b4 = tf.get_variable(name='b4', shape=(1, 7), initializer=tf.zeros_initializer())
'''
#TWO LAYER
hidden1 = tf.nn.relu(tf.matmul(X, w1) + b1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)
logits = tf.matmul(hidden2, w3) + b3

'''
#THREE LAYER
hidden1 = tf.nn.relu(tf.matmul(X, w1) + b1)
hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)
hidden3 = tf.nn.relu(tf.matmul(hidden2, w3) + b3)
logits = tf.matmul(hidden3, w4) + b4


# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='entropy')
loss = tf.reduce_mean(entropy, name='loss')


# Step 6: define optimizer
# using Adamn Optimizer with pre-defined learning rate to minimize loss
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)


# Step 7: calculate accuracy with training set
preds = tf.nn.softmax(logits)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))


#Summary terms
ent_summary = tf.summary.scalar('entropy_line', tf.reduce_mean(entropy))
loss_summary = tf.summary.scalar('loss_line', tf.reduce_mean(loss))
acc_summary = tf.summary.scalar('accuracy_line', tf.reduce_mean(accuracy))
#acc_mean_summary = tf.summary.scalar('mean_accuracy_line', mean_accuracy)

with tf.Session() as sess:
    #Init writer
    writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

    start_time = time.time()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver() #Initialize model saver

    for i in range(n_epochs): #Train model by number of epochs
        sess.run(train_init)	# drawing samples from train_data
        #LOCAL MEASURES
        total_loss = 0
        n_batches = 0
        accuracy_sum = 0
        
        #ADD TO SUMMARY
        summary_train_ent = sess.run(ent_summary)
        writer.add_summary(summary_train_ent, i)
        summary_train_loss = sess.run(loss_summary)
        writer.add_summary(summary_train_loss, i)
        summary_train_acc = sess.run(acc_summary)
        writer.add_summary(summary_train_acc, i)
        
        try:
            while True:
                _, l = sess.run([optimizer, loss])
                total_loss += l
                n_batches += 1
                #Accuracy by batch
                accuracy_batch = sess.run(accuracy/batch_size)                
                accuracy_sum += accuracy_batch
                
        except tf.errors.OutOfRangeError:
            pass
        
        #PRINTING
        print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))
        #Mean accuracy
        mean_accuracy = accuracy_sum/n_batches
        print('Average accuracy epoch {0}: {1}'.format(i, mean_accuracy))
    
    #save_path = saver.save(sess, "/Users/Ab/Desktop/HW2/models/model.ckpt")
    #print ("Model saved in file: ", save_path)
    print('Total time: {0} seconds'.format(time.time() - start_time))
    
writer.close()