#!/usr/bin/env python
# encoding: utf-8
from tensorflow.python.ops import gen_ctc_ops
from tensorflow.python.ops import ctc_ops
import tensorflow as tf
import numpy as np
import os 

train_path = "/home/lucklady/CNN_audio/timit/seq_train_fft"
test_path = "/home/lucklady/CNN_audio/timit/seq_test_fft"
initial_learning_rate = 0.001
learning_rate = 0.0001
drop_out = 0.7
batch_size = 1
training_iters = 20000
milestone = 0.7
display_step = 2


# Accounting the 0th indice +  space + blank label = 28 characters
n_classes = ord('z') - ord('a') + 1 + 1 + 1

sess = tf.InteractiveSession()

'''
#load data
encode_list = list()
for dirs in os.listdir(train_path):
    encode_list.append(dirs)

encode_dict = dict()
for i, dirs in enumerate(encode_list):
    encode_dict[dirs]  = i
    
#training data,label,len
size = 0
train_data = list()
train_label = list()
train_seq_len = []
num = 1
for d1 in os.listdir(train_path):
    for d2 in os.listdir(os.path.join(train_path,d1)):
        for file in os.listdir(os.path.join(train_path,d1,d2)):
            if(num<=1):
                temp_train_data = np.load(os.path.join(train_path,d1,d2,file))
                temp_label = temp_train_data['arr_1']
                train_seq_len.append(temp_label.shape[0])
                train_data.append(temp_train_data['arr_0'])
                train_label.append(temp_label)
                num = num + 1
            else:break

train_data = np.abs(np.asarray(train_data))
train_seq_len = np.asarray(train_seq_len)
train_label = np.asarray(train_label)

#testing data,label,len
size = 0
test_data = list()
test_label = list()
test_seq_len = []
num = 1
for d1 in os.listdir(test_path):
    for d2 in os.listdir(test_path+"/"+ d1):
        for file in os.listdir(os.path.join(test_path,d1,d2)):
            if num <=1:
                temp_test_data = np.load(os.path.join(test_path,d1,d2,file))
                temp_label = temp_test_data['arr_1']
                test_seq_len.append(temp_label.shape[0])
                test_label.append(temp_label)
                test_data.append(temp_test_data['arr_0'])    
                num = num + 1 
            else:break
            
test_seq_len = np.asarray(test_seq_len)
test_data = np.abs(np.asarray(test_data))
test_label = np.asarray(test_label)
print train_data.shape
print test_data.shape

#1 sigmoid >> softmax
#2 sparse matrix

'''

inputs = tf.placeholder(tf.float32, [None, 161, 257, 1])
y = tf.sparse_placeholder(tf.int32)
seq_len = tf.placeholder(tf.int32, [None])
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
pooled_outputs = []

times = 0
with tf.name_scope("conv-maxpool"):

    # Convolution Layer
    filter_shape = [1, 3, 1, 128]
    W0 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W0")
    b0 = tf.Variable(tf.constant(0.1, shape=[128]), name="b0")

    conv = tf.nn.conv2d(
        inputs,
        W0,
        strides=[1, 1, 3, 1],
        padding="SAME",
        name="conv")
    # Apply nonlinearity
    h = tf.nn.relu(tf.nn.bias_add(conv, b0), name="relu"+str(times))
    # Maxpooling over the outputs
    pooled = tf.nn.max_pool(
        h,
        ksize=[1, 1, 3, 1],
        strides=[1, 1, 1, 1],
        padding='SAME',
        name="pool")
    pooled_outputs.append(pooled)


# fore 4 conv layers without pool 0,1,2,3

while(times<4):
    with tf.name_scope("conv-maxout-%s" %times):
        # Convolution Layer
        filter_shape = [1, 5, 128, 128]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W"+str(times))
        b = tf.Variable(tf.constant(0.1, shape=[128], name="b"+str(times)))
        conv = tf.nn.conv2d(
            pooled_outputs[times],
            W,
            strides=[1, 1, 1, 1],
            padding="SAME",
            name="conv")

        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu"+str(times))

        pooled_outputs.append(h)
        times = times + 1 ##times = 4

times = times - 1        ##times = 3

with tf.name_scope("conv-maxout-%s" %times):
    # Convolution Layer
    filter_shape = [1, 5, 128, 256]
    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W"+str(times))
    b = tf.Variable(tf.constant(0.1, shape=[256]), name="b"+str(times))
    conv = tf.nn.conv2d(
        pooled_outputs[-1],
        W,
        strides=[1, 1, 1, 1],
        padding="SAME",
        name="conv")
    
    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu"+str(times))
    pooled_outputs.append(h)

times = times + 1    ##times = 4


# late 5 convn_classes layer without pool 4,5,6,7,8
while(times< 9):
    with tf.name_scope("conv-maxout-%s" %times):
        # Convolution Layer
        filter_shape = [1, 5, 256, 256]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W"+str(times))
        b = tf.Variable(tf.constant(0.1, shape=[256]), name="b"+str(times))
        conv = tf.nn.conv2d(
            pooled_outputs[-1],
            W,
            strides=[1, 1, 1, 1],
            padding="SAME",
            name="conv")

        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu"+str(times))

        pooled_outputs.append(h)
        times = times + 1  ## times = 9

## shape [batch, length, heigth, depth]
cnn_outputs = pooled_outputs[-1] 
c_o_shape  = cnn_outputs.get_shape().as_list()
## reshape to [batch*length, height*depth]
cnn_outputs = tf.reshape(cnn_outputs, [-1, \
        c_o_shape[2]*c_o_shape[3]])

print cnn_outputs
W_fc1 = tf.Variable(tf.truncated_normal([c_o_shape[2]*c_o_shape[3], \
        n_classes], stddev=0.1), dtype=tf.float32)
B_fc1 = tf.Variable(tf.constant(0., shape=[n_classes]), dtype=tf.float32)


with tf.name_scope("logits"):
    ## logits shape [batch*length, classes]
    logits = tf.matmul(cnn_outputs, W_fc1) + B_fc1
    ## reshape to [batch, length, classes]
    logits = tf.reshape(logits, [-1, c_o_shape[1], n_classes])
    ## transpose to [length, batch, classes] to fit the ctc_loss
    logits = tf.transpose(logits, [1, 0, 2])

cost = tf.reduce_mean(tf.nn.ctc_loss(y, logits, seq_len))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

decoded, _ = tf.nn.ctc_greedy_decoder(logits, seq_len)
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), y))



# Launch the graph
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:

        idx_train = np.random.choice(train_data.shape[0], batch_size)
        batch_x = np.expand_dims(train_data[idx_train], axis=-1)
        batch_y = train_label[idx_train]

        idx_test = np.random.choice(test_data.shape[0], batch_size)
        temp_batch_x = np.expand_dims(test_data[idx_test], axis=-1)
        temp_batch_y = train_label[idx_test]

        train_cost, _ = sess.run([cost, optimizer], feed_dict={inputs: batch_x, y: batch_y, seq_len: train_seq_len})
        train_ler = sess.run(ler, feed_dict={inputs: batch_x, y: batch_y, seq_len: train_seq_len})
        
        if step % display_step == 0:
            eva_ler, eva_cost = sess.run([ler, cost], feed_dict={inputs: batch_x, y: batch_y, seq_len: train_seq_len})
            #accuracy_ = sess.run(accuracy, feed_dict = {inputs: temp_batch_x, y: temp_batch_y, seq_len: test_seq_len})
            print ("after {0}epoch, train_ler = {1}, train_cost = {2}, eva_ler = {3}, eva_cost = {4}".format(step, batch_ler,train_cost, eva_ler, eva_cost))
            if(eva_ler <0.3):
                #saver.save(sess, "./checkpoint/model.ckpt")
                sess.run(y_conv, feed_dict={inputs: test_data, y: test_label})
                sess.run(accuracy, feed_dict = {inputs: temp_batch_x, y: temp_batch_y})
                print ("####thus far test accurancy is : %f #####" % (eva_ler))
                validate_best = eva_ler
            
                if eva_ler < validate_best:
                    validate_best = eva_ler
                    if validate_best < 0.2:
                        pass
                print ("####thus far best evaluation label error rate is : %f #####" %(validate_best))

        step += 1
print("Optimization Finished!")
sess.close()
