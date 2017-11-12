# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 07:04:02 2017

@author: Bright
"""

# "c:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"  --loop=1 --query-gpu=memory.used --format=csv memory.used [MiB]

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

####################################################

def make_parallel(fn, num_gpus, **kwargs):

    in_splits = {}  # create empty dictionary

    # for each of the tensors in kwargs, create a split and add it to the dictionary
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)  

    loss_split = [] # create empty list
    pred_split = [] 
    
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            # allow for variable reuse on GPUs beyond index 0
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                #pass the splits into the function and append results
                loss, correct_prediction = fn(**{k : v[i] for k, v in in_splits.items()})                              
                loss_split.append(loss)
                pred_split.append(correct_prediction)

    return tf.concat(loss_split, axis=0), tf.concat(pred_split, axis=0)

####################################################

mnist = input_data.read_data_sets('D:\\MNIST\\',one_hot=True)

x_train = mnist.train.images[:,:]
y_train = mnist.train.labels[:,:]

####################################################

def weight_variable(name,shape):
    initial=tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name, initializer= initial)

def bias_variable(name,shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name, initializer=initial)

def conv2d(x,W):    
    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    
def model(x, y, keep_prob):

    # reshape
    x_image = tf.reshape(x,[-1,28,28,1]) #represents batch, height, width, channels
    
    # first convolutional layer                         
    W_conv1 = weight_variable("w_conv1",[5,5,1,32]) #represent filter height, width, input channels, and output channels
    b_conv1 = bias_variable("b_conv1",[32])  
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # second convolutional layer    
    W_conv2 = weight_variable("w_conv2",[5,5,32,64])
    b_conv2 = bias_variable("b_conv2",[64])    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # fully-concected layer with 1024 neurons    
    W_fc1 = weight_variable("w_fc1",[7*7*64, 1024]) # 7*7*64 = 3136
    b_fc1 = bias_variable("b_fc1",[1024])    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # dropout    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # output layer (brings things back to ten values)   
    W_fc2 = weight_variable("w_fc2",[1024,10])
    b_fc2 = bias_variable("b_fc2",[10])
    
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv)
    correct_prediction = tf.equal( tf.argmax(y_conv,1), tf.argmax(y,1) )

    return loss, correct_prediction

####################################################

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32) # need one for each GPU

loss,correct_prediction = make_parallel(model, 2, x=x, y=y, keep_prob=keep_prob)

mean_loss = tf.reduce_mean(loss)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

train_op = tf.train.AdamOptimizer(1e-4).minimize(mean_loss,colocate_gradients_with_ops=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

####################################################

n_samples = len(x_train)
batch_size = 100
keep_prob_half = np.array([0.5,0.5])
keep_prob_full = np.array([1,1])


for i in range (20000):
    indices = np.random.choice(n_samples, batch_size)
    x_batch = x_train[indices]
    y_batch = y_train[indices]
    
    _, loss_val = sess.run([train_op, loss], {x: x_batch, y: y_batch, keep_prob: keep_prob_half} )

    if i%100 == 0:
        train_accuracy = sess.run(accuracy,{x: x_batch, y: y_batch, keep_prob: keep_prob_full})
        print("step %d, train accuracy %g"%(i,train_accuracy))
    
# Show test accuracy      
test_accuracy = sess.run(accuracy,{x: mnist.test.images, y: mnist.test.labels, keep_prob: keep_prob_full})
print(test_accuracy)    




