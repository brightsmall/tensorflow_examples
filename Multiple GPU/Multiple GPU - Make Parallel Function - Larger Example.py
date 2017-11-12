# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 07:18:18 2017

@author: Bright
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 07:04:02 2017

@author: Bright
"""

import numpy as np
import tensorflow as tf

####################################################

def make_parallel(fn, num_gpus, **kwargs):

    in_splits = {}  # create empty dictionary
    
    
    # for each of the tensors in kwargs, create a split and add it to the dictionary
    for k, v in kwargs.items():
        in_splits[k] = tf.split(v, num_gpus)  

    out_split = [] # create empty list
    
    for i in range(num_gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
            # allow for variable reuse on GPUs beyond index 0
            with tf.variable_scope(tf.get_variable_scope(), reuse=i > 0):
                #pass the splits into the function and append results
                out_split.append(fn(**{k : v[i] for k, v in in_splits.items()}))

    return tf.concat(out_split, axis=0)

####################################################

def generate_data():
    x_val = np.random.uniform(-10.0, 10.0, size=100)
    y_val = 5 * np.square(x_val) + 3
    return x_val, y_val

####################################################
    
def model(x, y):
    
    w = tf.get_variable("w", shape=[3, 1])  # w is a variable of dimension 3x1

    f = tf.stack([tf.square(x), x, tf.ones_like(x)], 1) # X^2 , x, and 1s in separate columns
    
    yhat = tf.squeeze(tf.matmul(f, w), 1) # multiply f by w and reduce extra dimension 

    loss = tf.square(yhat - y)
    return loss




x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

loss = make_parallel(model, 2, x=x, y=y)
mean_loss = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer(0.1).minimize(mean_loss,colocate_gradients_with_ops=True)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

for _ in range(5000):
    x_val, y_val = generate_data()
    _, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})

#_, loss_val = sess.run([train_op, loss], {x: x_val, y: y_val})

print(sess.run(tf.contrib.framework.get_variables_by_name("w")))


