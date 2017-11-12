# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 07:04:02 2017

@author: Bright
"""

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
    
def model(a, b):
    return a + b


a = tf.random_uniform([1000, 100]) # <tf.Tensor 'random_uniform:0' shape=(1000, 100) dtype=float32>
b = tf.random_uniform([1000, 100]) # <tf.Tensor 'random_uniform:0' shape=(1000, 100) dtype=float32>


c = make_parallel(model, 2, a=a, b=b)

tf.Session().run(c)