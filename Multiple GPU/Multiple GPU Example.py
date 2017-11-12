# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 06:22:35 2017

@author: Bright
"""

import tensorflow as tf



a = tf.random_uniform([1000, 100]) # <tf.Tensor 'random_uniform:0' shape=(1000, 100) dtype=float32>
b = tf.random_uniform([1000, 100]) # <tf.Tensor 'random_uniform:0' shape=(1000, 100) dtype=float32>


split_a = tf.split(a, 2)
"""
[<tf.Tensor 'split:0' shape=(500, 100) dtype=float32>,
 <tf.Tensor 'split:1' shape=(500, 100) dtype=float32>]
"""


split_b = tf.split(b, 2)
"""
[<tf.Tensor 'split:0' shape=(500, 100) dtype=float32>,
 <tf.Tensor 'split:1' shape=(500, 100) dtype=float32>]
"""



split_c = []
for i in range(2):
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=i)):
        split_c.append(split_a[i] + split_b[i])

c = tf.concat(split_c, axis=0) # <tf.Tensor 'concat:0' shape=(1000, 100) dtype=float32>

tf.Session().run(c)