# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 10:38:22 2017

@author: Bright
"""

#  Resources

#  http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
#  https://ireneli.eu/2017/01/17/tensorflow-07-word-embeddings-2-loading-pre-trained-vectors/


import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np

sess = tf.InteractiveSession()



###################################
#  Load Pre-trained word vectors  #
###################################

filename = 'W:\\Deep Learning\\glove.6B.300d.txt'

glove_vocab = []
glove_embd=[]

file = open(filename,'r',encoding='UTF-8')
for line in file.readlines():
    row = line.strip().split(' ')
    embed_vector = [float(i) for i in row[1:]] # convert to list of float
    glove_vocab.append(row[0])
    glove_embd.append(embed_vector)
print('Loaded GLOVE')
file.close()

vocab_size = len(glove_vocab)
embedding_dim = len(glove_embd[0])



#########################
#  load training data  #
#########################

tomatoes_neg = 'W:\\Deep Learning\\rt-polarity.neg'
tomatoes_pos = 'W:\\Deep Learning\\rt-polarity.pos'

pos_examples = list(open(tomatoes_pos,'r'))
neg_examples = list(open(tomatoes_neg,'r'))

pos_examples = [item.strip() for item in pos_examples]
neg_examples = [item.strip() for item in neg_examples]

pos_labels = [[0,1] for _ in pos_examples]
neg_labels = [[1,0] for _ in neg_examples]

x_text = pos_examples + neg_examples
y = np.concatenate([pos_labels, neg_labels])

del pos_examples
del neg_examples
del pos_labels
del neg_labels


#####################################################################################
#  Establish the vocabulary and convert x_text into x, which just contains the ids  #
#####################################################################################

max_document_length = max([len(x.split(" ")) for x in x_text])

## Create the vocabularyprocessor object, setting the max lengh of the documents.

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

## Transform the documents into vectors of ids
x = np.array(list(vocab_processor.fit_transform(x_text)))    

## Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping
sorted_vocab_dict = sorted(vocab_dict.items(), key = lambda x : x[1])
vocab_size = len(vocab_dict)

# create array of embeddings by using pretrained when available and random otherwise

embeddings_tmp=[]

for i in range(vocab_size):
    item = sorted_vocab_dict[i][0]
    if item in glove_vocab:
        embeddings_tmp.append(glove_embd[glove_vocab.index(item)])
    else:
        rand_num = np.random.uniform(low=-0.2, high=0.2,size=embedding_dim)
        #rand_str = ["%.6f" % x for x in rand_num]
        embeddings_tmp.append(rand_num)
        
embedding = np.asarray(embeddings_tmp)


########################################

# setup placeholders for training data

num_classes = 2

input_x = tf.placeholder(tf.int32, [None, max_document_length], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

######################################

# initialize word imbeddings as non-trainable
# note: this needs to run AFTER other global variables are initialized

with tf.name_scope("embedding"):
    W_embed = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W_embed")
    
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    
    embedding_init = W_embed.assign(embedding_placeholder)
    
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    
    embedded_chars = tf.nn.embedding_lookup(W_embed,input_x)
    
    embedded_chars_expanded = tf.expand_dims(embedded_chars,-1)

######################

num_filters = 128
pooled_outputs=[]

######################  3-grams

filter_size = 3
with tf.name_scope("conv-maxpool-3" ):
    # Convolution Layer
    filter_shape = [filter_size, embedding_dim, 1, num_filters]
    W_3 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_3")
    b_3 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_3")
    conv_3 = tf.nn.conv2d(embedded_chars_expanded,W_3,strides=[1,1,1,1], padding="VALID",name="conv_3")
    # Apply nonlinearity
    h_3= tf.nn.relu(tf.nn.bias_add(conv_3,b_3),name="relu_3")
    #Apply pooling
    pooled_3 = tf.nn.max_pool(h_3,ksize=[1,max_document_length - filter_size +1, 1,1],
                            strides=[1,1,1,1] , padding='VALID', name = "pool_3")
    pooled_outputs.append(pooled_3)

######################  4-grams

filter_size = 4    
with tf.name_scope("conv-maxpool-4" ):
    # Convolution Layer
    filter_shape = [filter_size, embedding_dim, 1, num_filters]
    W_4 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_4")
    b_4 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_4")
    conv_4 = tf.nn.conv2d(embedded_chars_expanded,W_4,strides=[1,1,1,1], padding="VALID",name="conv_4")
    # Apply nonlinearity
    h_4= tf.nn.relu(tf.nn.bias_add(conv_4,b_4),name="relu_4")
    #Apply pooling
    pooled_4 = tf.nn.max_pool(h_4,ksize=[1,max_document_length - filter_size +1, 1,1],
                            strides=[1,1,1,1] , padding='VALID', name = "pool_4")
    pooled_outputs.append(pooled_4)

######################  5-grams

filter_sizes = 5    
with tf.name_scope("conv-maxpool-5" ):
    # Convolution Layer
    filter_shape = [filter_size, embedding_dim, 1, num_filters]
    W_5 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_5")
    b_5 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_5")
    conv_5 = tf.nn.conv2d(embedded_chars_expanded,W_5,strides=[1,1,1,1], padding="VALID",name="conv_5")
    # Apply nonlinearity
    h_5= tf.nn.relu(tf.nn.bias_add(conv_5,b_5),name="relu_5")
    #Apply pooling
    pooled_5 = tf.nn.max_pool(h_5,ksize=[1,max_document_length - filter_size +1, 1,1],
                            strides=[1,1,1,1] , padding='VALID', name = "pool_5")
    pooled_outputs.append(pooled_5)   
    
# Combine pooled features
num_filters_total = num_filters * 3
h_pool = tf.concat(values=pooled_outputs, axis=3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

###########
with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    
##########    
with tf.name_scope("output"):
    W_output = tf.Variable(tf.truncated_normal([num_filters_total,num_classes],stddev=0.1), name="W_output")
    b_output = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_output")
    scores = tf.nn.xw_plus_b(h_drop,W_output,b_output,name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")
     
    
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
    loss = tf.reduce_mean(losses)
    
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    
    
###########

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


sess.run(tf.global_variables_initializer())

sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})



n_samples = len(x_text)
batch_size = 50

for i in range (1000):
    indices = np.random.choice(n_samples, batch_size)
    x_batch = x[indices]
    y_batch = y[indices]
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={input_x: x_batch, input_y: y_batch, dropout_keep_prob: 1.0})
        print("step %d, train accuracy %g"%(i,train_accuracy))
        
    train_step.run(feed_dict={input_x: x_batch, input_y: y_batch, dropout_keep_prob: 0.5})


test = sess.run(W_embed)

    
train_accuracy = accuracy.eval(feed_dict={input_x: x, input_y: y, dropout_keep_prob: 1.0})
