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

#filename = 'C:\\Users\\Administrator\\Documents\\Data\\glove.6B.300d.txt'

filename = 'C:\\Users\\Bright\\Documents\\Python\\CS224d\\GLOVE Vectors\\glove.6B.300d.txt'

glove_vocab = []
glove_embd=[]

embedding_dict = {}

file = open(filename,'r',encoding='UTF-8')
for line in file.readlines():
    row = line.strip().split(' ')
    vocab_word = row[0]
    glove_vocab.append(vocab_word)
    embed_vector = [float(i) for i in row[1:]] # convert to list of float
    embedding_dict[vocab_word]=embed_vector
    
print('Loaded GLOVE')
file.close()

glove_vocab_size = len(glove_vocab)
embedding_dim = len(embed_vector)



#########################
#  load training data  #
#########################

#tomatoes_neg = 'C:\\Users\\Administrator\\Documents\\Data\\rt-polarity.neg'
#tomatoes_pos = 'C:\\Users\\Administrator\\Documents\\Data\\rt-polarity.pos'

tomatoes_neg = 'C:\\Users\\Bright\\Documents\\Python\\CS224d\\Movie Reviews\\rt-polaritydata\\rt-polarity.neg'
tomatoes_pos = 'C:\\Users\\Bright\\Documents\\Python\\CS224d\\Movie Reviews\\rt-polaritydata\\rt-polarity.pos'



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
x = np.array(list(vocab_processor.fit_transform(x_text)))    # I think this is where the padding gets applied

## Extract word:id mapping from the object.
doc_vocab_dict = vocab_processor.vocabulary_._mapping
doc_sorted_dict = sorted(doc_vocab_dict.items(), key = lambda x : x[1])
doc_vocab_size = len(doc_vocab_dict)

# create array of embeddings by using pretrained when available and random otherwise

embeddings_tmp=[]

for i in range(doc_vocab_size):
    item = doc_sorted_dict[i][0]
    if item in glove_vocab:
        embeddings_tmp.append(embedding_dict[item])
    else:
        rand_num = np.random.uniform(low=-0.2, high=0.2,size=embedding_dim)
        embeddings_tmp.append(rand_num)
        
embedding = np.asarray(embeddings_tmp)


########################################

num_classes = 2

input_x = tf.placeholder(tf.int32, [None, max_document_length], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")



######################################

# initial word imbeddings as non-trainable

with tf.name_scope("embedding"):
    W = tf.Variable(tf.constant(0.0, shape=[doc_vocab_size, embedding_dim]), trainable=False, name="W")
    
    embedding_placeholder = tf.placeholder(tf.float32, [doc_vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    
    embedded_chars = tf.nn.embedding_lookup(W,input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars,-1)



######################

filter_sizes = [3,4,5]
num_filters = 128

pooled_outputs=[]

for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # Convolution Layer
        filter_shape = [filter_size, embedding_dim, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(embedded_chars_expanded,W,strides=[1,1,1,1], padding="VALID",name="conv")
        # Apply nonlinearity
        h= tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
        #Apply pooling
        pooled = tf.nn.max_pool(h,ksize=[1,max_document_length - filter_size +1, 1,1],
                            strides=[1,1,1,1] , padding='VALID', name = "pool")
        pooled_outputs.append(pooled)
    
# Combine pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(values=pooled_outputs, axis=3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

###########
with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    
##########    
with tf.name_scope("output"):
    W = tf.Variable(tf.truncated_normal([num_filters_total,num_classes],stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    scores = tf.nn.xw_plus_b(h_drop,W,b,name="scores")
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

for i in range (2000):
    indices = np.random.choice(n_samples, batch_size)
    x_batch = x[indices]
    y_batch = y[indices]
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={input_x: x_batch, input_y: y_batch, dropout_keep_prob: 1.0})
        print("step %d, train accuracy %g"%(i,train_accuracy))
        
    train_step.run(feed_dict={input_x: x_batch, input_y: y_batch, dropout_keep_prob: 0.5})


    
train_accuracy = accuracy.eval(feed_dict={input_x: x, input_y: y, dropout_keep_prob: 1.0})


##########################

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pygal

output_array = h_pool_flat.eval(feed_dict={input_x: x, input_y: y, dropout_keep_prob: 1.0})


pca = PCA(n_components=50).fit_transform(output_array)

tsne = TSNE(n_components=2, random_state=0)


out_tsne = tsne.fit_transform(pca[:1000,:])



plot_dict={}

for i in range(500):  
    chart_label = x_text[i]
    x_coord,y_coord = out_tsne[i]
    coord_pair = (x_coord,y_coord)
    plot_dict[chart_label]=coord_pair


             
xy_chart = pygal.XY(stroke=False, show_legend=False) 

for k,v in sorted(plot_dict.items()):
    xy_chart.add(k,[v])
    
xy_chart.render_to_file('C:\\Users\\Bright\\Documents\\Python\\CS224d\\pygal_movies.svg')
