
# https://medium.com/towards-data-science/lstm-by-example-using-tensorflow-feb0c1968537

# Version Notes

# Adding instrumentation for Tensorboard


import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import collections
import random
from scipy import spatial

#filepath_glove = 'C:\\Users\\Administrator\\Documents\\Data\\glove.6B.300d.txt'
filepath_glove = 'C:\\Users\\Bright\\Documents\\Python\\CS224d\\GLOVE Vectors\\glove.6B.300d.txt'

logs_path = 'C:\\Users\\Bright\\Dropbox\\AWS\\Aesop_logs\\'

##########################
#Load GLOVE vectors

glove_vocab = []
glove_embd=[]

embedding_dict = {}

file = open(filepath_glove,'r',encoding='UTF-8')
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

###########################


fable_text = """
long ago , the mice had a general council to consider what measures they could take to outwit their 
common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal 
to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the 
sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , 
we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a 
ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire 
while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said 
that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the 
old mouse said it is easy to propose impossible remedies .
"""

fable_text = fable_text.replace('\n','')

#this function puts all the words in a single column vector within a numpy array

def read_data(raw_text):
        content = raw_text
        #content = [x.strip() for x in content]
        content = content.split() #splits the text by spaces (default split character)
        content = np.array(content)
        content = np.reshape(content, [-1, ])
        return content

training_data = read_data(fable_text)

##################


  
#Create dictionary and reverse dictionary with word ids
  

def build_dictionaries(words):
    count = collections.Counter(words).most_common() #creates a list of word/count pairs 
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary) #dictionary value is the numerical order of the word in the count object
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dictionaries(training_data)
  
#Create embedding array

doc_vocab_size = len(dictionary)
dict_as_list = sorted(dictionary.items(), key = lambda x : x[1])

embeddings_tmp=[]

for i in range(doc_vocab_size):
    item = dict_as_list[i][0]
    if item in glove_vocab:
        embeddings_tmp.append(embedding_dict[item])
    else:
        rand_num = np.random.uniform(low=-0.2, high=0.2,size=embedding_dim)
        embeddings_tmp.append(rand_num)

# final embedding array corresponds to dictionary of words in the document
        
embedding = np.asarray(embeddings_tmp)

# create tree so that we can later search for closest vector to prediction
tree = spatial.KDTree(embedding)

##############################

# model paramaters

tf.reset_default_graph()

n_input = 3 # this is how many words are read at a time

n_hidden = 512

# create input placeholders


with tf.name_scope("embedding"):
    x = tf.placeholder(tf.int32, [None, n_input], name="x-input")
    embedding_placeholder = tf.placeholder(tf.float32, [doc_vocab_size, embedding_dim], name="embedding-input")
    W_embed = tf.Variable(tf.constant(0.0, shape=[doc_vocab_size, embedding_dim]), trainable=False, name="W-embeddings")
    embedding_init = W_embed.assign(embedding_placeholder)
    embedded_chars = tf.nn.embedding_lookup(W_embed,x)

    
# reshape input data
with tf.variable_scope("Model"):
    x_unstack =  tf.unstack(embedded_chars, n_input, 1)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    rnn_out, rnn_states = rnn.static_rnn(rnn_cell, x_unstack, dtype=tf.float32,scope="RNN")
    

 # RNN output node weights and biases
with tf.name_scope("Output"):
    W_out = tf.Variable(tf.random_normal([n_hidden, embedding_dim]), name="W_out")
    b_out = tf.Variable(tf.random_normal([embedding_dim]), name="b_out")
    pred = tf.matmul(rnn_out[-1], W_out) + b_out   # capture only the last output
    

# create optimizer
learning_rate = 0.001

with tf.name_scope("Train"):
    
    y = tf.placeholder(tf.float32, [None, embedding_dim], name="y-input")
    cost = tf.reduce_mean(tf.nn.l2_loss(pred-y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


tf.summary.scalar("cost", cost)

summary_op = tf.summary.merge_all()

# Model evaluation

#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize

init=tf.global_variables_initializer()


# Launch the graph
    
sess = tf.Session()

sess.run(init)
sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

writer=tf.summary.FileWriter(logs_path, graph=sess.graph)

step=0
offset = random.randint(0,n_input+1) #random integer between 0 and 3
end_offset = n_input+1 # in our case this is 4
acc_total = 0
loss_total = 0

training_iters = 5000 
display_step = 500



while step < training_iters:
    
    ### Genereate a minibatch ###
    
    # when offset gets close to the end of the training data, restart near the beginning

    if offset > (len(training_data) - end_offset):
        offset = random.randint(0, n_input+1)
        
    # get the integer representations for the input words
    
    x_integers = [[dictionary[str(training_data[i])]] for i in range(offset, offset+n_input)]
    x_integers = np.reshape(np.array(x_integers), [-1, n_input])
    
    # create embedding for target vector 
    
    y_position = offset+n_input
    y_integer = dictionary[training_data[y_position]]
    y_embedding = embedding[y_integer,:]
    y_embedding = np.reshape(y_embedding,[1,-1])

    
    _,loss, pred_, summary = sess.run([optimizer, cost,pred,summary_op], feed_dict = {x: x_integers, y: y_embedding})
    
    writer.add_summary(summary,step)
    
    loss_total += loss

    # display output to show progress
    
    if (step+1) % display_step ==0:
        words_in = [str(training_data[i]) for i in range(offset, offset+n_input)] 
        target_word = str(training_data[y_position])
        
        
        nearest_dist,nearest_idx = tree.query(pred_[0],3)
        nearest_words = [reverse_dictionary[idx] for idx in nearest_idx]
        
        print("%s - [%s] vs [%s]" % (words_in, target_word, nearest_words))
        print("Average Loss= " +  "{:.6f}".format(loss_total/display_step))
        loss_total=0
        
    step +=1
    offset += (n_input+1) # not sure why you need to skip ahead 4
        

print ("Finished Optimization")


###################################

def user_input():
    
        
    prompt = "Type %s words: " % n_input
    sentence = input(prompt)
    sentence = sentence.strip()
    words = sentence.split(' ')
    
    if len(words) != n_input:
        print("Not the right number of words")
        
    x_integers = [dictionary[str(words[i])] for i in range(len(words))]
    
    for i in range(60):
            #feed the starter words into the model
        keys = np.reshape(np.array(x_integers), [-1,n_input]) 
        pred_ = sess.run(pred, feed_dict={x: keys})
        
            # convert prediction vector to actual word
        nearest_dist,nearest_idx = tree.query(pred_[0],1)
        
        nearest_word = reverse_dictionary[nearest_idx] 
        
            # update sentence string (for final output)
        sentence = "%s %s" % (sentence, nearest_word)
        
            # update the model input by dropping the first word and adding the new word
        x_integers = x_integers[1:]
        x_integers.append(nearest_idx)
        
    print(sentence)

            
            
            



