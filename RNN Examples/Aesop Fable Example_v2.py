
# https://medium.com/towards-data-science/lstm-by-example-using-tensorflow-feb0c1968537



import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import collections
import random
#import textblob

logs_path = 'C:\\Users\\Administrator\\Dropbox\\AWS\\RNN Example\\Aesop data and logs'
writer = tf.summary.FileWriter(logs_path)


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


#################################################

def build_dataset(words):
    count = collections.Counter(words).most_common() #creates a list of word/count pairs 
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary) #dictionary value is the numerical order of the word in the count object
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)

vocab_size = len(dictionary)


##############################

# model paramaters

learning_rate = 0.001
training_iters = 50000 
display_step = 1000

n_input = 3 # this is how many words are read at a time

# number of units in RNN cell
# I think this is equivalent to saying that the state vector has dimension 512
n_hidden = 512

# create input placeholders

x = tf.placeholder("float", [None, n_input,1])
y = tf.placeholder("float", [None, vocab_size])




# RNN output node weights and biases
weights = { 'out': tf.Variable(tf.random_normal([n_hidden, vocab_size])) }

biases = {  'out': tf.Variable(tf.random_normal([vocab_size]))  }





def RNN(x, weights, biases):
    
    # reshape input data to [1, n_input]
    
    x = tf.reshape(x,[-1, n_input])

    
    # Generate a n_input-element sequence of inputs by x into 
    # into n_input sub-arrays along axis=1
    
    x = tf.split(x,n_input,1)
    
    
    # 2-layer LSTM, each layer has n_hidden units
    
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    
    # generate prediction
    
    # interesting that rnn.static_rnn sort of executes the cell
    
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    
    
    # capture only the last output
    
    return ( tf.matmul(outputs[-1], weights['out']) + biases['out'] )
    
    
pred = RNN(x, weights, biases)

# Loss and optimizer

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)


# Model evaluation

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initialize

init=tf.global_variables_initializer()

# Launch the graph

    
session = tf.Session()

session.run(init)
step=0
offset = random.randint(0,n_input+1) #random integer between 0 and 3
end_offset = n_input+1 # in our case tihs is 4
acc_total = 0
loss_total = 0

writer.add_graph(session.graph)

while step < training_iters:
    
    # Genereate a minibatch
    
    # when offset gets close to the end of the training data, restart near the beginning
    if offset > (len(training_data) - end_offset):
        offset = random.randint(0, n_input+1)
        
    # get the integer representations of the words and reshape into array
    symbols_in_keys = [[dictionary[str(training_data[i])]] for i in range(offset, offset+n_input)]
    symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input,1])
    
    # create onehot vector by creating vector of zeros and then changing the
    # element for the target word to equal 1.0
    symbols_out_onehot = np.zeros([vocab_size], dtype=float)
    symbols_out_onehot[dictionary[str(training_data[offset+n_input])]] = 1.0
    symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
    
    _,acc,loss, onehot_pred = session.run([optimizer, accuracy, cost,pred], \
                                          feed_dict = {x: symbols_in_keys, y: symbols_out_onehot})
    
    loss_total += loss
    acc_total += acc
    
    if (step+1) % display_step ==0:
        print("Iter= " + str(step+1) + \
              ", Average Loss= " +  "{:.6f}".format(loss_total/display_step) + \
              ", Average Accuracy= " +  "{:.2f}%".format(100*acc_total/display_step))
        acc_total = 0
        loss_total = 0
        symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
        symbols_out = training_data[offset + n_input]
        symbols_out_pred = reverse_dictionary[int(tf.argmax(onehot_pred,1).eval(session=session))]
        print("%s - [%s] vs [%s]" % (symbols_in, symbols_out, symbols_out_pred))
        
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
        
    symbols_in_keys = [dictionary[str(words[i])] for i in range(len(words))]
    for i in range(32):
            #feed the starter words into the model
        keys = np.reshape(np.array(symbols_in_keys), [-1,n_input,1]) 
        onehot_pred = session.run(pred, feed_dict={x: keys})
        
            # convert prediction vector to actual word
        onehot_pred_index = int(tf.argmax(onehot_pred,1).eval(session=session))
        
            # update sentence string (for final output)
        sentence = "%s %s" % (sentence, reverse_dictionary[onehot_pred_index])
        
            # update the model input by dropping the first word and adding the new word
        symbols_in_keys = symbols_in_keys[1:]
        symbols_in_keys.append(onehot_pred_index)
        
    print(sentence)

            
            
            



