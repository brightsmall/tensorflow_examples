# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 11:26:12 2018

@author: Bright
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import sparse

#file_path = "D:\\Collaborative Filtering\\ITR_Jan2017_Jul2018.csv"
file_path = "C:\\Users\\bsmal725\\Documents\\My Data\\Collaborate Filtering\\ITR_Jan2017_Jul2018.csv"

raw_data = pd.read_csv(file_path)
raw_data.rename(columns={'INTENT_TO_RCMMND_PROP_SCORE_ID':'ITREC'}, inplace=True)

# set parameters

marsha_count = raw_data['MARSHA'].nunique()
customer_count = raw_data['CUSTOMER_KEY'].nunique()
num_features = 8
batch_size = 1000


# create dictionary for MARSHAS

marsha_list = raw_data['MARSHA']
marsha_list = list(set(marsha_list))
marsha_list.sort()
index_list = list(range(len(marsha_list)))
marsha_dict = dict(zip(index_list,marsha_list))
marsha_dict_reverse = dict(zip(marsha_list,index_list))

# create dictionary for customers

customer_list = raw_data['CUSTOMER_KEY']
customer_list = list(set(customer_list))
customer_list.sort()
index_list = list(range(len(customer_list)))
customer_dict = dict(zip(index_list,customer_list))
customer_dict_reverse = dict(zip(customer_list,index_list))

# aggregate scores so there is only one hotel score per guest

cust_summary = raw_data.groupby(['MARSHA', 'CUSTOMER_KEY'])[['ITREC']].mean()
cust_summary.reset_index(inplace=True )

# mean normalization

marsha_means = raw_data.groupby(['MARSHA'])[['ITREC']].mean()
marsha_means.reset_index(inplace=True )
marsha_means.rename(columns={'ITREC':'MEAN_ITREC'}, inplace=True)

cust_summary = pd.merge(cust_summary,marsha_means, on='MARSHA')
cust_summary['NORM_ITREC'] = cust_summary['ITREC'] - cust_summary['MEAN_ITREC'] 

# convert to sparse matrix

cust_summary['MARSHA'] = [ marsha_dict_reverse[i] for i in cust_summary['MARSHA'] ]
cust_summary['CUSTOMER_KEY'] = [ customer_dict_reverse[i] for i in cust_summary['CUSTOMER_KEY'] ]

cust_summary_sparse = sparse.coo_matrix((cust_summary['NORM_ITREC'], (cust_summary['CUSTOMER_KEY'], cust_summary['MARSHA'] )), shape=(customer_count, marsha_count),dtype=np.float)
cust_summary_sparse = cust_summary_sparse.tocsr()

##########################
# Modeling
####################
    
    
# Randomly initialize 6635x8 matrix for MARSHAs and features
#X=np.random.randn(marsha_count,num_features)

# Randomly initialize 4729644 x 8 matrix for customers and features
Theta_Stored = np.random.randn(customer_count,num_features)
 

# run tensorflow

tf.reset_default_graph()

# placeholders and inputs

Y = tf.placeholder(tf.float32, [marsha_count,batch_size], name="Y")
R = tf.placeholder(tf.float32, [marsha_count,batch_size], name="R")

# Weights to train

X = tf.Variable(tf.truncated_normal([marsha_count, num_features], stddev=.01),  name="X")
Theta = tf.Variable(tf.truncated_normal( [batch_size,num_features], stddev=.01), name="Theta")


# initialize assign_op
rand_idx = np.random.choice(customer_count, batch_size, replace=False)
Theta_batch = Theta_Stored[rand_idx,:]
assign_op = tf.assign(Theta, Theta_batch)

y_ = tf.matmul(X,Theta,transpose_b=True)
J1 = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(y_ ,Y)),R))
J2 = tf.reduce_sum(tf.square(X)) + tf.reduce_sum(tf.square(Theta))
J = (J1 + J2)/2

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(J)



init = tf.global_variables_initializer()

# loop

sess = tf.Session()

sess.run(init)


    #loop
for i in range(100):

    
    # randomly sample indices    
    rand_idx = np.random.choice(customer_count, batch_size, replace=False)
    
    # create df with customer responses
    trg_sparse = cust_summary_sparse[rand_idx,:]
    trg_sparse = trg_sparse.transpose()
    trg_target = trg_sparse.todense()
    
    #survey_trg = pd.concat([cust_summary[cust_summary['CUSTOMER_KEY']==i] for i in rand_idx], ignore_index=True)
            
    # create batch matrix that has all MARSHAs and responses from 1000 customers
    #trg_target = np.zeros((marsha_count,batch_size),dtype=np.float)
    
    # reassign indices based on sample of 1000
    #trg_cols, trg_col_idx = np.unique(survey_trg['CUSTOMER_KEY'], return_inverse=True)
    #trg_target[survey_trg['MARSHA'], trg_col_idx] = survey_trg['ITREC']
    
    R_coord = trg_sparse.nonzero()
    R_coord = {'x':R_coord[0],'y':R_coord[1]}
    R_coord = pd.DataFrame(R_coord)
    R_coord['val']=1
    trg_R = sparse.coo_matrix((R_coord['val'], (R_coord['x'],R_coord['y'])), shape= (marsha_count,batch_size) ,dtype=np.float)
    trg_R = trg_R.todense()
    
    feed = {Y: trg_target, R: trg_R}
    
    # create batch of weights
    Theta_batch = Theta_Stored[rand_idx,:]
    
    # run training on batch
    
    sess.run(assign_op)
    
    _, loss_val, Theta_Update = sess.run([train_op, J, Theta], feed_dict=feed)

    for t_idx, val in enumerate(rand_idx):
        Theta_Stored[val,:] = Theta_Update[t_idx]






