# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 11:26:12 2018

@author: Bright
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import sparse

file_path = "D:\\Collaborative Filtering\\ITR_Jan2017_Jul2018.csv"

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

# convert to indices

cust_summary['MARSHA'] = [ marsha_dict_reverse[i] for i in cust_summary['MARSHA'] ]
cust_summary['CUSTOMER_KEY'] = [ customer_dict_reverse[i] for i in cust_summary['CUSTOMER_KEY'] ]


##########################
# Modeling
####################
    
    
# Randomly initialize 6635x8 matrix for MARSHAs and features
X=np.random.randn(marsha_count,num_features)

# Randomly initialize 4729644 x 8 matrix for customers and features
Theta = np.random.randn(customer_count,num_features)
     
# randomly sample indices
    
rand_idx = np.random.choice(customer_count, batch_size, replace=False)


  
survey_trg = pd.concat([cust_summary[cust_summary['CUSTOMER_KEY']==i] for i in rand_idx], ignore_index=True)
        
# create batch matrix that has all MARSHAs and responses from 1000 customers

trg_target = np.zeros((marsha_count,batch_size),dtype=np.float)

# reassign indices based on sample of 1000
trg_cols, trg_col_idx = np.unique(survey_trg['CUSTOMER_KEY'], return_inverse=True)

trg_target[survey_trg['MARSHA'], trg_col_idx] = survey_trg['ITREC']

# create batch of weights

Theta_batch = Theta[rand_idx,:]


# run tensorflow



# save weights back to Theta

Theta[rand_idx,:] = new_weights



