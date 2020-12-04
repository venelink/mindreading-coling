# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:01:15 2020

@author: Venelin Kovatchev


The class for the BILSTM classifier for the "What is on your mind" 
COLING2020 paper

The class uses tensor flow BILSTM as core model

Different methods take care of processing the data in a standardized way

"""

import pandas as pd
import numpy as np
import scipy
import nltk
import spacy
import gensim
import glob
import csv
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_score
import sklearn.model_selection
import sklearn.pipeline
import re
from sklearn import svm
from sklearn import *
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
import gensim.models.wrappers.fasttext
from scipy import sparse
import tensorflow_datasets as tfds
import tensorflow as tf
import collections
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import LeaveOneOut,KFold,train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.layers import Layer, Dense, Flatten, Activation, Permute
from tensorflow.keras.layers import Multiply, Lambda, Reshape, Dot, Concatenate, RepeatVector, \
    TimeDistributed, Permute, Bidirectional



from tensorflow.keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Flatten
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Convolution1D
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers


# Custom imports
# Some of those functions can probably be incorporated as methods in the class
from mr_generic_scripts import *

# Pre built self attention class
# Taken from https://github.com/uzaymacar/attention-mechanisms

class SelfAttention(Layer):
    """
    Layer for implementing self-attention mechanism. Weight variables were preferred over Dense()
    layers in implementation because they allow easier identification of shapes. Softmax activation
    ensures that all weights sum up to 1.
    @param (int) size: a.k.a attention length, number of hidden units to decode the attention before
           the softmax activation and becoming annotation weights
    @param (int) num_hops: number of hops of attention, or number of distinct components to be
           extracted from each sentence.
    @param (bool) use_penalization: set True to use penalization, otherwise set False
    @param (int) penalty_coefficient: the weight of the extra loss
    @param (str) model_api: specify to use TF's Sequential OR Functional API, note that attention
           weights are not outputted with the former as it only accepts single-output layers
    """
    def __init__(self, size, num_hops=8, use_penalization=True,
                 penalty_coefficient=0.1, model_api='functional', **kwargs):
        if model_api not in ['sequential', 'functional']:
            raise ValueError("Argument for param @model_api is not recognized")
        self.size = size
        self.num_hops = num_hops
        self.use_penalization = use_penalization
        self.penalty_coefficient = penalty_coefficient
        self.model_api = model_api
        super(SelfAttention, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        base_config['size'] = self.size
        base_config['num_hops'] = self.num_hops
        base_config['use_penalization'] = self.use_penalization
        base_config['penalty_coefficient'] = self.penalty_coefficient
        base_config['model_api'] = self.model_api
        return base_config

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.size, input_shape[2]),                                # (size, H)
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.num_hops, self.size),                                 # (num_hops, size)
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):  # (B, S, H)
        # Expand weights to include batch size through implicit broadcasting
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]
        hidden_states_transposed = Permute(dims=(2, 1))(inputs)                                     # (B, H, S)
        attention_score = tf.matmul(W1, hidden_states_transposed)                                   # (B, size, S)
        attention_score = Activation('tanh')(attention_score)                                       # (B, size, S)
        attention_weights = tf.matmul(W2, attention_score)                                          # (B, num_hops, S)
        attention_weights = Activation('softmax')(attention_weights)                                # (B, num_hops, S)
        embedding_matrix = tf.matmul(attention_weights, inputs)                                     # (B, num_hops, H)
        embedding_matrix_flattened = Flatten()(embedding_matrix)                                    # (B, num_hops*H)

        if self.use_penalization:
            attention_weights_transposed = Permute(dims=(2, 1))(attention_weights)                  # (B, S, num_hops)
            product = tf.matmul(attention_weights, attention_weights_transposed)                    # (B, num_hops, num_hops)
            identity = tf.eye(self.num_hops, batch_shape=(inputs.shape[0],))                        # (B, num_hops, num_hops)
            frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(product - identity)))  # distance
            self.add_loss(self.penalty_coefficient * frobenius_norm)  # loss

        if self.model_api == 'functional':
            return embedding_matrix_flattened, attention_weights
        elif self.model_api == 'sequential':
            return embedding_matrix_flattened

class MR_attention:
    
    def __init__(self, text_cols, age_list, v_size, max_len):
        
        # Initialize the core variables
        
        # The current classifier 
        self.mr_c = None
        
        # The current tokenizer
        self.mr_tok = None
        
        # Initialize model variables
        
        self.mr_set_model_vars(text_cols, age_list, v_size, max_len)
    
    # Function that sets model variables
    # Input: list of questions, list of ages, size of vocabulary, max len of sentence
    # Also includes certain pre-build variables for truncating
    # Also includes certain pre-built variables for dataset creation (batch size, shuffle buffer)
    def mr_set_model_vars(self, text_cols, age_list, v_size, max_len, 
                          trunc_type = 'post', padding_type = 'post', oov_tok = '<OOV>',
                          batch_size = 4, shuffle_buffer_size = 100):
        
        # List of questions
        self.q_list = text_cols
        
        # List of ages
        self.age_list = age_list
        
        # Size of the vocabulary
        self.v_size = v_size
        
        # Padding length
        self.max_len = max_len
        
        # Truncating type
        self.trunc_type = trunc_type
        
        # Padding type
        self.padding_type = padding_type
        
        # Token to replace OOV tokens
        self.oov_tok = oov_tok
        
        # Batch size for tf_dataset
        self.batch_size = batch_size
        
        # Shuffle buffer size
        self.shuffle_buffer_size = shuffle_buffer_size
        

    # Function that sets evaluation variables
    def mr_set_eval_vars(self, eval_q, eval_age, return_err = False):
        
        # Whether or not to perform evaluation by question
        self.eval_q = eval_q
        
        # Whether or not to perform evaluation by age
        self.eval_age = eval_age
        
        # Whether or not to return wrong predictions
        self.return_err = return_err

    # Convert the text from words to indexes and pad to a fixed length (needed for LSTM)
    # Input - text
    # Uses model variables for vocabulary size, token to be used for OOV, padding and truncating
    def mr_t2f(self, inp_text):
        
        # Check if a tokenizer already exists
        # If it is None, this is the first time we run the function -> fit the tokenizer
        if self.mr_tok == None:
            # Initialize the tokenizer
            self.mr_tok = Tokenizer(num_words = self.v_size, oov_token=self.oov_tok)
            
            # Fit the tokenizer
            self.mr_tok.fit_on_texts(inp_text)
            
        # Convert the dataset
        indexed_dataset = self.mr_tok.texts_to_sequences(inp_text)
        
        # Pad to max length
        padded_dataset = pad_sequences(indexed_dataset, 
                                       maxlen = self.max_len, 
                                       padding = self.padding_type, 
                                       truncating = self.trunc_type)
        
        # Return the converted dataset
        return padded_dataset
    
    # Function that created a tensorflow dataset from X and Y
    # Input: X and Y
    def mr_tf_data(self, var_X, var_y):
        
        # Convert the labels in proper format
        y_arr = var_y.to_numpy().astype(int)
        
        # Create the actual dataset and shuffle it        
        var_dataset = tf.data.Dataset.from_tensor_slices((var_X, y_arr))  
        var_dataset = var_dataset.shuffle(self.shuffle_buffer_size).batch(self.batch_size)
    
        return var_dataset
    
    # Function that converts a dataframe to a dataset
    # Input - dataframe
    def mr_to_dataset(self, cur_df):
        # X is the answer column
        cur_X = cur_df["Answer"]
        # Y is the score column
        cur_Y = cur_df["Score"]
        
        # Convert X to a one-hot vector representation
        # The vector is of a predefined fixed length and uses a fixed vocabulary size
        X_idx = self.mr_t2f(cur_X)
        
        # Create the dataset
        cur_dataset = self.mr_tf_data(X_idx,cur_Y)
        
        # Return everything
        return(X_idx, cur_Y, cur_dataset)
    
    # Function that trains the classifier
    # Input - a train set, and a val set
    def mr_train(self, train_df, val_df):
        
        # Reset the tokenizer and the model at the start of each training
        
        self.mr_c = None
        self.mr_tok = None       
        
        # Convert dataframes to datasets
        X_train_idx, y_train, train_dataset = self.mr_to_dataset(train_df)
        X_val_idx, y_val, val_dataset = self.mr_to_dataset(val_df)
        
        # Current shape var
        inp_shape = np.shape(X_train_idx[0])[0]
    
        # Define a vanilla self-attention model
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(inp_shape)),
            # Word embedding layers, size of the vocabulary X 64 dimensions
            tf.keras.layers.Embedding(self.v_size, 64),
            # Self attention layer, size is sentence length
            SelfAttention(size=self.max_len, num_hops=6, use_penalization=False,model_api='sequential'),
            # Flatten the output
            tf.keras.layers.Flatten(),
            # Dense relu layer on top of the attention
            tf.keras.layers.Dense(self.max_len, activation='relu'),
            # Add dropout to reduce overfitting
            tf.keras.layers.Dropout(.5),
            # Softmax classification for 3 classes
            tf.keras.layers.Dense(3,activation='softmax')
        ])    
        
        # Compile the model
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['accuracy'])        
        
        # Print the moodel setting
        print(model.summary())
        
        print('\n Training')
        
        # Train
        history = model.fit(train_dataset, epochs=20,
                            validation_data=val_dataset, 
                            validation_steps=30)
            
        # Update the current model variable
        self.mr_c = model
        
    # Function that evaluates the model on a test set
    # Input - test set
    def mr_test(self, test_df):
        
        # Initialize output vars
        acc_scores = []
        f1_scores = []
        
        # Convert the dataframe to a dataset
        X_test_idx, y_test, test_dataset = self.mr_to_dataset(test_df)
        
        print("Testing the model on the test set:")
        
        # Run the model internal evaluation on the test set
        test_loss, test_acc = self.mr_c.evaluate(test_dataset)
        
        # Get the actual predictions of the model for the test set
        y_pred = self.mr_c.predict_classes(X_test_idx)
        
        # Calculate macro F1
        macro_score = sklearn.metrics.f1_score(y_test.tolist(), 
                                               [float(ele) for ele in y_pred],
                                               average='macro')
        
        print('Test Macro F1: {} \n'.format(round(macro_score,2)))
        
        # Add the results to the output
        acc_scores.append(round(test_acc,2))
        f1_scores.append(round(macro_score,2))
        
        # Test by question (if requested)
        # Add the scores to the output
        # Otherwise add empty list
        if self.eval_q:
            qa_scores, qf_scores = self.mr_eval_col(test_df,"Question",self.q_list)
            
            acc_scores.append(qa_scores)
            f1_scores.append(qf_scores)
        else:
            acc_scores.append([])
            f1_scores.append([])
            
        # Test by age (if requested)
        # Add the scores to the output
        # Otherwise add empty list    
        if self.eval_age:
            aa_scores, af_scores = self.mr_eval_col(test_df,"Age",self.age_list)
            
            acc_scores.append(aa_scores)
            f1_scores.append(af_scores)
        else:
            acc_scores.append([])
            f1_scores.append([])
            
        return(acc_scores,f1_scores)
            
            
    # Function that evaluates the model by a specific column
    # Can also return the actual wrong predictions
    # Input - test set, column, values
    def mr_eval_col(self, test_df, col_name, col_vals):
        # Initialize output
        acc_scores = []
        f1_scores = []
        
        # Initialize output for wrong predictions, if needed
        if self.return_err:
            wrong_pred = []
        
        # Loop through all values
        for col_val in col_vals:
            
            # Initialize output for wrong predictions, if needed
            if self.return_err:
                cur_wrong = []
            
            # Get only the entries for the current value
            cur_q = test_df[test_df[col_name] == col_val].copy()
            
            # Convert dataframe to dataset
            X_test_idx, y_test, test_dataset = self.mr_to_dataset(cur_q)
            
            print("Evaluating column {} with value {}".format(col_name,col_val))
            
            # Print the internal evaluation
            test_loss, test_acc = self.mr_c.evaluate(test_dataset)
            
            # Get the actual predictions of the model for the test set
            y_pred = self.mr_c.predict_classes(X_test_idx)
            
            # Calculate macro F1
            macro_score = sklearn.metrics.f1_score(y_test.tolist(), 
                                                   [float(ele) for ele in y_pred],
                                                   average='macro')
            
            print('Macro F1: {} \n'.format(round(macro_score,2)))    
            
            # Add the results to the output
            acc_scores.append(round(test_acc,2))
            f1_scores.append(round(macro_score,2))
            
            if self.return_err:
                # Loop through all predictions and keep the incorrect ones
                # cur_q["Answer"], y_test, and y_pred are all matched, since they
                # are not shuffled (shuffle only applies to the test_dataset)
                for c_text,c_gold,c_pred in zip(cur_q["Answer"],y_test.tolist(),
                                                [float(ele) for ele in y_pred]):
                    if c_pred != c_gold:
                        cur_wrong.append([c_text,c_gold,c_pred])
                wrong_pred.append(cur_wrong)
            
        # Return the output
        if self.return_err:
            return(acc_scores,f1_scores, wrong_pred)
        else:
            return(acc_scores, f1_scores)
      
    
    # Function for a dummy one run on train-test
    # Input - full df, ratio for splitting on train/val/test, return errors or not
    def mr_one_train_test(self, full_df, test_r, val_r=0):
        
        # Split train and test
        train_df, test_df = train_test_split(full_df, test_size = test_r)
        
        # Check if we also need val
        if val_r > 0:
            train_df, val_df = train_test_split(train_df, test_size = val_r)
        else:
            # If not, validation is same as test
            val_df = test_df
            
        # Train the classifier
        self.mr_train(train_df, val_df)
        
        # Test the classifier
        return(self.mr_test(test_df))
    
    
    # Function for a dummy one-run on a provided train-test split
    # Input - train_df, test_df, ratio for splitting val
    def mr_one_run_pre_split(self,train_df, test_df, val_r = 0):
        # Check if we also need val
        if val_r > 0:
            train_df, val_df = train_test_split(train_df, test_size = val_r)
        else:
            # If not, validation is same as test
            val_df = test_df        
            
        # Train the classifier
        self.mr_train(train_df, val_df)
        
        # Test the classifier
        return(self.mr_test(test_df))   
   
      
    #Function for a dummy 10-fold cross validation
    # Input - full df, ratio for splitting on train/val/test, number of runs
    def mr_kfold_train_test(self, full_df, val_r=0.25, num_runs=10, r_state = 42):
        
        # Initialize output
        all_results = []        
        
        # Run k-fold split
        kf = KFold(n_splits=num_runs, shuffle=True, random_state = r_state)
        
        # Run different splits
        for train_index, test_index in kf.split(full_df):
            train_df = full_df.iloc[train_index]
            test_df = full_df.iloc[test_index]
            cur_acc, cur_f1 = self.mr_one_run_pre_split(train_df, test_df, val_r)
            
            all_results.append((cur_acc, cur_f1))
            
        return(all_results)
    

    
            
            
            
