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
import ktrain
from ktrain import text
from sklearn.metrics import accuracy_score


# Custom imports
# Some of those functions can probably be incorporated as methods in the class
from mr_generic_scripts import *

class MR_transformer:
    
    def __init__(self, text_cols, age_list, class_names, max_len):
        
        # Initialize the core variables
        
        # The current classifier 
        self.mr_c = None
        
        # The current text model
        self.mr_t = None
               
        # Initialize model variables
        
        self.mr_set_model_vars(text_cols, age_list, class_names, max_len)
    
    # Function that sets model variables
    # Input: list of questions, list of ages, size of vocabulary, max len of sentence
    # Also includes certain pre-build variables for truncating
    # Also includes certain pre-built variables for dataset creation (batch size, shuffle buffer)
    def mr_set_model_vars(self, text_cols, age_list, class_names, max_len,
                          model_name = 'distilbert-base-uncased', batch_size = 6, 
                          l_rate = 8e-5, train_iter = 4):
        
        # List of questions
        self.q_list = text_cols
        
        # List of ages
        self.age_list = age_list
        
        # Size of the vocabulary
        self.class_names = class_names
        
        # Padding length
        self.max_len = max_len
        
        # Transformer to use
        self.model_name = model_name
        
        # Batch size
        self.batch_size = batch_size
        
        # Learning rate
        self.l_rate = l_rate
        
        # Number of training iteratioins
        self.train_iter = train_iter
    

    # Function that sets evaluation variables
    def mr_set_eval_vars(self, eval_q, eval_age, return_err = False):
        
        # Whether or not to perform evaluation by question
        self.eval_q = eval_q
        
        # Whether or not to perform evaluation by age
        self.eval_age = eval_age
        
        # Whether or not to return wrong predictions
        self.return_err = return_err
        
          
   
    # Function that trains the classifier
    # Input - a train set, and a val set
    def mr_train(self, train_df, val_df):
        
        # Reset the model at the start of each training
        
        self.mr_t = text.Transformer(self.model_name, maxlen = self.max_len, 
                                     class_names = self.class_names)
        
        # Preprocess the training
        train_data = self.mr_t.preprocess_train(train_df["Answer"].values, train_df["Score"].values)
        
        # Preprocess the testing
        val_data = self.mr_t.preprocess_test(val_df["Answer"].values, val_df["Score"].values)
        
        # Get the actual classifier
        model = self.mr_t.get_classifier()
        learner = ktrain.get_learner(model, train_data=train_data, val_data=val_data, 
                                     batch_size=self.batch_size)
        
        # Train the model
        learner.fit_onecycle(self.l_rate, self.train_iter)
        
        # Print results for validation
        learner.validate(class_names=self.mr_t.get_classes())
        
        self.mr_c = ktrain.get_predictor(learner.model, preproc=self.mr_t)
        
        
        
    # Function that evaluates the model on a test set
    # Input - test set
    def mr_test(self, test_df):
        
        # Initialize output vars
        acc_scores = []
        f1_scores = []
        
        X_test = test_df['Answer'].values
        y_test = test_df['Score'].values
        
        print("Testing the model on the test set:")
               
        # Get the actual predictions of the model for the test set
        y_pred = self.mr_c.predict(X_test)
        
        # Calculate accuracy
        test_acc = accuracy_score(y_test.tolist(), [float(ele) for ele in y_pred])
        
        # Calculate macro F1
        macro_score = sklearn.metrics.f1_score(y_test.tolist(), 
                                               [float(ele) for ele in y_pred],
                                               average='macro')
        
        print('Test Accuracy: {} \n'.format(round(test_acc,2)))
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
            X_test = cur_q['Answer'].values
            y_test = cur_q['Score'].values
            
            print("Evaluating column {} with value {}".format(col_name,col_val))
                      
            # Get the actual predictions of the model for the test set
            y_pred = self.mr_c.predict(X_test)
            
            # Calculate accuracy
            test_acc = accuracy_score(y_test.tolist(), [float(ele) for ele in y_pred])
            
            # Calculate macro F1
            macro_score = sklearn.metrics.f1_score(y_test.tolist(), 
                                                   [float(ele) for ele in y_pred],
                                                   average='macro')
            
            print('Accuracy: {} \n'.format(round(test_acc,2)))
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
    
      
    
            
            
            
