# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:01:15 2020

@author: Venelin Kovatchev


The class for the BILSTM classifier for the "What is on your mind" 
COLING2020 paper

The class uses tensor flow BILSTM as core model

Different methods take care of processing the data in a standardized way

"""

# Import section
import pandas as pd
import numpy as np
import scipy
import nltk
import spacy
import gensim
import glob
import csv
from spellchecker import SpellChecker
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
from sklearn.base import BaseEstimator, TransformerMixin
import gensim.models.wrappers.fasttext
from scipy import sparse
from sklearn.model_selection import LeaveOneOut,KFold,train_test_split


# Custom imports
# Some of those functions can probably be incorporated as methods in the class
from mr_generic_scripts import *

# Dummy class for returning text or pos from spacy object
# Need that for smooth pipelines in scikit
# Takes a parameter for the type of output: text, pos
class ProcessSpacyData(BaseEstimator, TransformerMixin):
    def __init__(self, out_type = "text"):
            self.out_type = out_type

    # No training required, just return self
    def fit(self, X, y=None):
        return self
    
    # Get the corresponding output from the spacy object
    # Return a string (countvectorizer expected input)
    def transform(self, X, y=None):
        # Access text with token.text
        if self.out_type == "text":
            return X.apply(lambda x : " ".join([token.text for token in x]))
        # Access pos tag with token.pos_
        elif self.out_type == "pos":
            return X.apply(lambda x : " ".join([token.pos_ for token in x]))
        


class MR_SVM:
    
    def __init__(self, text_cols, age_list, C=0.1, gamma='scale', kernel='rbf'):
        
        # Initialize the core variables
        
        # The current classifier 
        self.mr_c = None
    
        
        # The current feature union
        self.feature_union = None
        
        # Initialize model variables
        
        self.mr_set_model_vars(text_cols, age_list, C, gamma, kernel)
    
    # Function that sets model variables
    # Input: list of questions, list of ages, size of vocabulary, max len of sentence
    # Also includes certain pre-build variables for truncating
    # Also includes certain pre-built variables for dataset creation (batch size, shuffle buffer)
    def mr_set_model_vars(self, text_cols, age_list, C, gamma, kernel):
        
        # List of questions
        self.q_list = text_cols
        
        # List of ages
        self.age_list = age_list
        
        # Regularization parameter
        self.C = C
        
        # Gamma parameter
        self.gamma = gamma
        
        # Kernel type
        self.kernel = kernel

    # Function that sets evaluation variables
    def mr_set_eval_vars(self, eval_q, eval_age, return_err = False):
        
        # Whether or not to perform evaluation by question
        self.eval_q = eval_q
        
        # Whether or not to perform evaluation by age
        self.eval_age = eval_age
        
        # Whether or not to return wrong predictions
        self.return_err = return_err
     

    # Function that prepares the data transformation
    # A simple feature prep for char n-gram, word, bigram, and pos frequencies
    # Takes an option for tfidf (default is false)
    # If tfidf is given, converts raw frequencies to tfidf
    # Takes an option for ngram range (default unigram and bigram)
    # Takes an option for char ngram range (default uni, bi, trigram)
    def ngram_features(self, tfidf = False, ng_range = (1,2), cng_range = (1,3) ):
        
        # Get the transformers to tokens and pos sequences
        tokenizer = ProcessSpacyData(out_type="text")
        pos_tagger = ProcessSpacyData(out_type="pos")
        
        # Get the vectorizers for characters, words, and pos tags
        # We take unigram and bigram word vectors, unigram, bigram, and trigram character vectors
        # We also take unigram POS vectors (frequency of parts of speech in the response)
        count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=ng_range)
        char_vectorizer = sklearn.feature_extraction.text.CountVectorizer(analyzer="char_wb",
                                                                          ngram_range=cng_range)
        pos_vectorizer = sklearn.feature_extraction.text.CountVectorizer()
        
        # Get the pipelines for each transformation
        # for word freq check the tfidf value
        if tfidf:
            # Initialize the tfidf vectorizer and add it to the pipeline instead of countvectorizer
            tfidf_vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(use_idf=True)
            
            word_freq = sklearn.pipeline.Pipeline([("tokenizer",tokenizer),
                                                   ("tfidfvectorizer",tfidf_vectorizer)])
        else:
            # If tfidf is false, just use count_vectorizer instead
            word_freq = sklearn.pipeline.Pipeline([("tokenizer",tokenizer),
                                                  ("countvectorizer", count_vectorizer)])
        
        char_freq = sklearn.pipeline.Pipeline([("tokenizer",tokenizer),
                                              ("charvectorizer", char_vectorizer)])
        
        pos_freq = sklearn.pipeline.Pipeline([("postagger",pos_tagger),
                                              ("posvectorizer", pos_vectorizer)])
        
        
        # Get a union of all features
        feature_union = sklearn.pipeline.FeatureUnion([("wordfreq", word_freq),
                                                       ("charfreq", char_freq),
                                                       ("posfreq", pos_freq)])
        
        # Set the pipeline
        self.feature_union = feature_union
    
   
   
    # Function that trains the classifier
    # Input - a train set, and a val set
    def mr_train(self, train_df):
        
        # Reset the model at the start of each training
        
        svm_cls = svm.SVC(C = self.C, gamma = self.gamma, kernel = self.kernel)
        
        # Final training pipeline
        self.mr_c = sklearn.pipeline.Pipeline([("features", self.feature_union),
                                        ("svm", svm_cls)])
        
        
        # Split X and Y
        X_train = train_df['Answer']
        y_train = train_df['Score']
        
        print("Training")
        self.mr_c.fit(X_train,y_train)
        
        
        
    # Function that evaluates the model on a test set
    # Input - test set
    def mr_test(self, test_df):
        
        # Initialize output vars
        acc_scores = []
        f1_scores = []
        
        # Split X and Y
        X_test = test_df['Answer']
        y_test = test_df['Score']
        
        print("Testing the model on the test set:")
        
        
        # Get the actual predictions of the model for the test set
        y_pred = self.mr_c.predict(X_test)
        
        # Calculate accuracy
        test_acc = sklearn.metrics.accuracy_score(y_test.tolist(), 
                                                  [float(ele) for ele in y_pred])
        
        
        # Calculate macro F1
        macro_score = sklearn.metrics.f1_score(y_test.tolist(), 
                                               [float(ele) for ele in y_pred],
                                               average='macro')
        
        print('Accuracy: {} \n'.format(round(test_acc,2)))
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
            
            # Split X and Y
            X_test = cur_q['Answer']
            y_test = cur_q['Score']
            
            print("Evaluating column {} with value {}".format(col_name,col_val))
                       
            # Get the actual predictions of the model for the test set
            y_pred = self.mr_c.predict(X_test)
            
            # Calculate accuracy
            test_acc = sklearn.metrics.accuracy_score(y_test.tolist(), 
                                                      [float(ele) for ele in y_pred])
            
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
    def mr_one_train_test(self, full_df, test_r):
        
        # Split train and test
        train_df, test_df = train_test_split(full_df, test_size = test_r)
                   
        # Train the classifier
        self.mr_train(train_df)
        
        # Test the classifier
        return(self.mr_test(test_df))
    
    
    # Function for a dummy one-run on a provided train-test split
    # Input - train_df, test_df, ratio for splitting val
    def mr_one_run_pre_split(self,train_df, test_df):
            
        # Train the classifier
        self.mr_train(train_df)
        
        # Test the classifier
        return(self.mr_test(test_df))   
   
      
    #Function for a dummy 10-fold cross validation
    # Input - full df, ratio for splitting on train/val/test, number of runs
    def mr_kfold_train_test(self, full_df, num_runs=10, r_state = 42):
        
        # Initialize output
        all_results = []        
        
        # Run k-fold split
        kf = KFold(n_splits=num_runs, shuffle=True, random_state = r_state)
        
        # Run different splits
        for train_index, test_index in kf.split(full_df):
            train_df = full_df.iloc[train_index]
            test_df = full_df.iloc[test_index]
            cur_acc, cur_f1 = self.mr_one_run_pre_split(train_df, test_df)
            
            all_results.append((cur_acc, cur_f1))
            
        return(all_results)
    

    
            
            
            
