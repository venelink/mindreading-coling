# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 12:28:11 2020

@author: Venelin
"""

import pandas as pd
import nltk
import numpy as np
import tensorflow as tf
import re
from datetime import datetime

# Mindreading general scripts and data used by all experiments

# Static variables
# Separate columns by type
bio_cols = ['ID', 'Study', 'School', 'Class', 'Child', 'SEN (0 = No, 1 = Yes)',
            'EAL (0 = No, 1 = Yes)', 'DOB', 'DOT', 'Gender (0 = Girl, 1 = Boy)', 'MHVS']

text_cols = ['SFQuestion_1_Text', 'SFQuestion_2_Text', 'SFQuestion_3_Text', 
             'SFQuestion_4_Text', 'SFQuestion_5_Text', 'SFQuestion_6_Text', 
             'SS_Brian_Text', 'SS_Peabody_Text', 'SS_Prisoner_Text', 
             'SS_Simon_Text', 'SS_Burglar_Text']

rate_cols = ['SFQ1_Rating', 'SFQ2_Rating', 'SFQ3_Rating', 
             'SFQ4_Rating', 'SFQ5_Rating', 'SFQ6_Rating', 
             'SS_Brian_Rating', 'SS_Peabody_Rating', 'SS_Prisoner_Rating',
             'SS_Simon_Rating', 'SS_Burglar_Rating']

tq_cols = ['SFQuestion_1_QT', 'SFQuestion_2_QT', 'SFQuestion_3_QT',
           'SFQuestion_4_QT', 'SFQuestion_5_QT', 'SFQuestion_6_QT',
           'SS_Brian_QT', 'SS_Peabody_QT', 'SS_Prisoner_QT',
           'SS_Simon_QT', 'SS_Burglar_QT']


# Add the questions
questions = ['Why did the men hide ? ', 'What does the woman think ? ', 'Why did the driver lock Harold in the van ? ', 
            'What is the deliveryman feeling and why ? ', 'Why did Harold pick up the cat ? ', 'Why does Harold fan Mildred ? ',
            'Why does Brian say this ? ', 'Why did she say that ? ', 'Why did the prisoner say that ? ',
            'Why will Jim look in the cupboard for the paddle ? ', 'Why did the burglar do that ? ']


# Function that reads all excel files and puts them in dataframes
def mr_excel_to_pd(fnames,calc_age=False):
    # Initialize an empty list to keep the dictionaries
    li = []
    
    
    # Read each file in panda dataframe
    for filename in fnames:
        # Try to read from excel
        try:
            #df = pd.read_excel(filename, index_col=None, header=0, encoding='utf-8')
            df = pd.read_excel(filename, index_col=None, header=0)
            # Fix incorrect column names
            df.rename(columns={'DOB DD/MM/YYYY -99 = Missing' : 'DOB',
                               'EAL 0 = No, 1 = Yes' : 'EAL (0 = No, 1 = Yes)',
                               'Gender 0 = Girl, 1 = Boy' : 'Gender (0 = Girl, 1 = Boy)',
                               'SEN 0 = No, 1 = Yes' : 'SEN (0 = No, 1 = Yes)',
                               'SEN (0 = No, 1= Yes)' : 'SEN (0 = No, 1 = Yes)',
                               'SFQ1_Rating ': 'SFQ1_Rating'},
                      inplace=True)
            
            # Check if we need to calculate age
            if calc_age:
                df['Age_Y'] = round((df['DOT'] - df['DOB']).dt.days/365)
                
            
            # Put it in the list
            li.append(df)
        # Otherwise print the file name for debug
        except:
            print(filename)
    
    # Concatenate all dataframes
    full_frame = pd.concat(li, axis=0, ignore_index=True)
    
    return full_frame

# Fucntion that creates a new column with the question and answer
# Loop through all text columns and question texts
def mr_create_qa(full_df,t_cols=text_cols,list_qs=questions):
    for text_col, question in zip(t_cols, list_qs):
        
        # Generate new column name
        new_cname = text_col.replace("_Text","_QT")
        
        # Create a new column that is the concatenation of the question and the answer
        full_df[new_cname] = question + full_df[text_col].astype(str)


# Def a funciton for basic preprocessing
# This is only for the DL experiments!
# Input is a string
def mr_tok_sc(response):
    # Tokenization (nltk for NNs, we only tokenizee)
    # Discard nans as they break nltk
    tok_res = nltk.word_tokenize(response) if not pd.isna(response) else []
    
    # Return a string
    
    return " ".join(tok_res)

# Create separate datasets for each question and one "large" dataset with all questions together
# Get a version with and without the question

def mr_create_datasets(full_df,X_cols,Y_cols):
    # Initialize a list
    q_datasets = []

    # Loop through all answer / value pairs
    for cur_X, cur_Y in zip(X_cols,Y_cols):
        # Get the data
        #cur_df = full_df[[cur_X,cur_Y,''DOB','DOT'']].copy()
        # Need to copy the following fields: ID, age, gender, question, answer
        cur_df = full_df[['"ID"',cur_X,cur_Y,'Age_Y','Gender (0 = Girl, 1 = Boy)']].copy()
        # Drop missing data and -99s
        cur_df[cur_X].replace('', np.nan, inplace=True)
        cur_df[cur_Y].replace(-99, np.nan, inplace=True)
        cur_df.dropna(inplace=True)
        
        # Add column for the current question (this is needed later on)
        cur_df["Question"] = cur_X
        
        # Rename columns for consistency
        cur_df.rename(columns={cur_X:'Answer'},inplace=True)
        cur_df.rename(columns={cur_Y:'Score'},inplace=True)
        cur_df.rename(columns={'Age_Y':'Age'},inplace=True)
        cur_df.rename(columns={'"ID"':'Child_ID'},inplace=True)
        cur_df.rename(columns={'Gender (0 = Girl, 1 = Boy)':'Gender'},inplace=True)
        
               
        # Get the year of birth from the date of birth
        #cur_df["YOB"] = cur_df["DOB"].astype(str).apply(mr_get_date)
        # Get the year of text from the date of test
        #cur_df["YOT"] = cur_df["DOT"].astype(str).apply(mr_get_date)
        # Calculate age as YOB - YOT (ugly fix due to messy date format in the 
        # excels, will have to redo it later with proper regex to parse all
        # the input)
        #cur_df["Age"] = cur_df["YOT"] - cur_df["YOB"]        
        #cur_df["Age"] = cur_df["Age_Y"]

        # Put the data in a list
        #q_datasets.append([cur_X,cur_data[cur_X],cur_data[cur_Y]])
        q_datasets.append([cur_X,cur_df])

    # Get a full dataset with all the data
    #full_X = pd.concat([X for [_,X,_] in q_datasets])
    #full_Y = pd.concat([Y for [_,_,Y] in q_datasets])
    full_dataset = pd.concat([X for [_,X] in q_datasets])

    # Add it to the list
    #q_datasets.append(["full",full_X,full_Y])  
    q_datasets.append(["full",full_dataset])
    
    return(q_datasets)

# Function for creating a tensorflow dataset from X and Y
def mr_tf_data(var_X,var_y,BATCH_SIZE = 4,SHUFFLE_BUFFER_SIZE = 100):
    
    # Get the labels in a format for softmax
    y_arr = var_y.to_numpy().astype(int)
    y_sm = tf.one_hot(y_arr,3)
    
    # Create the actual dataset and shuffle it        
    var_dataset = tf.data.Dataset.from_tensor_slices((var_X, y_sm))  
    var_dataset = var_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    
    return var_dataset

# Function for creating a tensorflow dataset from X and Y
def mr_tf_data_s(var_X,var_y,BATCH_SIZE = 4,SHUFFLE_BUFFER_SIZE = 100):
    
    # Get the labels in a format for softmax
    y_arr = var_y.to_numpy().astype(int)
    #y_sm = tf.one_hot(y_arr,3)
    
    # Create the actual dataset and shuffle it        
    var_dataset = tf.data.Dataset.from_tensor_slices((var_X, y_arr))  
    var_dataset = var_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    
    return var_dataset


# =============================================================================
# # Function that, given a test set maps them onto original datasets
# # col_list should either be text_cols or tq_cols
# # label list is always rate_cols
# 
# def mr_split_questions(full_df,X_test,col_list,lab_list=rate_cols):
#     
#     # Get the full data of the test set
#     test_df = full_df.iloc[X_test.index.values]
#     
#     # A very ugly loop to extract data, need to rework into efficient use of
#     # pandas when I have the timee
#     
#     
#     # Initialize the output
#     test_l = []
#     
#     # Loop over all questions and the corresponding rate columns
#     for cur_col,cur_lab in zip(col_list,lab_list):
#         # Select the whole row in the original dataframe that has corresponding value
#         # on the response or the QA
#         cur_df = full_df[full_df[cur_col].isin(X_test)].copy()
#         
#         # We need the following data:
#         # text response, expert score, date of birth and date of test
#         cur_df = cur_df[['DOB','DOT',cur_col,cur_lab]]
#         # Also add the current question
#         cur_df["Question"] = cur_col
#         # Rename columns for consistency
#         cur_df.rename(columns={cur_col:'Answer'},inplace=True)
#         cur_df.rename(columns={cur_lab:'Score'},inplace=True)
#                
#         # Get the year of birth from the date of birth
#         cur_df["YOB"] = cur_df["DOB"].astype(str).apply(mr_get_date)
#         # Get the year of text from the date of test
#         cur_df["YOT"] = cur_df["DOT"].astype(str).apply(mr_get_date)
#         # Calculate age as YOB - YOT (ugly fix due to messy date format in the 
#         # excels, will have to redo it later)
#         cur_df["Age"] = cur_df["YOT"] - cur_df["YOB"]
#         
#         
#         # The result of this is a table in the following format:
#         # Question; Answer; Score; DOB; DOT; YOB; YOT; Age
#         
#         # This format easily allows us to select based on question or age and then evaluate
#         
#         # Add the current question to the list
#         test_l.append(cur_df)
#         
#     # Merge all questions and return a dataframe
#     test_df = pd.concat(test_l, axis=0, ignore_index=True)
#     
#     return test_df
# =============================================================================

def mr_get_date(inp_date):
    
    # Find a pattern 20XX where X is a digit
    # Ignore the rest of the string
    year = re.search("20[0-9]{2}",inp_date)
    if year:
        # Return the year as a number
        return int(year.group(0))
    else:
        # If you can't find the pattern, return 0 for consistency
        return 0
        
    
def mr_get_data(folder_name):
    out_l = []
    for cur_q in text_cols:
        inp_file = folder_name + cur_q + ".xlsx"
        #cur_df = pd.read_excel(inp_file, index_col=None, header=0, encoding='utf-8')
        cur_df = pd.read_excel(inp_file, index_col=None, header=0)
        out_l.append([cur_q,cur_df])
    
    full_dataset = pd.concat([X for [_,X] in out_l])

    # Add it to the list
    out_l.append(["full",full_dataset])
    return(out_l)

def mr_get_qa_data(folder_name,t_cols=text_cols,list_qs=questions):
    out_l = []
    for cur_q, question in zip(t_cols, list_qs):
        inp_file = folder_name + cur_q + ".xlsx"
        #cur_df = pd.read_excel(inp_file, index_col=None, header=0, encoding='utf-8')
        cur_df = pd.read_excel(inp_file, index_col=None, header=0)
        cur_df['Answer'] = question + cur_df['Answer'].astype(str)
        
        out_l.append([cur_q,cur_df])
    
    full_dataset = pd.concat([X for [_,X] in out_l])

    # Add it to the list
    out_l.append(["full",full_dataset])
    return(out_l)