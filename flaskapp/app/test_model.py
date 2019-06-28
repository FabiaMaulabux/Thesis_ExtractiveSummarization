#!/usr/bin/env python
# coding: utf-8

# ##Testing

# In[2]:


# import mlflow
# from mlflow import keras
# keras_model = mlflow.keras.load_model("ANN model", run_id="af33541108df4983985c9f84160d0e22")
# model = keras_model


# In[3]:


#Importing data packages
import pandas as pd
import numpy as np
import mlflow
from mlflow import keras
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
import pickle
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import sys
import os
os.environ['KERAS_BACKEND']='tensorflow' # Why theano why not
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


# In[4]:


get_ipython().run_line_magic('run', '/Meetings/helpers/Data_Creation')


# In[5]:


get_ipython().run_line_magic('run', '/Meetings/helpers/Training_Model_2')


# In[6]:


def load_data2(df_original, df_new, start, stop):
  
  import pandas as pd
  for i in range(start, stop):
    transcript, summary = isolate(i, df_original)
    transcript2 = [BOW(item) for item in transcript]
    label = [Assign_Label(transcript, summary) for transcript in transcript]
    score = Assign_Score(transcript2)
    df_new = Append_DF(df_new, i + 1, transcript, label, score)
  print("Completed loading " + str(stop) + " meetings to dataframe " + str(df_new))
  return df_new


# In[7]:


def load_test(summary_table, start, stop):
  
  ami_df = table_to_df(summary_table)
  dftest = pd.DataFrame(columns=['Meeting ID', 'Transcript', 'Tokenized', 'Label','Length', 'Score'])
  dftest = load_data2(ami_df, dftest, start, stop)
  return dftest


# In[8]:


def predictions_df(meeting, t, ground_truth, df):
  
  df = df.append({'Summary_ID': meeting + 1, 'Prediction': t, 'Ground Truth': ground_truth}, ignore_index=True)
  return df


# In[9]:


def create_predictions(meeting, df, model, sentence_table, dftest, max_sequence, max_words, embedding_dim, summary_table, threshold):
  dftest  = dftest[dftest['Meeting ID'] == meeting + 1]
  dftest.iloc[:,3] = dftest.iloc[:,3].astype(str)

  #Transform meeting for input in model
  complete_df = table_to_df(sentence_table)
  complete_df = prep_df(complete_df)
  tokenizer, sequences, word_index = padsequence(complete_df.iloc[:,2], dftest.iloc[:,2], max_words, max_sequence)
  y_test = dftest.iloc[:,3]
  y_test = y_test.astype(str)

  macronum=sorted(set(y_test))
  macro_to_id = dict((note, number) for number, note in enumerate(macronum))

  def fun(i):
      return macro_to_id[i]

  y_test=y_test.apply(fun)

  labels = y_test
  
  #predictions = keras_model.predict(data)
  predictions = model.predict(sequences)
  #Create transcript based on predictions and threshold
  scores = ['none'] * len(predictions)
  labels = [0] * len(predictions)
  
  for i in range(len(predictions)):
    scores[i] = predictions[i][1]
    if scores[i] > threshold:
      labels[i] = 1
  
  dftest['Predicted'] = labels
  transcript = dftest.Transcript[dftest.Predicted == 1]
  
  #Concatenate all transcript sentences
  transcript = pd.DataFrame(transcript)
  t = ""
  
  for i in range(len(transcript)):
    t = t + str(transcript.iloc[i,0]) + " "
  
  #Grab original summary
  ami_df = table_to_df(summary_table)
  ground_truth = ami_df.iloc[meeting,3]
  df = predictions_df(meeting, t, ground_truth, df)
  return df


# In[10]:


def create_predictions_meta(meeting, df, model, sentence_table, dftest, max_sequence, max_words, embedding_dim, summary_table, threshold):
  dftest  = dftest[dftest['Meeting ID'] == meeting + 1]
  dftest.iloc[:,3] = dftest.iloc[:,3].astype(str)

  #Transform meeting for input in model
  complete_df = table_to_df(sentence_table)
  complete_df = prep_df(complete_df)
  tokenizer, sequences, word_index = padsequence(complete_df.iloc[:,2], dftest.iloc[:,2], max_words, max_sequence)
  y_test = dftest.iloc[:,3]
  y_test = y_test.astype(str)

  macronum=sorted(set(y_test))
  macro_to_id = dict((note, number) for number, note in enumerate(macronum))

  def fun(i):
      return macro_to_id[i]

  y_test=y_test.apply(fun)

  labels = y_test
  #predictions = keras_model.predict(data)
  #meta_test = np.array(dftest.iloc[:,4:6], dtype=object)

  #x_test = sequences.tolist()
#   print(type(meta_test))
#   print(type(sequences))
  predictions = model.predict([dftest.iloc[:,4:6], sequences])
  #Create transcript based on predictions and threshold
  scores = ['none'] * len(predictions)
  labels = [0] * len(predictions)
  
  for i in range(len(predictions)):
    scores[i] = predictions[i][1]
    if scores[i] > threshold:
      labels[i] = 1
  
  dftest['Predicted'] = labels
  transcript = dftest.Transcript[dftest.Predicted == 1]
  
  #Concatenate all transcript sentences
  transcript = pd.DataFrame(transcript)
  t = ""
  
  for i in range(len(transcript)):
    t = t + str(transcript.iloc[i,0]) + " "
  
  #Grab original summary
  ami_df = table_to_df(summary_table)
  ground_truth = ami_df.iloc[meeting,3]
  df = predictions_df(meeting, t, ground_truth, df)
  return df


# In[11]:


import rouge
log_array = [""] * 4

def prepare_results(p, r, f, metric):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def calculate_rouge(hypotheses, ground_truth):
  log_array = [""] * 4
  i = 0 
  for aggregator in ['Avg', 'Best', 'Individual']:
    print('Evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=['rouge-n'
                                      , 'rouge-l', 'rouge-w'
                                    ],
                           max_n=2,
                           limit_length=False,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

    all_hypothesis = hypotheses
    all_references = ground_truth

    scores = evaluator.get_scores(all_hypothesis, all_references)

    for metric, results in sorted(scores.items(), key=lambda x: x[0]):
        if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
            for hypothesis_id, results_per_ref in enumerate(results):
                nb_references = len(results_per_ref['p'])
                for reference_id in range(nb_references):
                    print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                    print('\t' + prepare_results(results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
            print()
        else:
              print(prepare_results(results['p'], results['r'], results['f'], metric))
              log_array[i] = prepare_results(results['p'], results['r'], results['f'], metric)
              i += 1
              
    print()
    return log_array


# In[12]:


def test_model(model, dftest, reload, summary_table, sentence_table, start, stop, max_sequence, max_words, embedding_dim, threshold):
  if reload == True:
    dftest = load_test(summary_table, start, stop)
  
  df_prediction = pd.DataFrame(columns=['Summary_ID', 'Prediction', 'Ground Truth'])

  for i in range(start,stop):
    df_prediction = create_predictions(i, df_prediction, model, sentence_table, dftest, max_sequence, max_words, embedding_dim, summary_table, threshold)
  
  hypotheses = df_prediction.iloc[:,1].values
  ground_truth = df_prediction.iloc[:,2].values
  
  log_array = calculate_rouge(hypotheses, ground_truth)
  
  return dftest, df_prediction, log_array


# In[13]:


def test_model_meta(model, dftest, reload, summary_table, sentence_table, start, stop, max_sequence, max_words, embedding_dim, threshold):
  if reload == True:
    dftest = load_test(summary_table, start, stop)
  
  df_prediction = pd.DataFrame(columns=['Summary_ID', 'Prediction', 'Ground Truth'])

  for i in range(start,stop):
    df_prediction = create_predictions_meta(i, df_prediction, model, sentence_table, dftest, max_sequence, max_words, embedding_dim, summary_table, threshold)
  
  hypotheses = df_prediction.iloc[:,1].values
  ground_truth = df_prediction.iloc[:,2].values
  
  log_array = calculate_rouge(hypotheses, ground_truth)
  
  return dftest, df_prediction, log_array

