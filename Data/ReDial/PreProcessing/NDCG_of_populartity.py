#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:00:12 2019


Finding NDCG of random and most popular on training set for valid set.


@author: nicholas
"""


import pandas as pd
from ast import literal_eval
import numpy as np
import torch


# Get the ratings

train_df = pd.read_csv('/Users/nicholas/ReDial_A19/Data/ReDial/ChronoTextSRGenres/Train.csv')
valid_df = pd.read_csv('/Users/nicholas/ReDial_A19/Data/ReDial/ChronoTextSRGenres/Val.csv')



#%%


###########################################
###                                     ###
###     Functions to evaluate NDCG      ###
###                                     ###
###########################################


from statistics import mean
topx = 100
max_movies_mentions = 8           # max number of movies mentions considered


# DCG (Discounted Cumulative Gain)   
# Needed to compare rankings when the numbre of item compared are not the same
# and/or when relevance is not binary

def _DCG(v, top):
    """
    V is vector of ranks, lowest is better
    top is the max rank considered 
    Relevance is 1 if items in rank vector, 0 else
    """
    
    discounted_gain = 0
    
    for i in np.round(v):
        if i <= top:
            discounted_gain += 1/np.log2(i+1)

    return round(discounted_gain, 2)


def _nDCG(v, top, nb_values=0):
    """
    DCG normalized with what would be the best evaluation.
    
    nb_values is the max number of good values there is. If not specified or bigger 
    than top, assumed to be same as top.
    """
    if nb_values == 0 or nb_values > top: nb_values = top
    dcg = _DCG(v, top)
    idcg = _DCG(np.arange(nb_values)+1, top)
    
    return round(dcg/idcg, 2)
    
# RR (Reciprocal Rank)
    
# Gives a value in [0,1] for the first relevant item in list.
# 1st = 1 and than lower until cloe to 0.
# Only consern with FIRST relevant item in the list.
    

def RR(v):
    return 1/np.min(v)
    
def Ranking(all_values, values_to_rank, topx = 0):
    """
    Takes 2 numpy array and return, for all values in values_to_rank,
    the ranks
    """    
    # If topx not mentionned (no top), it's for all the values
    if topx == 0: topx = len(all_values)
    
    # Initiate ranks
    ranks = np.zeros(len(values_to_rank))
    
    for i,v in enumerate(values_to_rank):
        ranks[i] = len(all_values[all_values > v]) + 1
      
    return ranks


def ndcg(logits, labels):
    """
    Bert metric, average of all batches
    """
    all_ndcg = np.zeros(len(logits))
    for i in range(len(logits)):
        idx_with_positive_mention = labels[i].nonzero().flatten().tolist()
        values_to_rank = logits[i][idx_with_positive_mention]
        ranks = Ranking(logits[i], values_to_rank, topx)
        all_ndcg[i] = _nDCG(ranks, topx, len(values_to_rank))
    
    return all_ndcg.mean() 


def ndcg_chrono(logits, labels, l_qt_movies_mentioned):
    """
    Bert metric, ndcg by qt of movies mentioned before prediction, 
    average over all batches
    """ 
    ndcg_by_qt_movies_mentioned = [[] for i in range(max_movies_mentions + 1)]    
    for i in range(len(logits)):
        idx_with_positive_mention = labels[i].nonzero().flatten().tolist()
        values_to_rank = logits[i][idx_with_positive_mention]
        ranks = Ranking(logits[i], values_to_rank, topx) 
        # Get qt of movies mentioned and add ndcg to the right list
        qt_movies_mentioned_this_example = l_qt_movies_mentioned[i]
        if qt_movies_mentioned_this_example > max_movies_mentions:
            # qt_movies_mentioned_this_example = max_movies_mentions
            continue
        ndcg_by_qt_movies_mentioned[qt_movies_mentioned_this_example].append(\
                                        _nDCG(ranks, topx, len(values_to_rank)))
    # Take the mean
    mean_ndcg_by_qt_movies_mentioned = []
    for l_ndcg in ndcg_by_qt_movies_mentioned:
        if l_ndcg == []: mean_ndcg_by_qt_movies_mentioned.append(0)
        else: mean_ndcg_by_qt_movies_mentioned.append(mean(l_ndcg))
    
    return mean_ndcg_by_qt_movies_mentioned    
 
    


#%%


# Evaluate popularity of movies according to train set

train_popularity = torch.zeros(48272)
count = 0

for str_ratings in train_df['ratings']:
    l_ratings = literal_eval(str_ratings)
    # Remove qt_movies_mentioned (the first rating in list)
    l_ratings = l_ratings[1:]
    for (movie, rating) in l_ratings:
        if movie == -1: break    # reached the filling
        else: train_popularity[movie] += rating
            
 #   print(l_ratings)
    count += 1
 #   if count >5: break



#%%


# Get the validation ratings
 
valid_ratings = torch.zeros(valid_df.shape[0], 48272) 
qt_movies_mentioned = []

for i, str_ratings in enumerate(valid_df['ratings']):
    l_ratings = literal_eval(str_ratings)
    # First rating has qt_movies_mentioned
    qt_movies_mentioned.append(l_ratings[0][1])
    l_ratings = l_ratings[1:]
    for (movie, rating) in l_ratings:
        if movie == -1: break    # reached the filling
        else: valid_ratings[i, movie] = rating
        
    count += 1
  #  if count >5: break
 

#%%

# Get train popularity in format of valid (Repeat train popularity as many time as valid size) 
  
train_populatity_repeat = torch.cat(valid_df.shape[0]*[train_popularity.unsqueeze(0)])


#%%

# NDCG as in output of model was always the most popular movies of the train set
ndcg_chrono(train_populatity_repeat, valid_ratings, qt_movies_mentioned)


#%%

# NDCG as in output of model was random
ndcg_chrono(torch.rand(valid_df.shape[0], 48272),valid_ratings, qt_movies_mentioned)



































































































































