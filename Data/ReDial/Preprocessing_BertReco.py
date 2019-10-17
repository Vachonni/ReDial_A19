#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:27:39 2019


ReDial Data to .csv with columns: ConvID - Text - Ratings ("[(UiD, rating)]")

FROM: ReDial Conversations 
TO: A CSV with columns: ConvID - Text - Ratings ("[(UiD, rating)]")


@author: nicholas
"""



### IMPORTS

import numpy as np
import json
import pandas as pd



### SETTINGS

# Path to the Conversation file.           
PATH = '/Users/nicholas/Desktop/data_10021/train_data.json'
# Path to dict of conversion from ReDialId to UniversalID
ReDID2uID_PATH = '/Users/nicholas/ReDial/DataProcessed/ReDID2uID.npy'



### LOADING

# Loading MovieLens in a DataFrame, only relevant columns
#df = pd.read_csv(ML_PATH, usecols = ['userId', 'movieId', 'rating'])

# Loading dict of conversion from ReDialId to UniversalID
ReDID2uID = np.load(ReDID2uID_PATH).item()

## Loading a numpy array with [text with @, text with titles, _, _]
#text_data = np.load('/Users/nicholas/GitRepo/DialogueMovieReco/Data/DataConvFilm.npy')


#%%


# Initiate list with Header   
ReDialRatings2List = []
count = 0 
max_nb_movie_rated = 19

#%%

# For all conversations
for line in open(PATH, 'r'):
    # Load conversation in a dict
    conv_dict = json.loads(line)
    
    # Check if the conv_dict and text_data correspond
#    first_text = conv_dict['messages'][0]['text']
#    assert first_text == text_data[count, 0][:len(first_text)], \
#                'not same text {}'.format(text_data[count, 0][:len(first_text)])    
    
    # Get the conversation ID
    ConvID = int(conv_dict['conversationId'])
    
    # Get the text
    text = ' '.join([message['text'] for message in conv_dict['messages']])
    
    # Get an non-empty movie form (seeker first, recommender 2nd, drop if none)
    if conv_dict['initiatorQuestions'] != []:
        questions_dict = conv_dict['initiatorQuestions']
    elif conv_dict['respondentQuestions'] != []:
        questions_dict = conv_dict['respondentQuestions']
    else:
        continue
   
#    # Finds max number of movie rated
#    qt_movies_rated = len(questions_dict)
#    if qt_movies_rated > max_nb_movie_rated: max_nb_movie_rated = qt_movies_rated 
    
    l_ratings = []
    # For all movies in movie form
    for movieReDID, values in questions_dict.items():
        # If we know the rating (==2 would be did not say)
        if values['liked'] == 0 or values['liked'] == 1:
            # Get the movie uID
            movieuID = ReDID2uID[int(movieReDID)]
            # Get the rating according to the liked value
            rating = values['liked']
            l_ratings.append((movieuID, rating))
            
    # Fill to have list of same lenght
    fill_size = max_nb_movie_rated - len(l_ratings)
    filling = [(-1,0)] * fill_size
    l_ratings = l_ratings + filling
    
    # Put list of ratings in text type (for .csv purposes in BertReco)
    l_ratings = str(l_ratings)
    ReDialRatings2List.append([ConvID, text, l_ratings])
        
    count += 1
 #   if count > 7: break
   

#%%
import random
# For train set, split it to get valid set
random.shuffle(ReDialRatings2List)
train_size = int(len(ReDialRatings2List)*0.9)
trainset = ReDialRatings2List[:train_size]
validset = ReDialRatings2List[train_size:]

#%%


# Creating a DataFrame

df = pd.DataFrame(validset)
df.columns = ['ConvID', 'text', 'ratings']
df.to_csv('Val.csv', index=False)



