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
import re
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


### TO EXTRACT Movies Mentions with @
re_filmId = re.compile('@[0-9]{5,6}')




#%%

count_conv = 0 
max_nb_movie_rated = 19  # Evaluated elsewhere 


# Split data between train and valid.
# There are 9976 conversation in initial train dataset,
# 8978 remains in train, 998 in valid
indices = np.arange(9976)
np.random.shuffle(indices)
valid_indices = indices[:998]

train_data = []
valid_data = []

#%%

# For all conversations
for line in open(PATH, 'r'):
    
    # Data for this conversation
    data = []
    
    # Load conversation in a dict
    conv_dict = json.loads(line)
    
    # Get the conversation ID
    ConvID = int(conv_dict['conversationId'])

    # Get a list of all the ratings 
    l_ratings = []
    # First, get an non-empty movie form 
    # (seeker first, recommender 2nd, drop if none)
    if conv_dict['initiatorQuestions'] != []:
        questions_dict = conv_dict['initiatorQuestions']
    elif conv_dict['respondentQuestions'] != []:
        questions_dict = conv_dict['respondentQuestions']
    else:
        continue
    # Second, treat all movies in movie form
    for movieReDID, values in questions_dict.items():
        # If we know the rating (==2 would be did not say)
        if values['liked'] == 0 or values['liked'] == 1:
            # Get the movie uID
            movieuID = ReDID2uID[int(movieReDID)]
            # Get the rating according to the liked value
            rating = values['liked']
            l_ratings.append((movieuID, rating))

    # Get all texts 1x1     
    text_buffer = ""
    count_messages = 0
    for message in conv_dict['messages']:
        # scan for @ movies mentions
        all_movies = re_filmId.findall(message['text'])
        # If movies are mentionned 
        if all_movies != []:
            if text_buffer == "":   # First utterance has a movie mention
                text_buffer += message['text']+' '  # Add new text
                count_messages += len(all_movies)   # Count this mention
                continue            # But don't try to predict on empty str
            # return the last one, as integer, without '@'
            movie_found_ReDID = int(all_movies[-1][1:])
            movie_found = ReDID2uID[int(movie_found_ReDID)]
            # check if @ was rated, if so return ratings to come
            l_ratings_to_come = []
            for i, (movieuID, rating) in enumerate(l_ratings):
                if movie_found == movieuID:
                    l_ratings_to_come = l_ratings[(i+1):]
                    break
            if l_ratings_to_come != []:              
                # Fill to have list of same lenght
                fill_size = max_nb_movie_rated - len(l_ratings_to_come)
                filling = [(-1,0)] * fill_size
                l_ratings_to_come += filling
                # Add the number of movies mentioned
                l_ratings_to_come = [(-2,count_messages)] + l_ratings_to_come
                # Put list of ratings in text type (for .csv purposes in BertReco)
                l_ratings_to_come = str(l_ratings_to_come)               
                data.append([ConvID, text_buffer, \
                             l_ratings_to_come])
                count_messages += len(all_movies)
                
        # Go to next text
        text_buffer += message['text']+' '
        
            
    # Put data in the right set 
    if count_conv in valid_indices:
        valid_data += data
    else:
        train_data += data
        # if so: add convID+count_message, text_buffer, 
        
    count_conv += 1
    if count_conv > 7: break
   

#%%


# Creating a DataFrame and saving it

df = pd.DataFrame(train_data)
df.columns = ['ConvID', 'text', 'ratings']
df.to_csv('ChronoSmall.csv', index=False)




#%% self.date = date_obj.group().strip()












































