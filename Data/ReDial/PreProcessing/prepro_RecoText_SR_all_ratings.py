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
# Path to ReDial Chrono AE valid data (used to reproduce same train-valid split)
AE_valid_PATH = '/Users/nicholas/ReDial_E19/Data/ReDialRnGChronoVALID.json'


### LOADING

# Loading MovieLens in a DataFrame, only relevant columns
#df = pd.read_csv(ML_PATH, usecols = ['userId', 'movieId', 'rating'])

# Loading dict of conversion from ReDialId to UniversalID
ReDID2uID = np.load(ReDID2uID_PATH).item()

## Loading a numpy array with [text with @, text with titles, _, _]
#text_data = np.load('/Users/nicholas/GitRepo/DialogueMovieReco/Data/DataConvFilm.npy')

# Loading ReDial Chrono AE valid data (used to reproduce same train-valid split)
with open (AE_valid_PATH, 'rb') as f:
    AE_valid_data = json.load(f)    

#%%

count_conv = 0 
max_nb_movie_rated = 19  # Evaluated elsewhere 

#
## Split data between train and valid.
## There are 9976 conversation in initial train dataset,
## 8978 remains in train, 998 in valid
#indices = np.arange(9976)
#np.random.shuffle(indices)
#valid_indices = indices[:998]
#
#train_data = []
#valid_data = []


# Same split as AE
valid_indices = list(set([int(c) for c, _, _, _ in AE_valid_data]))

train_data = []
valid_data = []

#%%

### TO EXTRACT Movies Mentions with @
re_filmId = re.compile('@[0-9]{5,6}')
### TO EXTRACT Date after movie title
re_date = re.compile(' *\([0-9]{4}\)| *$') 

#ls_film_ids = []
#ls_film = []
df_filmID = pd.read_csv('/Users/nicholas/Desktop/data_10021/movies_db.csv')


# Function called by .sub
# A re method to substitue matching pattern in a string
# Here, substitute film ID with film NL title
# This code also creates list of film Ids and list of film str with (dates)

def filmIdtoString(match):
    filmId = int(match.group()[1:])                  # Remove @ and from str to int
    if df_filmID[df_filmID['movieId'] == filmId].empty:
        print('Unknow movie', filmId)
        film_str = str(filmId)          # Put film ID since can't find title
    else:
        film_str_with_date = df_filmID[df_filmID['movieId'] == filmId][' movieName'].values[0]
        film_str = re_date.sub("", film_str_with_date)  # Remove date for more NL
  #  ls_film.append(film_str)
  #  print(film_str)
    
    return film_str





#%%

# For all conversations
for line in open(PATH, 'r'):
    
    # Data for this conversation
    data = []
    
    # Load conversation in a dict
    conv_dict = json.loads(line)
    
    # Get the conversation ID
    ConvID = int(conv_dict['conversationId'])
    # Get Seeker ID
    seekerID = conv_dict['initiatorWorkerId']
    # Get Recommender ID
    recommenderID = conv_dict['respondentWorkerId']

    
    # First, get an non-empty movie_form 
    # (seeker first, recommender 2nd, drop if none)
    if conv_dict['initiatorQuestions'] != []:
        movie_form = conv_dict['initiatorQuestions']
    elif conv_dict['respondentQuestions'] != []:
        movie_form = conv_dict['respondentQuestions']
    else:
        continue
    
    # Get a list of all the ratings 
    l_ratings = []
    # Second, retreive all movies in movie form with rating provided
    for movieReDID, values in movie_form.items():
        # If we know the rating (==2 would be did not say)
        if values['liked'] == 0 or values['liked'] == 1:
            # Get the movie uID
            movieuID = ReDID2uID[int(movieReDID)]
            # Get the rating according to the liked value
            rating = values['liked']
            l_ratings.append((movieuID, rating))


    # Get all texts 1x1     
    text_buffer = ""
    unique_movies_this_Conv = []
    previous_speaker = None
    
    # Identify who speaks first
    if conv_dict['messages'][0]['senderWorkerId'] == seekerID:
        speaker_token = 'S:: ' 
    else: speaker_token = 'R:: ' 
    
    
    
    # Treat all messages in the conversation
    for message in conv_dict['messages']: 

        # If speaker changes to Recommender, add data point with info already there
        if message['senderWorkerId'] == recommenderID and previous_speaker == 'S:: ':  
            # Create the number of movies mentioned for this data point and add to ratings
            l_ratings_BERT_format = [(-2, len(unique_movies_this_Conv))] + l_ratings
            # Create filling to have list of same lenght
            fill_size = max_nb_movie_rated - len(l_ratings)
            filling = [(-1,0)] * fill_size
            l_ratings_BERT_format += filling
            # Put list of ratings in text type (for .csv purposes in BertReco)           
            data.append([ConvID, text_buffer, str(l_ratings_BERT_format)])

        """ Prepare next round """
        # Update speaker token
        if message['senderWorkerId'] == seekerID:
            speaker_token = 'S:: ' 
        else: speaker_token = 'R:: '     
        previous_speaker = speaker_token
            
        # Update buffer 
        message_in_NL = re_filmId.sub(filmIdtoString, message['text']) 
        text_buffer += speaker_token + message_in_NL + ' '  # Add new text with NL title     
        
        # Update movies mentions in this message. Insure uniqueness.
        unique_movies_this_message = list(set(re_filmId.findall(message['text'])))
        unique_movies_this_Conv = list(set(unique_movies_this_Conv + \
                                           unique_movies_this_message))
            
            
    # Put data in the right set 
    if ConvID in valid_indices:
        valid_data += data
    else:
        train_data += data
        # if so: add convID+count_message, text_buffer, 
        
    count_conv += 1
  #  if count_conv > 7: break
   



#%%

# Creating a DataFrame and saving it

df = pd.DataFrame(train_data)
df.columns = ['ConvID', 'text', 'ratings']
df.to_csv('RecoText_SR_all_ratings.csv', index=False)




#%% self.date = date_obj.group().strip()












































