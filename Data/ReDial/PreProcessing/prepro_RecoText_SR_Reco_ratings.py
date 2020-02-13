#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 09:27:39 2019


ReDial Data to .csv with columns: ConvID - Text - Ratings ("[(UiD, rating)]")

FROM: ReDial Conversations 
TO: A CSV with columns: ConvID - Text - Ratings ("[(UiD, rating)]")


@author: nicholas
"""



###############
### IMPORTS
###############

import numpy as np
import json
import re
import pandas as pd
from ConversationClass import Conversation as Conv



###############
### PATHS
###############

# Path to the Conversation file.           
PATH = '/Users/nicholas/Desktop/data_10021/train_data.json'
# Path to dict of conversion from ReDialId to UniversalID
ReDID2uID_PATH = '/Users/nicholas/ReDial/DataProcessed/ReDID2uID.npy'
# Path to ReDial Chrono AE valid data (used to reproduce same train-valid split)
AE_valid_PATH = '/Users/nicholas/ReDial_E19/Data/ReDialRnGChronoVALID.json'



###############
### LOADING
###############

# Loading MovieLens in a DataFrame, only relevant columns
#df = pd.read_csv(ML_PATH, usecols = ['userId', 'movieId', 'rating'])

# Loading dict of conversion from ReDialId to UniversalID
#ReDID2uID = np.load(ReDID2uID_PATH).item()

# Loading dict of conversion from ReDID to ReDOrId
ReDID2ReDOrId = np.load('/Users/nicholas/ReDial_Utils/ReDiD_2_ReDOrId.npy').item()

## Loading a numpy array with [text with @, text with titles, _, _]
#text_data = np.load('/Users/nicholas/GitRepo/DialogueMovieReco/Data/DataConvFilm.npy')

# Loading ReDial Chrono AE valid data (used to reproduce same train-valid split)
with open (AE_valid_PATH, 'rb') as f:
    AE_valid_data = json.load(f)    




###############
### GLOBAL INIT
###############

count_conv = 0 
max_nb_movie_rated = 19  # Evaluated elsewhere 

# Same split as AE
valid_indices = list(set([int(c) for c, _, _, _ in AE_valid_data]))

train_data = []
valid_data = []



###############
### RE
###############

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




###############
### FUNCTIONS
###############
    

def MovieMentionsToRatings(movie_mentions, l_ratings):
    """
    NOTE: len of movies_mentions and recommended_ratings may not be equal
          (if no associated ratings found (e.g.:not in movie form, not rated)) 
    """
    # Remove @ 
    movie_mentions = [ m[1:] for m in movie_mentions]
    
    # Get associated ratings from movie form. Change to ReDOrId. Format:[(ReDOrId, rating)]
    recommended_ratings = []
    for m_mention in movie_mentions:
        for m_rated, rating in l_ratings:
            if m_mention == m_rated: 
                recommended_ratings.append((ReDID2ReDOrId[int(m_mention)], rating))
    
    return recommended_ratings



def RatingsToBertFormat(movies_message, movies_conv, l_ratings):
    
    recommended_ratings = MovieMentionsToRatings(movies_message, l_ratings)
   
    # Case where movies were mentioned but not rated
    if recommended_ratings == []:
        return []
    
    # Create the number of movies mentioned for this data point and add to ratings
    l_ratings_BERT_format = [(-2, len(movies_conv))] + recommended_ratings
    # Create filling to have list of same lenght
    fill_size = max_nb_movie_rated - len(recommended_ratings)
    filling = [(-1,0)] * fill_size
    l_ratings_BERT_format += filling

    return l_ratings_BERT_format



#%%




# For all conversations
for line in open(PATH, 'r'):
    
    
    ### LOCAL INIT
    
    # Create conversation object
    conv_obj = Conv(json.loads(line))
    
    # Get list of ratings in this conversation. If None (no movie form completed): drop
    l_ratings = conv_obj.GetRatings()
    if l_ratings == None: continue
    
    # Get messages by chucks of same speaker
    messages_by_chunks = conv_obj.Chunking()
    
    # Text buffer until Recommender mentions movies
    buffer = ""
    
    # Data for this conversation
    data = []
        

    
    
    
    ### TREATING MESSAGES (by chunks)    
    
    # Treat FIRST message chunk in this conversation
    buffer = re_filmId.sub(filmIdtoString, messages_by_chunks[0]['text']) # @ to film in NL
    unique_movie_mentions_message = set(re_filmId.findall(messages_by_chunks[0]['text']))
    unique_movie_mentions_conv = unique_movie_mentions_message
    
    
    # Treat REST of messages in this conversation
    for message in messages_by_chunks[1:]:
        
        unique_movie_mentions_message = set(re_filmId.findall(message['text']))
        undup_movie_mentions_message = unique_movie_mentions_message - \
                                        unique_movie_mentions_conv
       # print(undup_movie_mentions_message)
                                        
        # If it's message from Recommender and has new movie mentions that 
        # are rated in the movie form, add data point
        if message['speaker_token'] == 'R:: ' and undup_movie_mentions_message != set():
            ratigns_BERT_format = RatingsToBertFormat( \
                                                 undup_movie_mentions_message, 
                                                 unique_movie_mentions_conv,
                                                 l_ratings)
            # If ratings were found for mentioned movies
            if ratigns_BERT_format != []:
                data.append([conv_obj.id, buffer, ratigns_BERT_format])
        
        
        unique_movie_mentions_conv = unique_movie_mentions_conv.union( \
                                        undup_movie_mentions_message)
        buffer += ' ' + re_filmId.sub(filmIdtoString, message['text'])
          

            
            
    # Put data in the right set 
    if conv_obj.id in valid_indices:
        valid_data += data
    else:
        train_data += data

        
    count_conv += 1
   # if count_conv > 7: break
   



#%%

###############
### SAVING
###############
df = pd.DataFrame(train_data)
df.columns = ['ConvID', 'text', 'ratings']
df.to_csv('RecoText_SR_Reco_ratings_ReDOrId.csv', index=False)
















































