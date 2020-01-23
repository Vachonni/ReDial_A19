#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:02:04 2020


Creating CSV needed for fast.ai recommendation in an Item to User perspecive
(input items to get users to recommend to)


@author: nicholas
"""



import numpy as np
import pandas as pd

# Load Train (or Val) data

data = np.load('/Users/nicholas/ReDial_Utils/list_ReDial_ratings_ReDOrId2ConvOrId_TRAIN.npy', \
               allow_pickle=True)

#%%
# Load Item info (Abstract or KB)

dict_MovieText = np.load('/Users/nicholas/ReDial_CF2_MLP_dot/Data/DataReDial/str_MovieTitlesGenres_RT.npy', \
                             allow_pickle=True)



#%%

def Padding(l_user, l_ratings) -> list: 
    """
    Turn a list of users and a list of ratings into a "Padded" list of 
    tuples (ConvOrId, rating).
    Pad is 
        first item: (-2,0)                     - Second item indicates nb_movies_mentioned 
                                                 (usefull in user to movie setting, not here)
        fill up to lenght 610 with (-1, 0)    
    """
    
    end_pad = [(-1, 0)] * (610 - len(l_user))
    
    return [(-2, 0)] + list(zip(l_user, l_ratings)) + end_pad


#%%
for i in range(len(data)):
    # Convert 2 list of ratings to paddedlist of tuples
    data[i,2] = Padding(data[i,1], data[i,2])
    # Get the movie text (insure no nan)
    text = dict_MovieText[data[i,0]][0]
    if text=='': text = ' '
    data[i,1] = text

#%%

# Create CSV and save

df = pd.DataFrame(data)
df.columns = ['ConvID', 'text', 'ratings']
df.to_csv('/Users/nicholas/ReDial_A19/Data/ReDial/Item_TitleGenres/Train.csv', index=False)










