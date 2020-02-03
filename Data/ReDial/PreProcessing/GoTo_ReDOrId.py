#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:25:45 2020


From UiD to ReDOrID (ReDial id's from 0 to 7012)

ex for one data point:
    
    391,"S:: Hi there, how are you? I'm looking for movie recommendations R:: I am doing okay. What kind of movies do you like? ","[(-2, 0), (4819, 1), (354, 1), (4090, 1), (1314, 1), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0), (-1, 0)]"


@author: nicholas
"""



import pandas as pd
import numpy as np

from ast import literal_eval



# Load dict of conversion from UiD to ReDOrId
UiD_2_ReDOrId= np.load('/Users/nicholas/ReDial_Utils/UiD_2_ReDOrId.npy', allow_pickle=True).item()

# Load csv in numpy object array
UiD_data = pd.read_csv('/Users/nicholas/ReDial_A19/Data/ReDial/NEXTTextSRGenres/Val.csv'\
                       ).values


#%%

                       
# For each "user", change movie id from UiD to ReDOrId

for i in range(len(UiD_data)):
    str_ratings = UiD_data[i,2]
    list_ratings = literal_eval(str_ratings)
    # Treat all ratings, except first that indicates number of movies mentionned
    for j in range(1, len(list_ratings)):
        (old_idx, rating) = list_ratings[j]
        if old_idx == -1: break
        new_idx = UiD_2_ReDOrId[old_idx]
        list_ratings[j] = (new_idx, rating)
    # Put it back as a string
    str_ratings = str(list_ratings)
    UiD_data[i,2] = str_ratings
    
#%%

# Save it in csv
    
df = pd.DataFrame(UiD_data)
df.columns = ['ConvID', 'text', 'ratings']
df.to_csv('/Users/nicholas/ReDial_A19/Data/ReDial/NEXTTextSRGenres_ReDOrId/Val.csv', index=False)
