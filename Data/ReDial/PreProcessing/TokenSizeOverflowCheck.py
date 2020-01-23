#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 07:33:40 2019


Just checking the max lenght of ReDial Conversations when tokenized for Bert Reco


@author: nicholas
"""

import pandas as pd
from transformers import BertTokenizer, BertModel
import torch


#%%


dataPATH = '/Users/nicholas/ReDial_A19/Data/ReDial/ChronoTextSRGenres/Val.csv'

df = pd.read_csv(dataPATH)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#%%
count = 0
for conv in df['text']:
  #  print(conv, tokenizer.tokenize(conv), len(tokenizer.tokenize(conv)))
    if len(tokenizer.tokenize(conv)) > 512:
        count += 1
 #   if count > 3: break


#%%
 
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple