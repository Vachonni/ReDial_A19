#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:16:53 2019


Just copied the FAST-BERT example from: 
    https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384


@author: nicholas
"""

from pathlib import Path
import torch
# import apex


from fast_bert.data_cls import BertDataBunch
# from fast_bert.learner import BertLearner
# from fast_bert.metrics import accuracy






DATA_PATH = Path('./sample_data/multi_label_toxic_comments/data/')     # path for data files (train and val)
LABEL_PATH = Path('./sample_data/multi_label_toxic_comments/label/')  # path for labels file
MODEL_PATH=Path('../models/')    # path for model artifacts to be stored
LOG_PATH=Path('../logs/')       # path for log files to be stored



databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='train.csv',
                          val_file='val.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col=['toxic','severe_toxic','obscene','threat','insult','identity_hate'],
                          batch_size_per_gpu=16,
                          max_seq_length=512,
                          multi_gpu=True,
                          multi_label=True,
                          model_type='bert')





