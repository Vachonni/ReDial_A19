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




######################
###                ###
###      DATA      ###
###                ###
######################


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
                          batch_size_per_gpu=8,
                          max_seq_length=512,
                          multi_gpu=True,
                          multi_label=True,
                          model_type='bert')




######################
###                ###
###    LEARNER     ###
###                ###
######################


from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
#logger.setLevel('INFO')

logger.info('will my logger print?')

device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

print('hello')

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir='.',
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=False,
						is_fp16=False,
						multi_label=True,
						logging_steps=0)


#%%

######################
###                ###
###     TRAIN      ###
###                ###
######################

print('hello again')

learner.fit(epochs=100,
			lr=6e-5,
			validate=True,        	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="lamb")


#%%

######################
###                ###
###     SAVE       ###
###                ###
######################


learner.save_model()


#%%



texts = ['I really love the Netflix original movies',
		 'this movie is not worth watching']
predictions = learner.predict_batch(texts)


print('\n\n', predictions[0],'\n\n', predictions[1], '\n\n')



















































