#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:16:53 2019



BERT adapted for recommendation


From FAST-BERT example at: 
    https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384
    Up to date code is in: fast_bert-1.4.2.tar.gz



@author: nicholas

"""

from pathlib import Path
import torch
# import apex

nb_items = 300


######################
###                ###
###      DATA      ###
###                ###
######################

from data_reco import BertDataBunch


DATA_PATH = Path('./sample_data/multi_label_toxic_comments/data/')     # path for data files (train and val)
LABEL_PATH = Path('./sample_data/multi_label_toxic_comments/label/')  # path for labels file
MODEL_PATH=Path('../models/')    # path for model artifacts to be stored
LOG_PATH=Path('../logs/')       # path for log files to be stored


databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='RECOsmallDATA.csv',
                          val_file='RECOsmallDATA.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col=['ratings'],
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


from learner_reco import BertLearner
from fast_bert.metrics import accuracy_thresh
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
#logger.setLevel('INFO')

logger.info('will my logger print?')

device_cuda = torch.device("cpu")
metrics = [{'name': 'accuracy_tresh', 'function': accuracy_thresh}]

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

learner.fit(epochs=1,
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


texts = [
        'I really love the Netflix original movies',
		 'Jerk me jolly. I have a big penis, not to mention the species is thriving.',
         'People watching Netflix movies should die',
         'You are a big hairy ape like mamith.'
         ]
predictions = learner.predict_batch(texts)


for i in range(len(predictions)):
    print('\n\n',texts[i],'\n', predictions[i])










































