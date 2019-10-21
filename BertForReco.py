#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:16:53 2019



BERT adapted for recommendation


From FAST-BERT example at: 
    https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384
    Up to date code is in: fast_bert-1.4.2.tar.gz
    
    
    
    When using Bert for Recommendation:
        use ONE label column in the data with header 'ratings' and 
        label_col = ['ratings']. 
        'ratings' column shoul have all same number of 
        examples (fill with (-1, 0) if necessary).
    


@author: nicholas

"""

from pathlib import Path
import torch
# import apex

nb_items = 300




import argparse

parser = argparse.ArgumentParser(description='Bert for recommendation')

parser.add_argument('--log_path', type=str, metavar='', default='.',\
                    help='Path where all infos will be saved.')
parser.add_argument('--data_path', type=str, metavar='', default='.', \
                    help='Path to datasets')
parser.add_argument('--epoch', type=int, metavar='', default=1, \
                    help='Qt of epoch')

args = parser.parse_args()





######################
###                ###
###      DATA      ###
###                ###
######################

from data_reco import BertDataBunch


DATA_PATH = Path(args.data_path + '/Data/ReDial/')     # path for data files (train and val)
LABEL_PATH = Path(args.data_path + '/Data/multi_label_toxic_comments/label/')  # path for labels file
MODEL_PATH = Path(args.log_path)    # path for model artifacts to be stored
LOG_PATH = Path(args.log_path)       # path for log files to be stored


databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='ChronoTrain.csv',
                          val_file='ChronoVal.csv',
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



###########################  SHOUULD BE ADDED TO METRICS ###########################

import numpy as np 
from statistics import mean
topx = 100
max_movies_mentions = 10           # max number of movies mentions considered

# DCG (Discounted Cumulative Gain)   
# Needed to compare rankings when the numbre of item compared are not the same
# and/or when relevance is not binary

def _DCG(v, top):
    """
    V is vector of ranks, lowest is better
    top is the max rank considered 
    Relevance is 1 if items in rank vector, 0 else
    """
    
    discounted_gain = 0
    
    for i in np.round(v):
        if i <= top:
            discounted_gain += 1/np.log2(i+1)

    return round(discounted_gain, 2)

def _nDCG(v, top, nb_values=0):
    """
    DCG normalized with what would be the best evaluation.
    
    nb_values is the max number of good values there is. If not specified or bigger 
    than top, assumed to be same as top.
    """
    if nb_values == 0 or nb_values > top: nb_values = top
    dcg = _DCG(v, top)
    idcg = _DCG(np.arange(nb_values)+1, top)
    
    return round(dcg/idcg, 2)
    
# RR (Reciprocal Rank)
    
# Gives a value in [0,1] for the first relevant item in list.
# 1st = 1 and than lower until cloe to 0.
# Only consern with FIRST relevant item in the list.
    
def RR(v):
    return 1/np.min(v)
    
def Ranking(all_values, values_to_rank, topx = 0):
    """
    Takes 2 numpy array and return, for all values in values_to_rank,
    the ranks
    """    
    # If topx not mentionned (no top), it's for all the values
    if topx == 0: topx = len(all_values)
    
    # Initiate ranks
    ranks = np.zeros(len(values_to_rank))
    
    for i,v in enumerate(values_to_rank):
        ranks[i] = len(all_values[all_values > v]) + 1
      
    return ranks


def ndcg(logits, labels):
    """
    Bert metric, average of all batches
    """
    all_ndcg = np.zeros(len(logits))
    for i in range(len(logits)):
        idx_with_positive_mention = labels[i].nonzero().flatten().tolist()
        values_to_rank = logits[i][idx_with_positive_mention]
        ranks = Ranking(logits[i], values_to_rank, topx)
        all_ndcg[i] = _nDCG(ranks, topx, len(values_to_rank))
    
    return all_ndcg.mean() 


def ndcg_chrono(logits, labels, l_qt_movies_mentioned):
    """
    Bert metric, ndcg by qt of movies mentioned before prediction, 
    average over all batches
    """ 
    ndcg_by_qt_movies_mentioned = [[] for i in range(max_movies_mentions + 1)]    
    for i in range(len(logits)):
        idx_with_positive_mention = labels[i].nonzero().flatten().tolist()
        values_to_rank = logits[i][idx_with_positive_mention]
        ranks = Ranking(logits[i], values_to_rank, topx) 
        # Get qt of movies mentioned and add ndcg to the right list
        qt_movies_mentioned_this_example = l_qt_movies_mentioned[i]
        if qt_movies_mentioned_this_example > max_movies_mentions:
            qt_movies_mentioned_this_example = max_movies_mentions
        ndcg_by_qt_movies_mentioned[qt_movies_mentioned_this_example].append(\
                                        _nDCG(ranks, topx, len(values_to_rank)))
    # Take the mean
    mean_ndcg_by_qt_movies_mentioned = []
    for l_ndcg in ndcg_by_qt_movies_mentioned:
        if l_ndcg == []: mean_ndcg_by_qt_movies_mentioned.append(0)
        else: mean_ndcg_by_qt_movies_mentioned.append(mean(l_ndcg))
    
    return mean_ndcg_by_qt_movies_mentioned    
                      
 
    
# More possibilities    
#    if ranks.sum() == 0: print('warning, should always be at least one rank')
#    return ranks, ranks.mean(), round(float((1/ranks).mean()),4), RR(ranks), ndcg
    
###########################  SHOUULD BE ADDED TO METRICS ###########################
    

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
#logger.setLevel('INFO')

logger.info('will my logger print?')

device_cuda = torch.device("cuda")
metrics = [{'name': 'NDCG_CHRONO', 'function': ndcg_chrono}]

print('hello')

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=MODEL_PATH,
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

learner.fit(epochs=args.epoch,
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
    print('\n\n',texts[i],'\n', predictions[i][:5])










































