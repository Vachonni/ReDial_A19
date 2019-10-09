#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 16:31:43 2019

@author: nicholas
"""

from pathlib import Path
from fast_bert.prediction import BertClassificationPredictor


MODEL_PATH = Path('./model_out/')
LABEL_PATH = Path('./sample_data/multi_label_toxic_comments/label/')

predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path=LABEL_PATH, # location for labels.csv file
				multi_label=True,
				model_type='bert',
				do_lower_case=True)


# Batch predictions
texts = [
        'I really love the Netflix original movies',
		 'Jerk me jolly. I have a big penis, not to mention the species is thriving.',
         'People watching Netflix movies should die',
         'You are a big hairy ape like mamith.'
         ]

multiple_predictions = predictor.predict_batch(texts)

for i in range(len(multiple_predictions)):
    print('\n\n',texts[i],'\n', multiple_predictions[i])
    