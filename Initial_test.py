#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:16:53 2019


Just copied the FAST-BERT example from: 
    https://medium.com/huggingface/introducing-fastbert-a-simple-deep-learning-library-for-bert-models-89ff763ad384


@author: nicholas
"""

import torch
import apex

from pytorch_pretrained_bert.tokenization import BertTokenizer

from fast_bert.data import BertDataBunch
from fast_bert.learner import BertLearner
from fast_bert.metrics import accuracy