# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 21:43:41 2021

@author: prabal
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#from transformers import BertPreTrainedModel, BertModel
#from transformers import AutoConfig, AutoTokenizer

from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import seaborn as sns

from dataloader import Excerpt_Dataset, get_dfs
#from model import *
#from solver import *
#from checkpoint_load import *
#
MAX_LEN_TRAIN = 120
MAX_LEN_VALID = 60
MAX_LEN_TEST = 60
BATCH_SIZE = 4
LR = 1e-5
NUM_EPOCHS = 10
NUM_THREADS = 1  ## Number of threads for collecting dataset
MODEL_NAME = 'bert-base-uncased'

di = dict()
di["usnli_train"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/train.csv"
di["usnli_dev"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/dev.csv"
di["usnli_test"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/test.csv"

# Python program to convert
# JSON file to CSV


import json
import csv

import pandas as pd

def get_df_snli(path = "snli_1.0_train.txt",save_csv = False):
    df = pd.read_csv(path, usecols = ['gold_label','sentence1', 'sentence2'])
    df = df.rename(columns={'sentence1': 'pre', 'sentence2': 'hyp'})
#    df['gold_label_num'] = df.gold_label.apply(to_rating)
#    df.head()
    if save_csv:
        df.to_csv('snli_train.csv')
    return df


#df = pd.read_csv("snli_1.0_train.txt", sep="\t", usecols = ['gold_label','sentence1', 'sentence2']) #train
#df = pd.read_csv("snli_1.0_test.txt", sep="\t", usecols = ['gold_label','sentence1', 'sentence2'])  #test
#df = pd.read_csv("snli_1.0_dev.txt", sep="\t", usecols = ['gold_label','sentence1', 'sentence2'])  #dev
def to_rating(rating):
  if rating == 'neutral':
      return 1
  elif rating == 'entailment':
      return 0
  else: 
      return 2

#df['gold_label_num'] = df.gold_label.apply(to_rating)
#df.head()
#df.to_csv('snli_train.csv')
#df.to_csv('snli_test.csv')
#df.to_csv('snli_dev.csv')

di = dict()
di["usnli_train"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/train.csv"
di["usnli_dev"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/dev.csv"
di["usnli_test"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/test.csv"

df_train = pd.read_csv(di['usnli_train'])
df_dev = pd.read_csv(di['usnli_dev'])
df_test = pd.read_csv(di['usnli_test'])

def avg_uNLI(df,inf = 'ENT'):
    if inf == 'ENT':
        df_inf = df[df['nli']==2]
    elif inf == 'NEU':
        df_inf = df[df['nli']==1]
    else:
        df_inf = df[df['nli']==0]
    return np.mean(df_inf.unli)

def snli2unli(di,path_snli = "snli_1.0_train.txt",type_ = 'train'):
    df_snli = get_df_snli(path_snli)
    name = 'usnli_' + type_
    df_unli = pd.read_csv(di[name])
    avg_ent = avg_uNLI(df_unli,'ENT')
    avg_neu = avg_uNLI(df_unli,'NEU')
    avg_con = avg_uNLI(df_unli,'CON')
    df_snli_u = df_snli
    df_snli_u['unli'] = 0
    values = [avg_neu,avg_ent,avg_con]
    conditions = [(df_snli_u['gold_label'] == 'neutral'),
                  (df_snli_u['gold_label'] == 'entailment'),
                  (df_snli_u['gold_label'] == 'contradiction')]
    df_snli_u['unli'] = np.select(conditions, values)
    return df_snli_u
    
        