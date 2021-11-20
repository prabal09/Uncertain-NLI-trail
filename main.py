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

from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig, AutoTokenizer

from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import seaborn as sns

from dataloader import Excerpt_Dataset, get_dfs
from model import *
from solver import *
from checkpoint_load import *

MAX_LEN_TRAIN = 120
MAX_LEN_VALID = 120
MAX_LEN_TEST = 120
BATCH_SIZE = 4
LR = 1e-3
NUM_EPOCHS = 10
NUM_THREADS = 1  ## Number of threads for collecting dataset
MODEL_NAME = 'bert-base-uncased'

if __name__ == "__main__":
    ## Configuration loaded from AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME)
    ## Tokenizer loaded from AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ## Creating the model from the desired transformer model
    model = BertRegresser.from_pretrained(MODEL_NAME, config=config)
    ## GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## Putting model to device
    model = model.to(device)
    ## Takes as the input the logits of the positive class and computes the binary cross-entropy
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    ## Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=LR)

    di = dict()
    di["usnli_train"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/train.csv"
    di["usnli_dev"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/train.csv"
    di["usnli_test"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/train.csv"

    # dftrain = pd.read_csv('../input/commonlitreadabilityprize/train.csv')
    # dftest = pd.read_csv('../input/commonlitreadabilityprize/test.csv')
    # sample_submission = pd.read_csv('../input/commonlitreadabilityprize/sample_submission.csv')
    df_train,df_dev,df_test = get_dfs(di)
    ## Training Dataset
    train_set = Excerpt_Dataset(data=df_train, maxlen=MAX_LEN_TRAIN, tokenizer=tokenizer)
    valid_set = Excerpt_Dataset(data=df_dev, maxlen=MAX_LEN_VALID, tokenizer=tokenizer)
    test_set = Excerpt_Dataset(data=df_test, maxlen=MAX_LEN_TEST, tokenizer=tokenizer)

    ## Data Loaders
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    checkpoint = 'model_' + str(NUM_EPOCHS) + '.pt'
    # print(len(train_loader))
    if len(find_files(checkpoint)) !=0:
        model,optimizer,epoch,loss = load_checkpt(PATH = checkpoint,MODEL_NAME = 'bert-base-uncased')

    train(model=model,
      criterion=criterion,
      optimizer=optimizer,
      train_loader=train_loader,
      val_loader=valid_loader,
      epochs = NUM_EPOCHS,
     device = device)

    output = predict(model, test_loader, device)
    out2 = []
    for out in output:
        out2.append(out.cpu().detach().numpy())
    out = np.array(out2).reshape(len(out2))
    submission = pd.DataFrame({'id': df_test['id'], 'pre': df_test['pre'],'hyp':df_test['hyp'],'target':out})

    save_checkpt(EPOCH = NUM_EPOCHS,PATH = checkpoint)
