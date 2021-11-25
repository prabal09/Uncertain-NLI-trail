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

MAX_LEN_TRAIN = 50
MAX_LEN_VALID = 50
MAX_LEN_TEST = 50
BATCH_SIZE = 16
LR = 1e-5
NUM_EPOCHS = 3
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
    # criterion = nn.MSELoss().to(device)
    criterion = nn.BCELoss().to(device)
    ## Optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=LR)
    mode = 'SNLI+uNli'  #'SNLI-only'
    if mode == 'uNli':
        print("RUNNING : ",mode)
        di = dict()
        di["usnli_train"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/train.csv"
        di["usnli_dev"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/dev.csv"
        di["usnli_test"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/test.csv"
        df_train, df_dev, df_test = get_dfs(di)
        hyp_only = True #False
    if mode == 'SNLI+uNli':
        print("RUNNING : ",mode)
        di = dict()
        di["usnli_train"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/train.csv"
        di["usnli_dev"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/dev.csv"
        di["usnli_test"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/test.csv"
        df_train_unli, df_dev_unli, df_test_unli = get_dfs(di)

        di = dict()
        di["usnli_train"] = "C:/Users/praba/PycharmProjects/UncertainNLI/Uncertain-NLI-trail/snli_u_train.csv"
        di["usnli_dev"] = "C:/Users/praba/PycharmProjects/UncertainNLI/Uncertain-NLI-trail/snli_unli_dev.csv"
        di["usnli_test"] = "C:/Users/praba/PycharmProjects/UncertainNLI/Uncertain-NLI-trail/snli_unli_test.csv"

        df_train_snli, df_dev_snli, df_test_snli = get_dfs(di)
        # sample_submission = pd.read_csv('../input/commonlitreadabilityprize/sample_submission.csv')
        # df_train,df_dev,df_test = get_dfs(di)
        # df_train_snli = df_train_snli.drop(df_train_snli.index[91400:91500])
        # df_train_snli = df_train_snli.drop(df_train_snli.index[311000:312000])
        df_train = df_train_snli.append(df_train_unli).reset_index(drop=True)
        df_dev = df_dev_snli.append(df_dev_unli).reset_index(drop=True)
        df_test = df_test_snli.append(df_test_unli).reset_index(drop=True)
        # df_train, df_dev, df_test = pd.concat([df_train_snli,]),pd.concat([df_dev_snli,df_dev_unli]),pd.concat([df_test_snli,df_test_unli])
    elif mode == 'SNLI-only':
        print("RUNNING : ", mode)
        di = dict()
        di["usnli_train"] = "C:/Users/praba/PycharmProjects/UncertainNLI/Uncertain-NLI-trail/snli_u_train.csv"
        di["usnli_dev"] = "C:/Users/praba/PycharmProjects/UncertainNLI/Uncertain-NLI-trail/snli_unli_dev.csv"
        di["usnli_test"] = "C:/Users/praba/PycharmProjects/UncertainNLI/Uncertain-NLI-trail/snli_unli_test.csv"

        df_train, df_dev, df_test = get_dfs(di)
        # df_train = df_train.drop(df_train.index[91400:91500])
        # df_train = df_train.drop(df_train.index[311000:312000])
        df_train = df_train.reset_index(drop=True)
    ## Training Dataset
    train_set = Excerpt_Dataset(data=df_train, maxlen=MAX_LEN_TRAIN, tokenizer=tokenizer,hyp_only=hyp_only)
    valid_set = Excerpt_Dataset(data=df_dev, maxlen=MAX_LEN_VALID, tokenizer=tokenizer)
    test_set = Excerpt_Dataset(data=df_test, maxlen=MAX_LEN_TEST, tokenizer=tokenizer)

    ## Data Loaders
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    valid_loader = DataLoader(dataset=valid_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, num_workers=NUM_THREADS)
    checkpoint = 'model_' + str(NUM_EPOCHS) + '.pt'
    # print(len(train_loader))
    # if len(find_files(checkpoint)) !=0:
    #     model,optimizer,epoch,loss = load_checkpt(PATH = checkpoint,MODEL_NAME = 'bert-base-uncased')

    train(model=model,
      criterion=criterion,
      optimizer=optimizer,
      train_loader=train_loader,
      val_loader=valid_loader,
      epochs = NUM_EPOCHS,
     device = device)
    # loss = evaluate(model, criterion, train_loader, device)
    criterion_mse = nn.MSELoss().to(device)
    mse_loss_dev = evaluate(model, criterion = criterion_mse, dataloader= valid_loader, device=device)
    mse_loss_test = evaluate(model, criterion = criterion_mse, dataloader= test_loader, device=device)
    print('Final MSE Loss on dev data',mse_loss_dev)
    print('Final MSE Loss on test data',mse_loss_test)
    pearson_loss_dev = evaluate(model, criterion= 'pearson', dataloader=valid_loader, device=device)
    pearson_loss_test = evaluate(model, criterion= 'pearson', dataloader=test_loader, device=device)
    print('Final Pearson Loss on dev data', pearson_loss_dev)
    print('Final Pearson Loss on test data', pearson_loss_test)
    spearman_loss_dev = evaluate(model, criterion= 'spearman', dataloader=valid_loader, device=device)
    spearman_loss_test = evaluate(model, criterion= 'spearman', dataloader=test_loader, device=device)
    print('Final Spearman Loss on dev data',spearman_loss_dev)
    print('Final Spearman Loss on test data', spearman_loss_test)
    # output = predict(model, test_loader, device)
    # out2 = []
    # for out in output:
    #     out2.append(out.cpu().detach().numpy())
    # out = np.array(out2).reshape(len(out2))
    # submission = pd.DataFrame({'id': df_test['id'], 'pre': df_test['pre'],'hyp':df_test['hyp'],'target':out})
    # submission.to_csv('submission.csv', index=False)
    # save_checkpt(LOSS = loss,model = model,optimizer = optimizer,EPOCH = NUM_EPOCHS,PATH = checkpoint)
