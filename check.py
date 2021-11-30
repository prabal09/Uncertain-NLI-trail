import argparse
import os
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import numpy as np
import datetime
import pdb
from model import BERT_NLI,AutoTokenizer
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup,logging
from solver import train_epoch,eval_model
from dataloader import create_data_loader,get_dfs
import torch.nn as nn
from collections import defaultdict

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
def sentence_segmenter(text):
    current_position = 0
    cursor = 0
    sentences = []
    start = 0
    for c in range(len(text)):
        # print(text)
        if text[c] == "." or text[c] == "!":
            try:
                int(text[c-1])
            except ValueError:
                # print(text[start:start+10])
                # print(text[ind_c-5:ind_c+5])
                # int()
                sentences.append(text[current_position:cursor + 1])
                current_position = cursor + 2
        cursor += 1
    sentences = list(filter(('.').__ne__, sentences))
    sentences = list(filter(('').__ne__, sentences))
    return sentences

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_epochs', type=int, default=25)
    parser.add_argument('--pretrain_test_epoch', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--clf_test_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_step', type=int, default=50)

    parser.add_argument('--model_id', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--db_path', type=str, default=os.getcwd())

    parser.add_argument("--usnli_train", type=str, default="C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/train.csv",
                        help="Path to UNLI train (CSV format)")
    parser.add_argument("--usnli_dev", type=str, default="C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/dev.csv",
                        help="Path to UNLI dev (CSV format)")
    parser.add_argument("--usnli_test", type=str, default="C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/test.csv",
                        help="Path to UNLI test (CSV format)")
    parser.add_argument('--max_len', default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--tokenizer', default=AutoTokenizer.from_pretrained('bert-base-uncased'))
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    parser.add_argument('--device', default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    config = parser.parse_args()

    # warnings.filterwarnings("ignore")

    logging.set_verbosity_error()
    tokenizer = config.tokenizer
    model = BERT_NLI()
    device = config.device
    model = model.to(device)
    BATCH_SIZE = config.batch_size
    EPOCHS = config.epochs
    MAX_LEN = config.max_len
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    df_train, df_dev, df_test = get_dfs(config)
    # print(df_train.head(3))
    # print(len(df_train.pre[0]))
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    # print(len(train_data_loader))
    data = iter(train_data_loader)
    d = next(data)
    print(d.keys())
    pdb.set_trace()
    print(d['targets'][0])