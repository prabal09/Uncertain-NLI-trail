import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
class Excerpt_Dataset(Dataset):

    def __init__(self, data, maxlen, tokenizer):
        #Store the contents of the file in a pandas dataframe
        self.df = data.reset_index()
        #Initialize the tokenizer for the desired transformer model
        self.tokenizer = tokenizer
        #Maximum length of the tokens list to keep all the sequences of fixed size
        self.maxlen = maxlen

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        #Select the sentence and label at the specified index in the data frame
        excerpt1 = self.df.loc[index, 'pre']
        excerpt2 = self.df.loc[index, 'hyp']
        # print('premise',excerpt1)
        # print('hypothesis',excerpt2)
        try:
            target = self.df.loc[index, 'unli']
        except:
            target = 0.0
        identifier = self.df.loc[index, 'id']
        #Preprocess the text to be suitable for the transformer
        tokens1 = self.tokenizer.tokenize(excerpt1)
        tokens2 = self.tokenizer.tokenize(excerpt2)
        tokens = ['[CLS]'] + tokens1 + ['[SEP]'] + tokens2
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]']
        #Obtain the indices of the tokens in the BERT Vocabulary
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(input_ids)
        #Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attention_mask = (input_ids != 0).long()
        # print('target',target)
        target = torch.tensor([target], dtype=torch.float32)

        # print(target.size())
        # target = target.reshape((-1, 1))
        return input_ids, attention_mask, target

def get_dfs(di):
    df_train = pd.read_csv(di['usnli_train'])
    df_dev = pd.read_csv(di['usnli_dev'])
    df_test = pd.read_csv(di['usnli_test'])
    return df_train,df_dev,df_test
