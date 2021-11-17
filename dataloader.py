import pdb

import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
MAX_LEN = 160

class NLIDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]
    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = NLIDataset(
    reviews=(['[CLS]'] + df.pre + ['[SEP]'] + df.hyp+ ['[SEP]']).to_numpy(),
    targets=df.unli.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  pdb.set_trace()
  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

def get_dfs(config):
    df_train = pd.read_csv(config.usnli_train)
    df_dev = pd.read_csv(config.usnli_dev)
    df_test = pd.read_csv(config.usnli_test)
    return df_train,df_dev,df_test