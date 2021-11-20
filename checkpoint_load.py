import torch
from transformers import AutoConfig, AutoTokenizer
from model import BertRegresser
import torch.optim as optim
import os
def save_checkpt(LOSS,model,optimizer,EPOCH = 10,PATH = "model.pt"):
    torch.save({
            'epoch': EPOCH,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)


def load_checkpt(PATH,MODEL_NAME = 'bert-base-uncased'):
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = BertRegresser.from_pretrained(MODEL_NAME, config=config)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    model.train()
    return model,optimizer,epoch,loss



def find_files(filename, search_path = os.getcwd()):
   result = []
   for root, dir, files in os.walk(search_path):
      if filename in files:
         result.append(os.path.join(root, filename))
   return result

# if __name__ == "__main__":
#     checkpoint = 'model_' + '.pt'
#     print(find_files(checkpoint))
