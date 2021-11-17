from transformers import AutoTokenizer, AutoModel
from torch import nn
import pdb

class BERT_NLI(nn.Module):

    def __init__(self, n_classes=1):
        super(BERT_NLI, self).__init__()
        # self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.bert = AutoModel.from_pretrained('bert-base-cased')
        # self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        # self.activation = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        pdb.set_trace()
        _,pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        print(pooled_output.shape)
        print(_.shape)
        # output = self.drop(pooled_output)
        output = self.out(pooled_output)
        # output = self.activation(output)
        return output