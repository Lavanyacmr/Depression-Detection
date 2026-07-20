import torch.nn as nn
from transformers import DistilBertModel

class DistilBERTTextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, metadata=None):
        cls = self.enc(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return torch.sigmoid(self.fc(cls))