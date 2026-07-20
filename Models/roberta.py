import torch.nn as nn
from .tdd_net import TextEncoder

class RoBERTaTextOnly(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.enc = TextEncoder(dropout=dropout)
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, metadata=None):
        return torch.sigmoid(self.fc(self.enc(input_ids, attention_mask)))