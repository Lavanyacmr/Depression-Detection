import torch.nn as nn
from transformers import AutoModel

class SqueezeBERTTextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained('squeezebert/squeezebert-uncased')
        self.fc = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, metadata=None):
        x = self.enc(input_ids, attention_mask).pooler_output
        return torch.sigmoid(self.fc(x))