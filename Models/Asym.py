import torch.nn as nn
from .tdd_net import TextEncoder

class ASYM(nn.Module):
    def __init__(self, hidden_dim=256, attention_heads=8, ff_dim=512, dropout=0.3):
        super().__init__()
        self.text_enc = TextEncoder(dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=attention_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj = nn.Linear(768, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, metadata=None):
        x = self.text_enc(input_ids, attention_mask)
        x = self.proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return torch.sigmoid(self.fc(x))