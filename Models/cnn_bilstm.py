import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, cnn_filters=128, kernel_size=5,
                 hidden_units=128, dropout=0.5, padding_idx=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.conv = nn.Conv1d(embedding_dim, cnn_filters, kernel_size, padding=kernel_size // 2)
        self.lstm = nn.LSTM(cnn_filters, hidden_units, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout if 2 > 1 else 0)
        self.fc = nn.Linear(hidden_units * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, metadata=None):
        x = self.embedding(input_ids)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return torch.sigmoid(self.fc(out))