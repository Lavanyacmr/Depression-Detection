import torch
import torch.nn as nn
from transformers import RobertaModel
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import Dataset

class RetrievalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, sent_model, max_len=128, top_k=5):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.top_k = top_k
        self.emb = sent_model.encode(texts, convert_to_tensor=True)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        sim = util.cos_sim(self.emb[idx].unsqueeze(0), self.emb).squeeze(0)
        sim[idx] = -1
        top = torch.topk(sim, min(self.top_k, len(sim) - 1)).indices
        retrieved = [self.texts[i] for i in top.cpu().numpy()]
        combined = self.texts[idx] + " [SEP] " + " [SEP] ".join(retrieved)
        enc = self.tokenizer(combined, max_length=self.max_len, padding='max_length',
                             truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label': self.labels[idx]
        }

class SpeechTRAG(nn.Module):
    def __init__(self, hidden_dim=512, attention_heads=8, ff_dim=512, num_layers=6, dropout=0.1):
        super().__init__()
        self.enc = RobertaModel.from_pretrained('roberta-base')
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=attention_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(768, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask, metadata=None):
        x = self.enc(input_ids, attention_mask).pooler_output
        x = self.proj(x).unsqueeze(1)
        x = self.transformer(x).squeeze(1)
        return torch.sigmoid(self.fc(x))