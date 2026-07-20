import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocessing.metadata_extraction import METADATA_COLS

TEXT_COLS = ["Posts", "Captions", "Comments"]
TARGET = "Risk_Label"

def concat_text(row):
    parts = [str(row[c]) for c in TEXT_COLS if isinstance(row[c], str) and row[c].strip()]
    return " ".join(parts) if parts else ""

class DepressionDataset(Dataset):
    def __init__(self, texts, metadata, labels, tokenizer, max_len=128):
        self.texts = texts
        self.metadata = torch.tensor(metadata, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'metadata': self.metadata[idx],
            'label': self.labels[idx]
        }

def load_and_preprocess(filepath, tokenizer, max_len=128):
    df = pd.read_excel(filepath, sheet_name="Participants")
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].apply(lambda x: 1 if str(x).strip().lower() in ["high risk", "1"] else 0)
    from preprocessing.metadata_extraction import preprocess_metadata
    df, scaler = preprocess_metadata(df, fit=True)
    texts = df.apply(concat_text, axis=1).tolist()
    meta = df[METADATA_COLS].values.astype(np.float32)
    labels = df[TARGET].values.astype(np.float32)
    dataset = DepressionDataset(texts, meta, labels, tokenizer, max_len)
    return dataset, df, scaler

def get_texts_and_metadata(dataset, indices):
    texts = [dataset.texts[i] for i in indices]
    meta = np.array([dataset.metadata[i].numpy() for i in indices])
    labels = dataset.labels[indices].flatten().cpu().numpy()
    return texts, meta, labels