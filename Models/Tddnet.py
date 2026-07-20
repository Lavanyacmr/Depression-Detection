import torch
import torch.nn as nn
from transformers import RobertaModel

class TextEncoder(nn.Module):
    def __init__(self, model_name='roberta-base', dropout=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids, attention_mask).pooler_output
        return self.dropout(out)

class MetadataEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

class AlignmentProjection(nn.Module):
    def __init__(self, text_dim=768, meta_dim=128, common_dim=448):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, common_dim)
        self.meta_proj = nn.Linear(meta_dim, common_dim)

    def forward(self, te, me):
        return self.text_proj(te), self.meta_proj(me)

class AttentionClassificationHead(nn.Module):
    def __init__(self, fused_dim=896, num_heads=8, dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(fused_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fused_dim, 1)

    def forward(self, z):
        z = z.unsqueeze(1)
        attn_out, _ = self.attn(z, z, z)
        z = z + attn_out
        z = self.norm(z).squeeze(1)
        z = self.dropout(z)
        return torch.sigmoid(self.fc(z))

class TDDNet(nn.Module):
    def __init__(self, metadata_dim, text_model='roberta-base', meta_embed_dim=128,
                 common_dim=448, num_heads=8, dropout=0.3):
        super().__init__()
        self.text_enc = TextEncoder(text_model, dropout)
        self.meta_enc = MetadataEncoder(metadata_dim, meta_embed_dim, dropout)
        self.align = AlignmentProjection(768, meta_embed_dim, common_dim)
        self.cls = AttentionClassificationHead(common_dim * 2, num_heads, dropout)

    def forward(self, input_ids, attention_mask, metadata):
        te = self.text_enc(input_ids, attention_mask)
        me = self.meta_enc(metadata)
        tp, mp = self.align(te, me)
        fused = torch.cat([tp, mp], dim=1)
        return self.cls(fused)