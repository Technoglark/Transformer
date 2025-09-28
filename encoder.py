import torch
import torch.nn as nn
from AttentionLayer import MultiHeadAttentionLayer


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, device):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttentionLayer(embed_size, device, n_heads=6).to(device)

        self.norm1 = nn.LayerNorm(embed_size).to(device)
        self.norm2 = nn.LayerNorm(embed_size).to(device)
        self.ff_size = embed_size * 2

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, self.ff_size),
            nn.ReLU(),
            nn.Linear(self.ff_size, embed_size)
        ).to(device)

        self.device = device

    
    def forward(self, x, padding_mask):
        # x - [batch_size, seq_len, embed_size]
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        attention_out = self.attention(x, x, x, padding_mask)
        x = self.norm1(x + attention_out)

        ff_out = self.feed_forward(x)

        x = self.norm2(x + ff_out)

        return x
    
class Encoder(nn.Module):
    def __init__(self, embed_size, device, n_layers=6):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(EncoderLayer(embed_size, device) for _ in range(n_layers))
    
    def forward(self, x, padding_mask):
        for layer in self.layers:
            x = layer(x, padding_mask)

        return x

        