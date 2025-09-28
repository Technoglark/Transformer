import torch
import torch.nn as nn
from AttentionLayer import MultiHeadAttentionLayer


class DecoderLayer(nn.Module):
    def __init__(self, embed_size, device):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttentionLayer(embed_size, device, n_heads=6).to(device)
        self.attention = MultiHeadAttentionLayer(embed_size, device, n_heads=6).to(device)

        self.ff_size = embed_size * 2

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, self.ff_size),
            nn.ReLU(),
            nn.Linear(self.ff_size, embed_size)
        ).to(device)

        self.norm1 = nn.LayerNorm(embed_size).to(device)
        self.norm2 = nn.LayerNorm(embed_size).to(device)
        self.norm3 = nn.LayerNorm(embed_size).to(device)

        self.device = device


    def forward(self, x, encoder_outs, input_mask, combined_mask):
        
        input_mask = input_mask.unsqueeze(1).unsqueeze(2)

        if combined_mask.dim() == 2:
            combined_mask = combined_mask.unsqueeze(0).unsqueeze(1)
        elif combined_mask.dim() == 3:
            combined_mask = combined_mask.unsqueeze(1)
        else:
            combined_mask = combined_mask

        self_attention_outs = self.self_attention(x, x, x, mask=combined_mask)
        x = self.norm1(x + self_attention_outs)
       

        attention_outs = self.attention(x, encoder_outs, encoder_outs, input_mask)
        x = self.norm2(x + attention_outs)

        ff_outs = self.feed_forward(x)
        x = self.norm3(x + ff_outs)

        return x

class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size, device, n_layers=6):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList(DecoderLayer(embed_size, device) for _ in range(n_layers))

        self.linear = nn.Linear(embed_size, vocab_size).to(device)
        self.device = device
    
    def forward(self, x, encoder_outs, input_mask, decoder_mask):

        mask = torch.triu(torch.ones(x.shape[1], x.shape[1]), diagonal=1).bool().to(self.device)
        decoder_mask = decoder_mask.unsqueeze(1)
        combined_mask = mask | decoder_mask

        for layer in self.layers:
            x = layer(x, encoder_outs, input_mask, combined_mask)

        return self.linear(x)

