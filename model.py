import torch
import torch.nn as nn
from AttentionLayer import AttentionLayer
from encoder import Encoder
from decoder import Decoder
from position_encoding import PositionalEncoding

PAD_IDX = 0

class Transformer(nn.Module):
    def __init__(self, embed_size, vocab_size, device):
        super().__init__()

        self.device = device
        self.encoder = Encoder(embed_size, device, n_layers=6).to(device)
        self.decoder = Decoder(embed_size, vocab_size, device, n_layers=6).to(device)
        self.positional_encoding = PositionalEncoding(max_len=2000, embed_size=embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX).to(device)


    def make_input_mask(self, input_batch, pad_idx):
        return (input_batch == pad_idx).to(self.device)


    def initialize_embedding(self, embedding_weights):
        embedding_weights = embedding_weights.to(self.device)
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=False)


    def forward(self, input_batch, target_batch):

        # input_batch - [batch_size, seq_len]

        input_embedded = self.positional_encoding(self.embedding(input_batch).to(self.device))
        target_embedded = self.positional_encoding(self.embedding(target_batch).to(self.device))

        input_padding_mask = self.make_input_mask(input_batch, PAD_IDX)

        encoder_outs = self.encoder(input_embedded, input_padding_mask)
        target_padding_mask = self.make_input_mask(target_batch, PAD_IDX)
        decoder_outs = self.decoder(target_embedded, encoder_outs, input_padding_mask, target_padding_mask)

        return decoder_outs
