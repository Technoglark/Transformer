import torch
import torch.nn as nn
import math

class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        

    def forward(self, Q, K, V, mask=None):
        # x - [batch_size, seq_len, hidden_size]

        head_dim = Q.size(3)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attention = torch.softmax(scores, dim=-1)
        weighted = torch.matmul(attention, V)

        return weighted
    


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, device, n_heads=6):
        super(MultiHeadAttentionLayer, self).__init__()

        assert hidden_size % n_heads == 0

        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads
        self.Q = nn.Linear(hidden_size, hidden_size).to(device)
        self.K = nn.Linear(hidden_size, hidden_size).to(device)
        self.V = nn.Linear(hidden_size, hidden_size).to(device)
        self.attention = AttentionLayer()

        self.W = nn.Linear(hidden_size, hidden_size).to(device)
        

    def forward(self, q, k, v, mask=None):
        query = self.Q(q)
        key = self.K(k)
        value = self.V(v)

        batch_size = q.size(0)

        # [batch_size, seq_len, num_heads, head_dim]
        query = query.view(batch_size, -1, self.n_heads, self.head_dim)
        key = key.view(batch_size, -1, self.n_heads, self.head_dim)
        value = value.view(batch_size, -1, self.n_heads, self.head_dim)

        # [batch_size, num_heads, seq_len, head_dim]
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_outs = self.attention(query, key, value, mask)
        # [batch_size, hum_heads, seq_len, head_dim] -> [batch_size, seq_len, hum_heads, head_dim]
        attention_outs = attention_outs.permute(0, 2, 1, 3).contiguous()

        
        # [batch_size, seq_len, hum_heads, head_dim] -> [batch_size, seq_len, hidden_size]
        concatenated_outs = attention_outs.view(batch_size, -1, self.hidden_size)

        return self.W(concatenated_outs)




        




    
