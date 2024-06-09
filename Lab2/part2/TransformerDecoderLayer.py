import torch
import torch.nn as nn
import torch.nn.functional as F
from datautils import generate_tgt_mask


class Attention(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, hidden_dim: int):
        super(Attention, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.Q = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.K = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.V = nn.Linear(embed_dim, hidden_dim, bias=False)

    def forward(self, values, keys, queries, mask=None):
        values = self.V(values)
        keys = self.K(keys)
        queries = self.Q(queries)

        scaled_dot_product = torch.bmm(queries, keys.transpose(1, 2)) / (self.hidden_dim ** 0.5)

        if mask is not None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(scaled_dot_product, dim=-1)

        output = torch.bmm(attention, values)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, heads: int, dropout: float):
        super(MultiHeadAttention, self).__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.heads = heads
        self.hidden_dim = embed_dim // heads

        assert self.hidden_dim * heads == embed_dim, "embed_dim 必须被注意力头整除"

        self.multi_head_attention_layers = nn.ModuleList([
            Attention(self.seq_len, self.embed_dim, self.hidden_dim)
            for _ in range(self.heads)
        ])

        self.out_linear = nn.Linear(self.hidden_dim * self.heads, self.embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, queries, mask=None):
        attention_outputs = torch.cat([
            attention_layer(values, keys, queries, mask)
            for attention_layer in self.multi_head_attention_layers
        ], dim=-1)
        output = self.out_linear(attention_outputs)
        output = self.dropout(output)
        return output


class Expert(nn.Module):
    def __init__(self, embed_dim: int):
        super(Expert, self).__init__()
        self.embed_dim = embed_dim
        self.expert_layer = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.ReLU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim)
        )

    def forward(self, x):
        return self.expert_layer(x)


class TopKRouter(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int, active_experts: int):
        super(TopKRouter, self).__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.active_experts = active_experts
        # 使用简单的 MLP 作为 Router
        self.top_k_router = nn.Sequential(
            nn.Linear(self.embed_dim, self.num_experts),
            nn.ReLU()
        )

    def forward(self, x):
        scores = self.top_k_router(x)  # (batch_size, seq_len, num_experts)

        top_k_values, top_k_indices = torch.topk(scores, self.active_experts,
                                                 dim=-1)  # (batch_size, seq_len, active_experts)

        mask = torch.zeros_like(scores).scatter(-1, top_k_indices, 1)  # (batch_size, seq_len, num_experts)

        # mask 中被选中的位置为 1，未被选中的位置为 0

        masked_scores = scores.masked_fill(mask == 0, float('-inf'))
        router_weight = torch.softmax(masked_scores, dim=-1)

        return router_weight, mask


class SparseMoE(nn.Module):
    def __init__(self, embed_dim: int, num_experts: int, active_experts: int):
        super(SparseMoE, self).__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.active_experts = active_experts
        self.experts = nn.ModuleList([Expert(self.embed_dim) for _ in range(self.num_experts)])
        self.router = TopKRouter(self.embed_dim, self.num_experts, self.active_experts)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        router_output, mask = self.router(x)

        # 初始化一个全0的输出张量
        outputs = torch.zeros_like(x)

        # 遍历所有专家并将输出加权累加
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)  # 获取当前专家的输出
            # 使用mask和router_output来加权输出
            weight = router_output[:, :, i:i + 1] * mask[:, :, i:i + 1]
            outputs += weight * expert_output

        return outputs


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            n_heads: int,
            seq_len: int,
            num_experts: int,
            active_experts: int,
            dropout: float
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(seq_len, embed_dim, n_heads, dropout)
        self.encoder_decoder_attention = MultiHeadAttention(seq_len, embed_dim, n_heads, dropout)
        self.moe_ffn = SparseMoE(embed_dim, num_experts, active_experts)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # self-attention
        x = self.norm1(x)
        x = x + self.dropout1(self.self_attention(x, x, x, tgt_mask))

        # encoder-decoder attention
        x = self.norm2(x)
        x = x + self.dropout2(self.encoder_decoder_attention(memory, memory, x, src_mask))

        # moe ffn
        x = self.norm3(x)
        x = x + self.dropout3(self.moe_ffn(x))

        return x


class SparseMoETransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            embed_dim: int,
            n_layers: int,
            n_heads: int,
            num_experts: int,
            active_experts: int,
            dropout: float
    ):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        super(SparseMoETransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(seq_len, embed_dim, dropout)
        self.transformer_layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, n_heads, seq_len, num_experts, active_experts, dropout)
            for _ in range(n_layers)
        ])
        self.out_linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, src_mask=None, tgt_mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)

        for layer in self.transformer_layers:
            x = layer(x, x, src_mask, tgt_mask)

        output = self.out_linear(x)
        return output

    def generate(self, input_tokens, max_new_tokens):
        device = next(self.parameters()).device
        input_tokens = input_tokens.to(device)

        if input_tokens.size(1) >= self.seq_len:
            input_tokens = input_tokens[:, :self.seq_len]
        else:
            input_tokens = F.pad(input_tokens, (0, self.seq_len - input_tokens.size(1)))

        for _ in range(max_new_tokens):
            if input_tokens.size(1) >= self.seq_len:
                input_tokens = input_tokens[:, -self.seq_len:]

            tgt_mask = generate_tgt_mask(input_tokens.size(1)).to(device)
            output = self(input_tokens, tgt_mask=tgt_mask)
            last_token_logits = output[:, -1, :]  # 取最后一个 token 的 logits
            probs = F.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat([input_tokens, next_token], dim=-1)

            if next_token.item() == self.vocab_size - 1:  # Assuming the last vocab index is an EOS token
                break

        return input_tokens
