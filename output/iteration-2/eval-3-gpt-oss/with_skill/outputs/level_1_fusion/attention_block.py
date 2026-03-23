"""
Component: AttentionBlock
Source: data/gpt_oss.py
Abstraction Level: fusion
Parent: TransformerBlock

Operations: [RMSNorm, Linear (QKV), RotaryEmbedding, SDPA, Linear (output), residual add]

Input Shapes:
  - x: [32, 2880] dtype=bfloat16

Output Shapes:
  - output: [32, 2880] dtype=bfloat16

Weight Shapes:
  - sinks: [64]
  - norm.scale: [2880]
  - qkv.weight: [5120, 2880]
  - qkv.bias: [5120]
  - out.weight: [2880, 4096]
  - out.bias: [2880]
"""

import math

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, device=device, dtype=torch.float32))

    def forward(self, x):
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(x, cos, sin):
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base, dtype, initial_context_length=4096,
                 scaling_factor=1.0, ntk_alpha=1.0, ntk_beta=32.0, device=None):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self):
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device) / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0
            d_half = self.head_dim / 2
            low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
            high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
            assert 0 < low < high < d_half - 1
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq
            ramp = (torch.arange(d_half, dtype=torch.float32, device=freq.device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(self, query, key):
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window)
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class Model(nn.Module):
    """AttentionBlock with RMSNorm, QKV projection, RoPE, SDPA, output projection, and residual."""
    def __init__(self, layer_idx=0):
        super().__init__()
        hidden_size = 2880
        head_dim = 64
        num_attention_heads = 64
        num_key_value_heads = 8

        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.sliding_window = 128 if layer_idx % 2 == 0 else 0
        self.sinks = nn.Parameter(torch.empty(num_attention_heads, dtype=torch.bfloat16))
        self.norm = RMSNorm(hidden_size)
        qkv_dim = head_dim * (num_attention_heads + 2 * num_key_value_heads)
        self.qkv = nn.Linear(hidden_size, qkv_dim, dtype=torch.bfloat16)
        self.out = nn.Linear(head_dim * num_attention_heads, hidden_size, dtype=torch.bfloat16)
        self.sm_scale = 1 / math.sqrt(head_dim)
        self.rope = RotaryEmbedding(
            head_dim, 150000.0, torch.float32,
            initial_context_length=4096, scaling_factor=32.0,
            ntk_alpha=1.0, ntk_beta=32.0,
        )

    def forward(self, x):
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[:, self.num_attention_heads * self.head_dim : (self.num_attention_heads + self.num_key_value_heads) * self.head_dim].contiguous()
        v = qkv[:, (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim].contiguous()
        q = q.view(-1, self.num_key_value_heads, self.num_attention_heads // self.num_key_value_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        t = self.out(t)
        t = x + t
        return t


def get_inputs():
    return [torch.randn(32, 2880, dtype=torch.bfloat16)]


def get_init_inputs():
    return [0]  # layer_idx


def get_expected_output_shape():
    return [(32, 2880)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Shape mismatch: {actual} vs {expected}"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
