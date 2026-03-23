"""
Component: RotaryEmbedding (RoPE with YaRN)
Source: data/gpt_oss.py
Abstraction Level: kernel
Parent: AttentionBlock

Operations: [RotaryEmbedding - frequency computation, cos/sin, apply rotary]

Input Shapes:
  - query: [32, 8, 8, 64] dtype=bfloat16
  - key: [32, 8, 64] dtype=bfloat16

Output Shapes:
  - query: [32, 8, 8, 64] dtype=bfloat16
  - key: [32, 8, 64] dtype=bfloat16

Weight Shapes: (no learnable parameters)
"""

import math

import torch
import torch.nn as nn


def _apply_rotary_emb(x, cos, sin):
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class Model(nn.Module):
    def __init__(self, head_dim=64, base=150000.0, initial_context_length=4096,
                 scaling_factor=32.0, ntk_alpha=1.0, ntk_beta=32.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta

    def _compute_concentration_and_inv_freq(self):
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim
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
        t = torch.arange(num_tokens, dtype=torch.float32)
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


def get_inputs():
    return [
        torch.randn(32, 8, 8, 64, dtype=torch.bfloat16),  # query
        torch.randn(32, 8, 64, dtype=torch.bfloat16),       # key
    ]


def get_init_inputs():
    return [64, 150000.0, 4096, 32.0, 1.0, 32.0]


def get_expected_output_shape():
    return [(32, 8, 8, 64), (32, 8, 64)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert isinstance(output, tuple) and len(output) == 2
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output[0].shape, output[1].shape]
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
