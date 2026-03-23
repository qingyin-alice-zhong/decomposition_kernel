"""
Component: Scaled Dot-Product Attention with GQA and Sink Tokens
Source: data/gpt_oss.py
Abstraction Level: kernel
Parent: AttentionBlock

Operations: [SDPA with GQA, causal mask, sliding window, sink tokens]

Input Shapes:
  - Q: [32, 8, 8, 64] dtype=bfloat16
  - K: [32, 8, 64] dtype=bfloat16
  - V: [32, 8, 64] dtype=bfloat16
  - S: [64] dtype=bfloat16 (sink parameters)

Output Shapes:
  - output: [32, 4096] dtype=bfloat16

Weight Shapes:
  - sinks: [64] (nn.Parameter)
"""

import math

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_attention_heads=64, num_key_value_heads=8,
                 head_dim=64, sliding_window=128):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.sm_scale = 1 / math.sqrt(head_dim)
        self.sinks = nn.Parameter(torch.empty(num_attention_heads, dtype=torch.bfloat16))

    def forward(self, Q, K, V):
        n_tokens, n_heads, q_mult, d_head = Q.shape
        assert K.shape == (n_tokens, n_heads, d_head)
        assert V.shape == (n_tokens, n_heads, d_head)
        K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
        V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
        S = self.sinks.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
        mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
        if self.sliding_window > 0:
            mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-self.sliding_window)
        QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
        QK *= self.sm_scale
        QK += mask[None, None, :, :]
        QK = torch.cat([QK, S], dim=-1)
        W = torch.softmax(QK, dim=-1)
        W = W[..., :-1]
        attn = torch.einsum("hmqk,khmd->qhmd", W, V)
        return attn.reshape(n_tokens, -1)


def get_inputs():
    return [
        torch.randn(32, 8, 8, 64, dtype=torch.bfloat16),  # Q
        torch.randn(32, 8, 64, dtype=torch.bfloat16),       # K
        torch.randn(32, 8, 64, dtype=torch.bfloat16),       # V
    ]


def get_init_inputs():
    return [64, 8, 64, 128]


def get_expected_output_shape():
    return [(32, 4096)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
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
