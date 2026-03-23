"""
Step 3a Refactored: AttentionBlock -> Kernels
Decomposes AttentionBlock into:
  - RMSNorm (L0)
  - QKV Linear (L0)
  - RotaryEmbedding (L0)
  - SDPA with GQA (L0)
  - Output Linear (L0)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "children"))

from rms_norm_2880_fp32 import Model as RMSNorm
from linear_2880x5120_bf16 import Model as QKVLinear
from rotary_embedding_64_fp32 import Model as RoPE
from sdpa_gqa_64_bf16 import Model as SDPA
from linear_4096x2880_bf16 import Model as OutLinear


class RefactoredModel(nn.Module):
    def __init__(self, layer_idx=0):
        super().__init__()
        self.num_attention_heads = 64
        self.num_key_value_heads = 8
        self.head_dim = 64

        self.norm = RMSNorm(2880)
        self.qkv = QKVLinear(2880, 5120)
        self.rope = RoPE(64, 150000.0, 4096, 32.0, 1.0, 32.0)
        sliding_window = 128 if layer_idx % 2 == 0 else 0
        self.sdpa = SDPA(64, 8, 64, sliding_window)
        self.out = OutLinear(4096, 2880)

    def forward(self, x):
        t = self.norm(x)
        qkv = self.qkv(t)

        # Split QKV - data plumbing (slicing/reshaping)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[:, self.num_attention_heads * self.head_dim : (self.num_attention_heads + self.num_key_value_heads) * self.head_dim].contiguous()
        v = qkv[:, (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim].contiguous()

        # Reshape for GQA - data plumbing
        q = q.view(-1, self.num_key_value_heads, self.num_attention_heads // self.num_key_value_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)

        # Apply RoPE
        q, k = self.rope(q, k)

        # Attention
        t = self.sdpa(q, k, v)

        # Output projection
        t = self.out(t)

        # Residual add - data plumbing
        t = x + t
        return t


def get_inputs():
    return [torch.randn(32, 2880, dtype=torch.bfloat16)]


def get_init_inputs():
    return [0]


if __name__ == "__main__":
    model = RefactoredModel(*get_init_inputs())
    model.eval()
    with torch.no_grad():
        inputs = get_inputs()
        output = model(*inputs)
        print(f"Output shape: {output.shape}")
        print("PASS" if output.shape == (32, 2880) else "FAIL")
