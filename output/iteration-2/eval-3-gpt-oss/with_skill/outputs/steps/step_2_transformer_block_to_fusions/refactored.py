"""
Step 2 Refactored: TransformerBlock -> Fusions
Decomposes TransformerBlock into:
  - AttentionBlock (L1 fusion) - includes norm, QKV, RoPE, SDPA, output proj, residual
  - MLPBlock (L1 fusion) - includes norm, gate, MoE routing, SwiGLU MLP, residual
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "children"))

from attention_block import Model as AttentionBlock
from mlp_block import Model as MLPBlock


class RefactoredModel(nn.Module):
    def __init__(self, layer_idx=0):
        super().__init__()
        self.attn = AttentionBlock(layer_idx)
        self.mlp = MLPBlock()

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        return x


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
