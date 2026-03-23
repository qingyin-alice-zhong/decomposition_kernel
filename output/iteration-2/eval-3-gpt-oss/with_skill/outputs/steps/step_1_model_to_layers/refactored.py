"""
Step 1 Refactored: GPT-OSS Model -> Layers
Decomposes the Transformer model into:
  - Embedding (L0 kernel)
  - TransformerBlock x N (L2 layer)
  - RMSNorm (L0 kernel)
  - Unembedding Linear (L0 kernel)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add children directory to path
sys.path.insert(0, str(Path(__file__).parent / "children"))

from transformer_block import Model as TransformerBlock
from embedding_201088x2880_bf16 import Model as Embedding
from rms_norm_2880_fp32 import Model as FinalNorm
from linear_2880x201088_bf16 import Model as Unembedding


class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding(201088, 2880)
        self.block_0 = TransformerBlock(0)  # layer_idx=0 (sliding window)
        self.block_1 = TransformerBlock(1)  # layer_idx=1 (no sliding window)
        self.norm = FinalNorm(2880)
        self.unembedding = Unembedding(2880, 201088)

    def forward(self, x):
        x = self.embedding(x)
        x = self.block_0(x)
        x = self.block_1(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x


def get_inputs():
    return [torch.randint(0, 201088, (32,), dtype=torch.int32)]


def get_init_inputs():
    return []


if __name__ == "__main__":
    model = RefactoredModel()
    model.eval()
    with torch.no_grad():
        inputs = get_inputs()
        output = model(*inputs)
        print(f"Output shape: {output.shape}")
        print("PASS" if output.shape == (32, 201088) else "FAIL")
