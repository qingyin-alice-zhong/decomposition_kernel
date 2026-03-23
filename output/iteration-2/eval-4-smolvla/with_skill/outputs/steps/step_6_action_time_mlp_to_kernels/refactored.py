"""
Refactored: Action-Time MLP -> [linear_1440x720, silu, linear_720x720]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "children"))

import torch
import torch.nn as nn

from linear_1440x720_fp32 import Model as Linear1440x720
from linear_720x720_fp32 import Model as Linear720x720
from silu_fp32 import Model as SiLU

EXPERT_HIDDEN_SIZE = 720
CHUNK_SIZE = 50


class RefactoredModel(nn.Module):
    """Action-Time MLP refactored to use child kernel modules."""
    def __init__(self):
        super().__init__()
        _mlp_in = Linear1440x720(EXPERT_HIDDEN_SIZE * 2, EXPERT_HIDDEN_SIZE)
        self.mlp_in = _mlp_in.linear  # expose inner linear
        _mlp_out = Linear720x720(EXPERT_HIDDEN_SIZE, EXPERT_HIDDEN_SIZE)
        self.mlp_out = _mlp_out.linear
        self.silu = SiLU()

    def forward(self, action_time_emb):
        x = self.mlp_in(action_time_emb)
        x = self.silu(x)
        x = self.mlp_out(x)
        return x


def get_inputs():
    return [torch.randn(1, CHUNK_SIZE, EXPERT_HIDDEN_SIZE * 2)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, CHUNK_SIZE, EXPERT_HIDDEN_SIZE)]
