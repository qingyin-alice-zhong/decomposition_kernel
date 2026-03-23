"""
Refactored: Expert MLP -> [linear_720x2048 x2, silu, linear_2048x720]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "children"))

import torch
import torch.nn as nn

from linear_720x2048_fp32 import Model as Linear720x2048
from linear_2048x720_fp32 import Model as Linear2048x720
from silu_fp32 import Model as SiLU

EXPERT_HIDDEN_SIZE = 720
EXPERT_INTERMEDIATE_SIZE = 2048


class RefactoredModel(nn.Module):
    """Expert SwiGLU MLP refactored to use child kernel modules."""
    def __init__(self):
        super().__init__()
        _gate = Linear720x2048(EXPERT_HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE)
        self.gate_proj = _gate.linear
        _up = Linear720x2048(EXPERT_HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE)
        self.up_proj = _up.linear
        _down = Linear2048x720(EXPERT_INTERMEDIATE_SIZE, EXPERT_HIDDEN_SIZE)
        self.down_proj = _down.linear
        self._silu = SiLU()

    def forward(self, x):
        return self.down_proj(self._silu(self.gate_proj(x)) * self.up_proj(x))


def get_inputs():
    return [torch.randn(1, 50, EXPERT_HIDDEN_SIZE)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 50, EXPERT_HIDDEN_SIZE)]
