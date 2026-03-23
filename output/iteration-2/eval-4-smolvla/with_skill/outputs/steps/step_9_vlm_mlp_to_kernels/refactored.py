"""
Refactored: VLM MLP -> [linear_960x2560 x2, silu, linear_2560x960]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "children"))

import torch
import torch.nn as nn

from linear_960x2560_fp32 import Model as Linear960x2560
from linear_2560x960_fp32 import Model as Linear2560x960
from silu_fp32 import Model as SiLU

TEXT_HIDDEN_SIZE = 960
TEXT_INTERMEDIATE_SIZE = 2560


class RefactoredModel(nn.Module):
    """VLM SwiGLU MLP refactored to use child kernel modules."""
    def __init__(self):
        super().__init__()
        _gate = Linear960x2560(TEXT_HIDDEN_SIZE, TEXT_INTERMEDIATE_SIZE)
        self.gate_proj = _gate.linear
        _up = Linear960x2560(TEXT_HIDDEN_SIZE, TEXT_INTERMEDIATE_SIZE)
        self.up_proj = _up.linear
        _down = Linear2560x960(TEXT_INTERMEDIATE_SIZE, TEXT_HIDDEN_SIZE)
        self.down_proj = _down.linear
        self._silu = SiLU()

    def forward(self, x):
        return self.down_proj(self._silu(self.gate_proj(x)) * self.up_proj(x))


def get_inputs():
    return [torch.randn(1, 113, TEXT_HIDDEN_SIZE)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 113, TEXT_HIDDEN_SIZE)]
