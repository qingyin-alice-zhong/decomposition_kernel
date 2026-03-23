"""
Refactored: Connector -> [linear_12288x960]
Pixel shuffle reshape is data plumbing (no params).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "children"))

import torch
import torch.nn as nn

from linear_12288x960_fp32 import Model as Linear12288x960

VISION_HIDDEN_SIZE = 768
SCALE_FACTOR = 4
CONNECTOR_IN_SIZE = VISION_HIDDEN_SIZE * SCALE_FACTOR * SCALE_FACTOR  # 12288
CONNECTOR_OUT_SIZE = 960


class RefactoredModel(nn.Module):
    """Connector refactored: pixel shuffle + child linear projection."""
    def __init__(self):
        super().__init__()
        _proj = Linear12288x960(CONNECTOR_IN_SIZE, CONNECTOR_OUT_SIZE)
        self.proj = _proj.linear  # expose inner linear so weight name is proj.weight

    def forward(self, image_hidden_states):
        B, L, D = image_hidden_states.shape
        h = w = int(L ** 0.5)  # 32
        image_hidden_states = image_hidden_states.view(B, h, w, D)
        image_hidden_states = image_hidden_states.view(
            B, h // SCALE_FACTOR, SCALE_FACTOR, w // SCALE_FACTOR, SCALE_FACTOR, D
        )
        image_hidden_states = image_hidden_states.permute(0, 1, 3, 2, 4, 5)
        image_hidden_states = image_hidden_states.reshape(
            B, h // SCALE_FACTOR * w // SCALE_FACTOR, D * SCALE_FACTOR * SCALE_FACTOR
        )
        return self.proj(image_hidden_states)


def get_inputs():
    return [torch.randn(1, 1024, 768)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 64, 960)]
