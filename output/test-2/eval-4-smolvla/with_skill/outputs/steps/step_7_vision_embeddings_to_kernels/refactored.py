"""
Refactored: Vision Embeddings -> [conv2d_3x768_16x16, embedding_1024x768]
Flatten/transpose/add are data plumbing.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "children"))

import torch
import torch.nn as nn

from conv2d_3x768_16x16_fp32 import Model as Conv2dPatch
from embedding_1024x768_fp32 import Model as PosEmbedding

VISION_HIDDEN_SIZE = 768
VISION_PATCH_SIZE = 16
VISION_IMAGE_SIZE = 512
VISION_NUM_POSITIONS = (VISION_IMAGE_SIZE // VISION_PATCH_SIZE) ** 2  # 1024
VISION_NUM_CHANNELS = 3


class RefactoredModel(nn.Module):
    """Vision Embeddings refactored to use child kernel modules."""
    def __init__(self):
        super().__init__()
        _conv = Conv2dPatch(VISION_NUM_CHANNELS, VISION_HIDDEN_SIZE, VISION_PATCH_SIZE, VISION_PATCH_SIZE)
        self.patch_embedding = _conv.conv  # expose inner Conv2d
        _pos = PosEmbedding(VISION_NUM_POSITIONS, VISION_HIDDEN_SIZE)
        self.position_embedding = _pos.embedding  # expose inner Embedding
        self.register_buffer("position_ids", torch.arange(VISION_NUM_POSITIONS).unsqueeze(0))

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        patch_embeds = patch_embeds + self.position_embedding(self.position_ids)
        return patch_embeds


def get_inputs():
    return [torch.randn(1, 3, 512, 512)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 1024, 768)]
