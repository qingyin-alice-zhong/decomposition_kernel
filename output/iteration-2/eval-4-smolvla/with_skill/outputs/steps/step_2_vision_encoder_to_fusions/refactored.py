"""
Refactored: SigLIP Vision Encoder -> [vision_embeddings, vision_encoder_layer x12, layer_norm_768]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "children"))

import torch
import torch.nn as nn

from vision_embeddings import Model as VisionEmbeddings
from vision_encoder_layer import Model as VisionEncoderLayer
from layer_norm_768_fp32 import Model as LayerNorm768

VISION_NUM_LAYERS = 12
VISION_LAYER_NORM_EPS = 1e-6


class RefactoredModel(nn.Module):
    """SigLIP Vision Encoder refactored to use child modules."""
    def __init__(self):
        super().__init__()
        self.embeddings = VisionEmbeddings()
        self.layers = nn.ModuleList([VisionEncoderLayer() for _ in range(VISION_NUM_LAYERS)])
        # Expose the inner LayerNorm directly so weight names match original
        _post_ln = LayerNorm768(768, eps=VISION_LAYER_NORM_EPS)
        self.post_layernorm = _post_ln.layer_norm

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)
        for layer in self.layers:
            x = layer(x)
        x = self.post_layernorm(x)
        return x


def get_inputs():
    return [torch.randn(1, 3, 512, 512)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 1024, 768)]
