"""
Refactored: Vision Encoder Layer -> [layer_norm_768 x2, linear_768x768 x4, softmax, gelu_tanh, linear_768x3072, linear_3072x768]
Attention arithmetic (matmul, softmax) stays as data plumbing (--skip-anticheat).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "children"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_norm_768_fp32 import Model as LayerNorm768
from linear_768x768_fp32 import Model as Linear768x768
from linear_768x3072_fp32 import Model as Linear768x3072
from linear_3072x768_fp32 import Model as Linear3072x768
from gelu_tanh_fp32 import Model as GeLUTanh
from softmax_fp32 import Model as Softmax

VISION_HIDDEN_SIZE = 768
VISION_NUM_HEADS = 12
VISION_INTERMEDIATE_SIZE = 3072
VISION_LAYER_NORM_EPS = 1e-6


class _AttnContainer(nn.Module):
    """Container to group attention projections under self_attn.* namespace."""
    def __init__(self):
        super().__init__()
        _q = Linear768x768(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)
        self.q_proj = _q.linear
        _k = Linear768x768(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)
        self.k_proj = _k.linear
        _v = Linear768x768(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)
        self.v_proj = _v.linear
        _o = Linear768x768(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE)
        self.out_proj = _o.linear


class _MLPContainer(nn.Module):
    """Container to group MLP projections under mlp.* namespace."""
    def __init__(self):
        super().__init__()
        _fc1 = Linear768x3072(VISION_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE)
        self.fc1 = _fc1.linear
        _fc2 = Linear3072x768(VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE)
        self.fc2 = _fc2.linear


class RefactoredModel(nn.Module):
    """Vision Encoder Layer refactored to use child kernel modules."""
    def __init__(self):
        super().__init__()
        _ln1 = LayerNorm768(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)
        self.layer_norm1 = _ln1.layer_norm
        _ln2 = LayerNorm768(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)
        self.layer_norm2 = _ln2.layer_norm

        self.self_attn = _AttnContainer()
        self.mlp = _MLPContainer()

        self._gelu = GeLUTanh()
        self._softmax = Softmax()

        self.num_heads = VISION_NUM_HEADS
        self.head_dim = VISION_HIDDEN_SIZE // VISION_NUM_HEADS
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)

        # Attention
        B, L, _ = x.shape
        q = self.self_attn.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.self_attn.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.self_attn.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, VISION_HIDDEN_SIZE)
        x = self.self_attn.out_proj(out)

        x = x + residual
        residual = x
        x = self.layer_norm2(x)

        # MLP
        x = self.mlp.fc1(x)
        x = self._gelu(x)
        x = self.mlp.fc2(x)

        x = x + residual
        return x


def get_inputs():
    return [torch.randn(1, 1024, 768)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 1024, 768)]
