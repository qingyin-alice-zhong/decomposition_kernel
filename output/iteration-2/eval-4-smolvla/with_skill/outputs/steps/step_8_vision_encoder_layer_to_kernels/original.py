"""
Component: SigLIP Vision Encoder Layer
Abstraction Level: fusion
Parent: SigLIP Vision Encoder
Children: [layer_norm_768_fp32, linear_768x768_fp32, softmax_fp32, gelu_tanh_fp32,
           linear_768x3072_fp32, linear_3072x768_fp32]

Operations: LayerNorm, Q/K/V projection, scaled dot-product attention, output projection,
            residual add, LayerNorm, MLP (fc1 + GELU + fc2), residual add

Input Shapes:
  - hidden_states: (1, 1024, 768) dtype=float32

Output Shapes:
  - output: (1, 1024, 768) dtype=float32

Weight Shapes:
  - layer_norm1.weight/bias: (768,)
  - self_attn.q_proj.weight/bias: (768, 768) / (768,)
  - self_attn.k_proj.weight/bias: (768, 768) / (768,)
  - self_attn.v_proj.weight/bias: (768, 768) / (768,)
  - self_attn.out_proj.weight/bias: (768, 768) / (768,)
  - layer_norm2.weight/bias: (768,)
  - mlp.fc1.weight/bias: (3072, 768) / (3072,)
  - mlp.fc2.weight/bias: (768, 3072) / (768,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VISION_HIDDEN_SIZE = 768
VISION_NUM_HEADS = 12
VISION_INTERMEDIATE_SIZE = 3072
VISION_LAYER_NORM_EPS = 1e-6


class SigLIPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = VISION_NUM_HEADS
        self.head_dim = VISION_HIDDEN_SIZE // VISION_NUM_HEADS
        self.q_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.k_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.v_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.out_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, VISION_HIDDEN_SIZE)
        return self.out_proj(out)


class SigLIPMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(VISION_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE, bias=True)
        self.fc2 = nn.Linear(VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class Model(nn.Module):
    """Single SigLIP Vision Encoder Layer: LN + Attn + Residual + LN + MLP + Residual."""
    def __init__(self):
        super().__init__()
        self.self_attn = SigLIPAttention()
        self.layer_norm1 = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)
        self.mlp = SigLIPMLP()
        self.layer_norm2 = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = x + residual
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


def get_inputs():
    return [torch.randn(1, 1024, 768)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 1024, 768)]

def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            expected = get_expected_output_shape()
            assert tuple(output.shape) == tuple(expected[0]), f"Got {output.shape}, expected {expected[0]}"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): [{output.shape}]")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
