"""
Component: SigLIP Vision Encoder
Abstraction Level: layer
Parent: SmolVLA
Children: [vision_embeddings, vision_encoder_layer (x12), vision_post_layernorm]

Operations: Patch embedding, position embedding, 12x (LayerNorm + MultiHeadAttention + MLP), post LayerNorm

Input Shapes:
  - pixel_values: (1, 3, 512, 512) dtype=float32

Output Shapes:
  - image_hidden_states: (1, 1024, 768) dtype=float32

Weight Shapes:
  - patch_embedding.weight: (768, 3, 16, 16)
  - position_embedding.weight: (1024, 768)
  - layers[i].layer_norm1.weight/bias: (768,)
  - layers[i].self_attn.{q,k,v,out}_proj.weight/bias: (768, 768) / (768,)
  - layers[i].layer_norm2.weight/bias: (768,)
  - layers[i].mlp.fc1.weight/bias: (3072, 768) / (3072,)
  - layers[i].mlp.fc2.weight/bias: (768, 3072) / (768,)
  - post_layernorm.weight/bias: (768,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

VISION_HIDDEN_SIZE = 768
VISION_NUM_HEADS = 12
VISION_INTERMEDIATE_SIZE = 3072
VISION_NUM_LAYERS = 12
VISION_PATCH_SIZE = 16
VISION_IMAGE_SIZE = 512
VISION_NUM_POSITIONS = (VISION_IMAGE_SIZE // VISION_PATCH_SIZE) ** 2  # 1024
VISION_NUM_CHANNELS = 3
VISION_LAYER_NORM_EPS = 1e-6


class SigLIPVisionEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            VISION_NUM_CHANNELS, VISION_HIDDEN_SIZE,
            kernel_size=VISION_PATCH_SIZE, stride=VISION_PATCH_SIZE, padding=0
        )
        self.position_embedding = nn.Embedding(VISION_NUM_POSITIONS, VISION_HIDDEN_SIZE)
        self.register_buffer("position_ids", torch.arange(VISION_NUM_POSITIONS).unsqueeze(0))

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        patch_embeds = patch_embeds + self.position_embedding(self.position_ids)
        return patch_embeds


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


class SigLIPEncoderLayer(nn.Module):
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


class Model(nn.Module):
    """SigLIP Vision Encoder: patch embedding + 12 transformer layers + post-norm."""
    def __init__(self):
        super().__init__()
        self.embeddings = SigLIPVisionEmbeddings()
        self.layers = nn.ModuleList([SigLIPEncoderLayer() for _ in range(VISION_NUM_LAYERS)])
        self.post_layernorm = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)

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

def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
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
