"""
Component: MLPBlock (MoE)
Source: data/gpt_oss.py
Abstraction Level: fusion
Parent: TransformerBlock

Operations: [RMSNorm, Linear (gate), TopK, Softmax, Einsum (MLP1), SwiGLU, Einsum (MLP2), Einsum (combine), residual add]

Input Shapes:
  - x: [32, 2880] dtype=bfloat16

Output Shapes:
  - output: [32, 2880] dtype=bfloat16

Weight Shapes:
  - norm.scale: [2880]
  - gate.weight: [4, 2880], gate.bias: [4]
  - mlp1_weight: [4, 5760, 2880]
  - mlp1_bias: [4, 5760]
  - mlp2_weight: [4, 2880, 2880]
  - mlp2_bias: [4, 2880]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features, device=device, dtype=torch.float32))

    def forward(self, x):
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def swiglu(x, alpha=1.702, limit=7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return out_glu * (x_linear + 1)


class Model(nn.Module):
    """MLPBlock with MoE routing, SwiGLU activation, and residual connection."""
    def __init__(self):
        super().__init__()
        hidden_size = 2880
        intermediate_size = 2880
        num_experts = 4
        experts_per_token = 2
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.swiglu_limit = 7.0
        self.norm = RMSNorm(hidden_size)
        self.gate = nn.Linear(hidden_size, num_experts, dtype=torch.bfloat16)
        self.mlp1_weight = nn.Parameter(torch.empty((num_experts, intermediate_size * 2, hidden_size), dtype=torch.bfloat16))
        self.mlp1_bias = nn.Parameter(torch.empty((num_experts, intermediate_size * 2), dtype=torch.bfloat16))
        self.mlp2_weight = nn.Parameter(torch.empty((num_experts, hidden_size, intermediate_size), dtype=torch.bfloat16))
        self.mlp2_bias = nn.Parameter(torch.empty((num_experts, hidden_size), dtype=torch.bfloat16))

    def forward(self, x):
        t = self.norm(x)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)
        t += mlp2_bias
        t = torch.einsum("bec,be->bc", t, expert_weights)
        return x + t


def get_inputs():
    return [torch.randn(32, 2880, dtype=torch.bfloat16)]


def get_init_inputs():
    return []


def get_expected_output_shape():
    return [(32, 2880)]


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
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), f"Shape mismatch: {actual} vs {expected}"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
