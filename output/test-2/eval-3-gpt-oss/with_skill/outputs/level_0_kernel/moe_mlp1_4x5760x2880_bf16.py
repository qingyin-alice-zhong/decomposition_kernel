"""
Component: MoE MLP1 (Expert Linear + Bias via Einsum)
Source: data/gpt_oss.py
Abstraction Level: kernel
Parent: MLPBlock

Operations: [Parameter indexing, einsum "beck,bk->bec", bias add]

Input Shapes:
  - x: [32, 2880] dtype=bfloat16
  - expert_indices: [32, 2] dtype=int64

Output Shapes:
  - output: [32, 2, 5760] dtype=bfloat16

Weight Shapes:
  - mlp1_weight: [4, 5760, 2880]
  - mlp1_bias: [4, 5760]
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_experts=4, intermediate_size=2880, hidden_size=2880):
        super().__init__()
        self.mlp1_weight = nn.Parameter(
            torch.empty((num_experts, intermediate_size * 2, hidden_size), dtype=torch.bfloat16)
        )
        self.mlp1_bias = nn.Parameter(
            torch.empty((num_experts, intermediate_size * 2), dtype=torch.bfloat16)
        )

    def forward(self, x, expert_indices):
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        t = torch.einsum("beck,bk->bec", mlp1_weight, x) + mlp1_bias
        return t


def get_inputs():
    return [
        torch.randn(32, 2880, dtype=torch.bfloat16),
        torch.randint(0, 4, (32, 2), dtype=torch.int64),
    ]


def get_init_inputs():
    return [4, 2880, 2880]


def get_expected_output_shape():
    return [(32, 2, 5760)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
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
