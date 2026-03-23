"""
Component: MoE Expert Combine (Weighted Sum)
Source: data/gpt_oss.py
Abstraction Level: kernel
Parent: MLPBlock

Operations: [einsum "bec,be->bc"]

Input Shapes:
  - expert_outputs: [32, 2, 2880] dtype=bfloat16
  - expert_weights: [32, 2] dtype=bfloat16

Output Shapes:
  - output: [32, 2880] dtype=bfloat16
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, expert_outputs, expert_weights):
        return torch.einsum("bec,be->bc", expert_outputs, expert_weights)


def get_inputs():
    return [
        torch.randn(32, 2, 2880, dtype=torch.bfloat16),
        torch.randn(32, 2, dtype=torch.bfloat16),
    ]


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
