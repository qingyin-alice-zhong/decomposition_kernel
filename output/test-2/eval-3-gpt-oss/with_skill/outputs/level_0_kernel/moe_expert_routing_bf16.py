"""
Component: MoE Expert Routing (TopK + Softmax)
Source: data/gpt_oss.py
Abstraction Level: kernel
Parent: MLPBlock

Operations: [topk, softmax]

Input Shapes:
  - gate_logits: [32, 4] dtype=bfloat16

Output Shapes:
  - expert_weights: [32, 2] dtype=bfloat16
  - expert_indices: [32, 2] dtype=int64
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, experts_per_token=2):
        super().__init__()
        self.experts_per_token = experts_per_token

    def forward(self, gate_logits):
        experts = torch.topk(gate_logits, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices
        return expert_weights, expert_indices


def get_inputs():
    return [torch.randn(32, 4, dtype=torch.bfloat16)]


def get_init_inputs():
    return [2]


def get_expected_output_shape():
    return [(32, 2), (32, 2)]


def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert isinstance(output, tuple) and len(output) == 2
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output[0].shape, output[1].shape]
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
