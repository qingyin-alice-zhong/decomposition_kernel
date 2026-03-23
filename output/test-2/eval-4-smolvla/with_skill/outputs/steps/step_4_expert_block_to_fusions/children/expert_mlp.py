"""
Component: Expert SwiGLU MLP (LlamaMLP)
Abstraction Level: fusion
Parent: Expert Transformer Block
Children: [linear_720x2048_fp32, linear_2048x720_fp32, silu_fp32]

Operations: gate_proj + SiLU, up_proj, elementwise mul, down_proj

Input Shapes:
  - x: (1, 50, 720) dtype=float32

Output Shapes:
  - output: (1, 50, 720) dtype=float32

Weight Shapes:
  - gate_proj.weight: (2048, 720)
  - up_proj.weight: (2048, 720)
  - down_proj.weight: (720, 2048)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERT_HIDDEN_SIZE = 720
EXPERT_INTERMEDIATE_SIZE = 2048


class Model(nn.Module):
    """Expert SwiGLU MLP: gate + SiLU * up -> down."""
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(EXPERT_INTERMEDIATE_SIZE, EXPERT_HIDDEN_SIZE, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def get_inputs():
    return [torch.randn(1, 50, EXPERT_HIDDEN_SIZE)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 50, EXPERT_HIDDEN_SIZE)]

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
            assert tuple(output.shape) == tuple(expected[0])
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
