"""
Component: VLM SwiGLU MLP (LlamaMLP)
Abstraction Level: fusion
Parent: VLM Transformer Block
Children: [linear_960x2560_fp32, linear_2560x960_fp32, silu_fp32]

Operations: gate_proj + SiLU, up_proj, elementwise mul, down_proj

Input Shapes:
  - x: (1, L, 960) dtype=float32

Output Shapes:
  - output: (1, L, 960) dtype=float32

Weight Shapes:
  - gate_proj.weight: (2560, 960)
  - up_proj.weight: (2560, 960)
  - down_proj.weight: (960, 2560)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_HIDDEN_SIZE = 960
TEXT_INTERMEDIATE_SIZE = 2560


class Model(nn.Module):
    """VLM SwiGLU MLP: gate + SiLU * up -> down."""
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(TEXT_INTERMEDIATE_SIZE, TEXT_HIDDEN_SIZE, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def get_inputs():
    return [torch.randn(1, 113, TEXT_HIDDEN_SIZE)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 113, TEXT_HIDDEN_SIZE)]

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
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
