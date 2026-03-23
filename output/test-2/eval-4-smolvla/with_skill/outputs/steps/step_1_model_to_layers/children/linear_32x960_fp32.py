"""
Component: State Projection
Abstraction Level: kernel
Parent: SmolVLA
Children: []

Operations: nn.Linear

Input Shapes:
  - state: (1, 32) dtype=float32

Output Shapes:
  - output: (1, 960) dtype=float32

Weight Shapes:
  - linear.weight: (960, 32)
  - linear.bias: (960,)
"""

import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

def get_inputs():
    return [torch.randn(1, 32)]

def get_init_inputs():
    return [32, 960]

def get_expected_output_shape():
    return [(1, 960)]

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
