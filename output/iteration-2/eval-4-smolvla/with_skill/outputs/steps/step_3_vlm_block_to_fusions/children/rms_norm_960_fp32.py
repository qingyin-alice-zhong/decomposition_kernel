"""
Component: RMSNorm (960-dim)
Abstraction Level: kernel
Parent: VLM Transformer Block / SmolVLA final norm
Children: []

Operations: RMSNorm

Input Shapes:
  - x: (1, L, 960) dtype=float32

Output Shapes:
  - output: (1, L, 960) dtype=float32

Weight Shapes:
  - weight: (960,)
"""

import torch
import torch.nn as nn

HIDDEN_SIZE = 960
RMS_NORM_EPS = 1e-5


class Model(nn.Module):
    def __init__(self, hidden_size, eps=RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


def get_inputs():
    return [torch.randn(1, 113, HIDDEN_SIZE)]

def get_init_inputs():
    return [HIDDEN_SIZE]

def get_expected_output_shape():
    return [(1, 113, HIDDEN_SIZE)]

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
