"""
Component: Linear 3072x768 (Vision MLP fc2, with bias)
Abstraction Level: kernel
Parent: Vision Encoder MLP
Children: []
Operations: nn.Linear (with bias)
Input Shapes: - x: (1, 1024, 3072) dtype=float32
Output Shapes: - output: (1, 1024, 768) dtype=float32
Weight Shapes: - linear.weight: (768, 3072), linear.bias: (768,)
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
    def forward(self, x):
        return self.linear(x)

def get_inputs():
    return [torch.randn(1, 1024, 3072)]
def get_init_inputs():
    return [3072, 768]
def get_expected_output_shape():
    return [(1, 1024, 768)]
def run_tests():
    try:
        model = Model(*get_init_inputs()); model.eval()
        with torch.no_grad():
            output = model(*get_inputs())
            assert tuple(output.shape) == tuple(get_expected_output_shape()[0])
            print("PASS"); return True
    except Exception as e:
        print(f"FAIL: {e}"); return False

if __name__ == "__main__":
    import sys; sys.exit(0 if run_tests() else 1)
