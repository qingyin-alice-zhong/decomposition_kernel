"""
Component: LayerNorm (768-dim, vision encoder)
Abstraction Level: kernel
Parent: Vision Encoder Layer
Children: []
Operations: nn.LayerNorm
Input Shapes: - x: (1, 1024, 768) dtype=float32
Output Shapes: - output: (1, 1024, 768) dtype=float32
Weight Shapes: - weight: (768,), bias: (768,)
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
    def forward(self, x):
        return self.layer_norm(x)

def get_inputs():
    return [torch.randn(1, 1024, 768)]
def get_init_inputs():
    return [768]
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
