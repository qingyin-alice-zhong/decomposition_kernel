"""
Component: Vision Patch Embedding Conv2d
Abstraction Level: kernel
Parent: Vision Encoder Embeddings
Children: []
Operations: nn.Conv2d
Input Shapes: - x: (1, 3, 512, 512) dtype=float32
Output Shapes: - output: (1, 768, 32, 32) dtype=float32
Weight Shapes: - conv.weight: (768, 3, 16, 16), conv.bias: (768,)
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        return self.conv(x)

def get_inputs():
    return [torch.randn(1, 3, 512, 512)]
def get_init_inputs():
    return [3, 768, 16, 16]
def get_expected_output_shape():
    return [(1, 768, 32, 32)]
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
