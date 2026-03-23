"""
Component: Softmax
Abstraction Level: kernel
Parent: Attention blocks
Children: []
Operations: Softmax along last dimension
Input Shapes: - x: (1, H, L, L) dtype=float32
Output Shapes: - output: (1, H, L, L) dtype=float32
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return F.softmax(x, dim=-1)

def get_inputs():
    return [torch.randn(1, 15, 113, 113)]
def get_init_inputs():
    return []
def get_expected_output_shape():
    return [(1, 15, 113, 113)]
def run_tests():
    try:
        model = Model(); model.eval()
        with torch.no_grad():
            output = model(*get_inputs())
            assert tuple(output.shape) == tuple(get_expected_output_shape()[0])
            print("PASS"); return True
    except Exception as e:
        print(f"FAIL: {e}"); return False

if __name__ == "__main__":
    import sys; sys.exit(0 if run_tests() else 1)
