"""
Component: Vision Position Embedding
Abstraction Level: kernel
Parent: Vision Encoder Embeddings
Children: []
Operations: nn.Embedding
Input Shapes: - position_ids: (1, 1024) dtype=int64
Output Shapes: - output: (1, 1024, 768) dtype=float32
Weight Shapes: - embedding.weight: (1024, 768)
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_positions, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(num_positions, hidden_size)
    def forward(self, position_ids):
        return self.embedding(position_ids)

def get_inputs():
    return [torch.arange(1024).unsqueeze(0)]
def get_init_inputs():
    return [1024, 768]
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
