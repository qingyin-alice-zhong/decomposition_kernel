"""
Component: Text Embedding
Abstraction Level: kernel
Parent: SmolVLA
Children: []

Operations: nn.Embedding lookup

Input Shapes:
  - tokens: (1, 48) dtype=int64

Output Shapes:
  - embeddings: (1, 48, 960) dtype=float32

Weight Shapes:
  - embedding.weight: (49280, 960)
"""

import torch
import torch.nn as nn

VOCAB_SIZE = 49280
HIDDEN_SIZE = 960


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, tokens):
        return self.embedding(tokens)


def get_inputs():
    return [torch.randint(0, VOCAB_SIZE, (1, 48))]

def get_init_inputs():
    return [VOCAB_SIZE, HIDDEN_SIZE]

def get_expected_output_shape():
    return [(1, 48, HIDDEN_SIZE)]

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
