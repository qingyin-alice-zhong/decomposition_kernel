"""
Component: Action-Time MLP Fusion
Abstraction Level: fusion
Parent: SmolVLA
Children: [linear_1440x720_fp32, silu_fp32, linear_720x720_fp32]

Operations: Linear projection, SiLU activation, Linear projection

Input Shapes:
  - action_time_emb: (1, 50, 1440) dtype=float32

Output Shapes:
  - output: (1, 50, 720) dtype=float32

Weight Shapes:
  - mlp_in.weight: (720, 1440)
  - mlp_in.bias: (720,)
  - mlp_out.weight: (720, 720)
  - mlp_out.bias: (720,)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERT_HIDDEN_SIZE = 720
CHUNK_SIZE = 50


class Model(nn.Module):
    """Action-Time MLP: Linear + SiLU + Linear."""
    def __init__(self):
        super().__init__()
        self.mlp_in = nn.Linear(EXPERT_HIDDEN_SIZE * 2, EXPERT_HIDDEN_SIZE)
        self.mlp_out = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_HIDDEN_SIZE)

    def forward(self, action_time_emb):
        x = self.mlp_in(action_time_emb)
        x = F.silu(x)
        x = self.mlp_out(x)
        return x


def get_inputs():
    return [torch.randn(1, CHUNK_SIZE, EXPERT_HIDDEN_SIZE * 2)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, CHUNK_SIZE, EXPERT_HIDDEN_SIZE)]

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
