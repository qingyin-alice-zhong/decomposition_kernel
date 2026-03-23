"""
Component: SmolVLM Connector (Pixel Shuffle + Linear Projection)
Abstraction Level: fusion
Parent: SmolVLA
Children: [linear_12288x960_fp32]

Operations: Pixel shuffle reshape, linear projection

Input Shapes:
  - image_hidden_states: (1, 1024, 768) dtype=float32

Output Shapes:
  - output: (1, 64, 960) dtype=float32

Weight Shapes:
  - proj.weight: (960, 12288)
"""

import torch
import torch.nn as nn

VISION_HIDDEN_SIZE = 768
SCALE_FACTOR = 4
CONNECTOR_IN_SIZE = VISION_HIDDEN_SIZE * SCALE_FACTOR * SCALE_FACTOR  # 12288
CONNECTOR_OUT_SIZE = 960  # TEXT_HIDDEN_SIZE


class Model(nn.Module):
    """Pixel shuffle + linear projection connector."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(CONNECTOR_IN_SIZE, CONNECTOR_OUT_SIZE, bias=False)

    def forward(self, image_hidden_states):
        B, L, D = image_hidden_states.shape
        h = w = int(L ** 0.5)  # 32
        image_hidden_states = image_hidden_states.view(B, h, w, D)
        image_hidden_states = image_hidden_states.view(
            B, h // SCALE_FACTOR, SCALE_FACTOR, w // SCALE_FACTOR, SCALE_FACTOR, D
        )
        image_hidden_states = image_hidden_states.permute(0, 1, 3, 2, 4, 5)
        image_hidden_states = image_hidden_states.reshape(
            B, h // SCALE_FACTOR * w // SCALE_FACTOR, D * SCALE_FACTOR * SCALE_FACTOR
        )
        return self.proj(image_hidden_states)


def get_inputs():
    return [torch.randn(1, 1024, 768)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 64, 960)]

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
