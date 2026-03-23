"""
Component: SigLIP Vision Embeddings (Patch Conv + Position Embedding)
Abstraction Level: fusion
Parent: SigLIP Vision Encoder
Children: [conv2d_3x768_16x16_fp32, embedding_1024x768_fp32]

Operations: Conv2d patch embedding, flatten, transpose, position embedding lookup, add

Input Shapes:
  - pixel_values: (1, 3, 512, 512) dtype=float32

Output Shapes:
  - patch_embeddings: (1, 1024, 768) dtype=float32

Weight Shapes:
  - patch_embedding.weight: (768, 3, 16, 16)
  - patch_embedding.bias: (768,)
  - position_embedding.weight: (1024, 768)
"""

import torch
import torch.nn as nn

VISION_HIDDEN_SIZE = 768
VISION_PATCH_SIZE = 16
VISION_IMAGE_SIZE = 512
VISION_NUM_POSITIONS = (VISION_IMAGE_SIZE // VISION_PATCH_SIZE) ** 2  # 1024
VISION_NUM_CHANNELS = 3


class Model(nn.Module):
    """SigLIP Vision Embeddings: patch conv + position embedding."""
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            VISION_NUM_CHANNELS, VISION_HIDDEN_SIZE,
            kernel_size=VISION_PATCH_SIZE, stride=VISION_PATCH_SIZE, padding=0
        )
        self.position_embedding = nn.Embedding(VISION_NUM_POSITIONS, VISION_HIDDEN_SIZE)
        self.register_buffer("position_ids", torch.arange(VISION_NUM_POSITIONS).unsqueeze(0))

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        patch_embeds = patch_embeds + self.position_embedding(self.position_ids)
        return patch_embeds


def get_inputs():
    return [torch.randn(1, 3, 512, 512)]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, 1024, 768)]

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
