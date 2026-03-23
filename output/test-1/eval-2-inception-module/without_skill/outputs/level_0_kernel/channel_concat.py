"""
Kernel-level component: Channel Concatenation.
Concatenates a list of tensors along the channel dimension (dim=1).
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Standalone channel concatenation operation."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, *branches):
        return torch.cat(branches, dim=1)


batch_size = 10
channels_list = [192, 208, 48, 64]  # out_1x1, out_3x3, out_5x5, pool_proj
height = 224
width = 224


def get_inputs():
    return [torch.rand(batch_size, c, height, width) for c in channels_list]


def get_init_inputs():
    return []


def run_tests():
    model = Model(*get_init_inputs())
    inputs = get_inputs()
    output = model(*inputs)
    total_channels = sum(channels_list)
    assert output.shape == (batch_size, total_channels, height, width), \
        f"Expected shape {(batch_size, total_channels, height, width)}, got {output.shape}"
    # Verify concatenation order is preserved
    offset = 0
    for i, c in enumerate(channels_list):
        assert torch.equal(output[:, offset:offset+c], inputs[i]), \
            f"Branch {i} data not correctly placed in concatenated output"
        offset += c
    print("PASS")


if __name__ == "__main__":
    run_tests()
