"""
Level 1 (Fusion): Fused Conv2d + BatchNorm2d
Represents the fusion of convolution and batch normalization into a single operation.
In inference mode, BN can be folded into convolution weights.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))

    @staticmethod
    def fuse(conv, bn):
        """Fuse Conv2d and BatchNorm2d into a single Conv2d for inference."""
        fused_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            stride=conv.stride, padding=conv.padding, bias=True
        )
        # Fold BN into conv weights
        bn_weight = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_conv.weight.data = conv.weight.data * bn_weight.view(-1, 1, 1, 1)
        fused_conv.bias.data = (bn.bias - bn.weight * bn.running_mean /
                                torch.sqrt(bn.running_var + bn.eps))
        return fused_conv


# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10


def get_inputs():
    return [torch.rand(batch_size, in_channels, 56, 56)]


def get_init_inputs():
    return [in_channels, out_channels, 3, stride, 1]


def run_tests():
    model = Model(*get_init_inputs()).eval()
    inputs = get_inputs()
    with torch.no_grad():
        output = model(*inputs)
    assert output.shape == (batch_size, out_channels, 56, 56), f"Shape mismatch: {output.shape}"

    # Test fusion produces same output
    with torch.no_grad():
        fused = Model.fuse(model.conv, model.bn)
        fused_output = fused(inputs[0])
    assert torch.allclose(output, fused_output, atol=1e-5), "Fused output does not match"
    print("PASS")


if __name__ == "__main__":
    run_tests()
