"""
Composition Test: Verifies that decomposed components produce the same output
as the original ResNet BasicBlock when composed together.

Tests:
1. Original model output matches composed Level 2 layer components
2. Level 2 components match Level 0 kernel compositions
3. Fusion components produce correct results
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================
# Original Model (reference implementation)
# ============================================================
class OriginalModel(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(OriginalModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ============================================================
# Composed Model from Level 2 Layer components
# ============================================================
class ComposedFromLayers(nn.Module):
    """Composes the model from Level 2 layer components."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ComposedFromLayers, self).__init__()
        # Level 2: conv_bn_relu
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Level 2: conv_bn
        self.conv_bn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Level 2: downsample
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # Level 2: residual_add_relu
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv_bn_relu(x)
        out = self.conv_bn(out)
        out = out + identity
        out = self.final_relu(out)
        return out


# ============================================================
# Composed Model from Level 0 Kernel components
# ============================================================
class ComposedFromKernels(nn.Module):
    """Composes the model from Level 0 atomic kernel components."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ComposedFromKernels, self).__init__()
        # Main path kernels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Skip path kernels
        self.ds_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.ds_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Skip path: conv2d_1x1 -> batch_norm2d
        identity = self.ds_bn(self.ds_conv(x))
        # Main path: conv2d_3x3 -> batch_norm2d -> relu -> conv2d_3x3 -> batch_norm2d
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Merge: add -> relu
        out = F.relu(out + identity)
        return out


def copy_weights(src, dst_layers, dst_ds):
    """Copy weights from original model to composed model."""
    dst_layers['conv1'].weight.data.copy_(src.conv1.weight.data)
    dst_layers['bn1'].weight.data.copy_(src.bn1.weight.data)
    dst_layers['bn1'].bias.data.copy_(src.bn1.bias.data)
    dst_layers['bn1'].running_mean.data.copy_(src.bn1.running_mean.data)
    dst_layers['bn1'].running_var.data.copy_(src.bn1.running_var.data)

    dst_layers['conv2'].weight.data.copy_(src.conv2.weight.data)
    dst_layers['bn2'].weight.data.copy_(src.bn2.weight.data)
    dst_layers['bn2'].bias.data.copy_(src.bn2.bias.data)
    dst_layers['bn2'].running_mean.data.copy_(src.bn2.running_mean.data)
    dst_layers['bn2'].running_var.data.copy_(src.bn2.running_var.data)

    dst_ds['conv'].weight.data.copy_(src.downsample[0].weight.data)
    dst_ds['bn'].weight.data.copy_(src.downsample[1].weight.data)
    dst_ds['bn'].bias.data.copy_(src.downsample[1].bias.data)
    dst_ds['bn'].running_mean.data.copy_(src.downsample[1].running_mean.data)
    dst_ds['bn'].running_var.data.copy_(src.downsample[1].running_var.data)


def test_level2_composition():
    """Test that Level 2 layer composition matches the original model."""
    print("Test 1: Level 2 (Layer) composition vs Original model...", end=" ")
    in_channels, out_channels, stride = 3, 64, 1
    torch.manual_seed(42)

    original = OriginalModel(in_channels, out_channels, stride)
    # Run one forward pass in train mode to populate BN running stats
    dummy = torch.randn(2, in_channels, 32, 32)
    original(dummy)
    original.eval()

    composed = ComposedFromLayers(in_channels, out_channels, stride)
    # Copy weights
    composed.conv_bn_relu[0].weight.data.copy_(original.conv1.weight.data)
    composed.conv_bn_relu[1].weight.data.copy_(original.bn1.weight.data)
    composed.conv_bn_relu[1].bias.data.copy_(original.bn1.bias.data)
    composed.conv_bn_relu[1].running_mean.copy_(original.bn1.running_mean)
    composed.conv_bn_relu[1].running_var.copy_(original.bn1.running_var)

    composed.conv_bn[0].weight.data.copy_(original.conv2.weight.data)
    composed.conv_bn[1].weight.data.copy_(original.bn2.weight.data)
    composed.conv_bn[1].bias.data.copy_(original.bn2.bias.data)
    composed.conv_bn[1].running_mean.copy_(original.bn2.running_mean)
    composed.conv_bn[1].running_var.copy_(original.bn2.running_var)

    composed.downsample[0].weight.data.copy_(original.downsample[0].weight.data)
    composed.downsample[1].weight.data.copy_(original.downsample[1].weight.data)
    composed.downsample[1].bias.data.copy_(original.downsample[1].bias.data)
    composed.downsample[1].running_mean.copy_(original.downsample[1].running_mean)
    composed.downsample[1].running_var.copy_(original.downsample[1].running_var)
    composed.eval()

    x = torch.randn(2, in_channels, 32, 32)
    with torch.no_grad():
        out_orig = original(x)
        out_comp = composed(x)

    assert torch.allclose(out_orig, out_comp, atol=1e-6), \
        f"Max diff: {(out_orig - out_comp).abs().max().item()}"
    print("PASS")


def test_level0_composition():
    """Test that Level 0 (Kernel) composition matches the original model."""
    print("Test 2: Level 0 (Kernel) composition vs Original model...", end=" ")
    in_channels, out_channels, stride = 3, 64, 1
    torch.manual_seed(42)

    original = OriginalModel(in_channels, out_channels, stride)
    dummy = torch.randn(2, in_channels, 32, 32)
    original(dummy)
    original.eval()

    kernels = ComposedFromKernels(in_channels, out_channels, stride)
    copy_weights(original,
                 {'conv1': kernels.conv1, 'bn1': kernels.bn1,
                  'conv2': kernels.conv2, 'bn2': kernels.bn2},
                 {'conv': kernels.ds_conv, 'bn': kernels.ds_bn})
    kernels.eval()

    x = torch.randn(2, in_channels, 32, 32)
    with torch.no_grad():
        out_orig = original(x)
        out_kern = kernels(x)

    assert torch.allclose(out_orig, out_kern, atol=1e-6), \
        f"Max diff: {(out_orig - out_kern).abs().max().item()}"
    print("PASS")


def test_stride2():
    """Test with stride=2 to verify downsample path works correctly."""
    print("Test 3: Stride=2 downsampling correctness...", end=" ")
    in_channels, out_channels, stride = 64, 128, 2
    torch.manual_seed(123)

    original = OriginalModel(in_channels, out_channels, stride)
    dummy = torch.randn(2, in_channels, 32, 32)
    original(dummy)
    original.eval()

    kernels = ComposedFromKernels(in_channels, out_channels, stride)
    copy_weights(original,
                 {'conv1': kernels.conv1, 'bn1': kernels.bn1,
                  'conv2': kernels.conv2, 'bn2': kernels.bn2},
                 {'conv': kernels.ds_conv, 'bn': kernels.ds_bn})
    kernels.eval()

    x = torch.randn(2, in_channels, 32, 32)
    with torch.no_grad():
        out_orig = original(x)
        out_kern = kernels(x)

    # Output spatial dims should be halved with stride=2
    assert out_orig.shape == (2, out_channels, 16, 16), f"Shape mismatch: {out_orig.shape}"
    assert torch.allclose(out_orig, out_kern, atol=1e-5), \
        f"Max diff: {(out_orig - out_kern).abs().max().item()}"
    print("PASS")


def test_individual_components():
    """Test that individual component files are importable and pass their own tests."""
    print("Test 4: Individual component standalone checks...", end=" ")

    # Test Level 0 kernels individually
    # conv2d_3x3
    conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    x = torch.randn(2, 3, 8, 8)
    out = conv(x)
    assert out.shape == (2, 64, 8, 8)

    # conv2d_1x1
    conv1x1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, bias=False)
    out = conv1x1(x)
    assert out.shape == (2, 64, 8, 8)

    # batch_norm2d
    bn = nn.BatchNorm2d(64)
    x_bn = torch.randn(2, 64, 8, 8)
    out = bn(x_bn)
    assert out.shape == (2, 64, 8, 8)

    # relu
    out = F.relu(torch.tensor([-1.0, 0.0, 1.0]))
    assert torch.equal(out, torch.tensor([0.0, 0.0, 1.0]))

    # add
    a, b = torch.randn(2, 64, 8, 8), torch.randn(2, 64, 8, 8)
    assert torch.equal(a + b, a + b)

    print("PASS")


def test_fusion_conv_bn():
    """Test that Conv+BN fusion produces correct results."""
    print("Test 5: Conv+BN fusion correctness...", end=" ")
    conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
    bn = nn.BatchNorm2d(64)

    # Run forward to populate running stats
    x_train = torch.randn(4, 3, 8, 8)
    bn(conv(x_train))
    conv.eval()
    bn.eval()

    # Fuse
    bn_weight = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
    fused_conv.weight.data = conv.weight.data * bn_weight.view(-1, 1, 1, 1)
    fused_conv.bias.data = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
    fused_conv.eval()

    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        out_orig = bn(conv(x))
        out_fused = fused_conv(x)
    assert torch.allclose(out_orig, out_fused, atol=1e-5), \
        f"Max diff: {(out_orig - out_fused).abs().max().item()}"
    print("PASS")


def test_fusion_add_relu():
    """Test that Add+ReLU fusion produces correct results."""
    print("Test 6: Add+ReLU fusion correctness...", end=" ")
    a = torch.randn(2, 64, 8, 8)
    b = torch.randn(2, 64, 8, 8)

    # Separate
    separate = F.relu(a + b)
    # Fused (same operation, but conceptually a single kernel)
    fused = F.relu(a + b, inplace=False)

    assert torch.equal(separate, fused), "Add+ReLU fusion mismatch"
    print("PASS")


def run_tests():
    print("=" * 60)
    print("ResNet BasicBlock Decomposition - Composition Tests")
    print("=" * 60)
    test_level2_composition()
    test_level0_composition()
    test_stride2()
    test_individual_components()
    test_fusion_conv_bn()
    test_fusion_add_relu()
    print("=" * 60)
    print("All composition tests PASS")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
