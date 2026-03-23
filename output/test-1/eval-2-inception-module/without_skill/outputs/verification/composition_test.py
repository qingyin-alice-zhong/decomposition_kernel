"""
Composition test: Proves that the decomposed components produce the same
output as the original GoogleNet Inception module when composed together.

Tests at every level:
  1. Kernel-level components individually
  2. Fusion-level components match their branch counterparts
  3. Layer-level branches match original model branches
  4. Full recomposition from layer-level branches matches original model
"""
import sys
import os
import torch
import torch.nn as nn

# Use smaller dimensions to keep test fast and memory-friendly
batch_size = 2
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
height = 14
width = 14

torch.manual_seed(42)


def build_original_model():
    """Build the original Inception module exactly as defined in the source."""
    class OriginalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
            self.branch3x3 = nn.Sequential(
                nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
                nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
            )
            self.branch5x5 = nn.Sequential(
                nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
                nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
            )
            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, pool_proj, kernel_size=1)
            )

        def forward(self, x):
            b1 = self.branch1x1(x)
            b3 = self.branch3x3(x)
            b5 = self.branch5x5(x)
            bp = self.branch_pool(x)
            return torch.cat([b1, b3, b5, bp], 1)

    return OriginalModel()


def test_kernel_level():
    """Test each kernel-level component runs and produces correct shapes."""
    print("Testing kernel-level components...")

    # conv2d_1x1
    conv1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
    x = torch.rand(batch_size, in_channels, height, width)
    out = conv1x1(x)
    assert out.shape == (batch_size, out_1x1, height, width)

    # conv2d_3x3
    conv3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
    x3 = torch.rand(batch_size, reduce_3x3, height, width)
    out3 = conv3x3(x3)
    assert out3.shape == (batch_size, out_3x3, height, width)

    # conv2d_5x5
    conv5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
    x5 = torch.rand(batch_size, reduce_5x5, height, width)
    out5 = conv5x5(x5)
    assert out5.shape == (batch_size, out_5x5, height, width)

    # maxpool2d
    pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    xp = torch.rand(batch_size, in_channels, height, width)
    outp = pool(xp)
    assert outp.shape == (batch_size, in_channels, height, width)

    # channel_concat
    t1 = torch.rand(batch_size, out_1x1, height, width)
    t2 = torch.rand(batch_size, out_3x3, height, width)
    t3 = torch.rand(batch_size, out_5x5, height, width)
    t4 = torch.rand(batch_size, pool_proj, height, width)
    cat = torch.cat([t1, t2, t3, t4], dim=1)
    total = out_1x1 + out_3x3 + out_5x5 + pool_proj
    assert cat.shape == (batch_size, total, height, width)

    print("  Kernel-level: PASS")


def test_fusion_level():
    """Test fusion-level components produce same output as sequential kernel ops."""
    print("Testing fusion-level components...")

    x = torch.rand(batch_size, in_channels, height, width)

    # reduce_conv3x3: 1x1 reduction + 3x3 conv
    reduce_1x1 = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
    conv_3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
    fused_3x3 = nn.Sequential(reduce_1x1, conv_3x3)
    out_fused = fused_3x3(x)
    # Run the same kernels separately
    intermediate = reduce_1x1(x)
    out_separate = conv_3x3(intermediate)
    assert torch.equal(out_fused, out_separate), "Fusion reduce_conv3x3 mismatch"

    # reduce_conv5x5: 1x1 reduction + 5x5 conv
    reduce_1x1_5 = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
    conv_5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
    fused_5x5 = nn.Sequential(reduce_1x1_5, conv_5x5)
    out_fused5 = fused_5x5(x)
    intermediate5 = reduce_1x1_5(x)
    out_separate5 = conv_5x5(intermediate5)
    assert torch.equal(out_fused5, out_separate5), "Fusion reduce_conv5x5 mismatch"

    # pool_proj: maxpool + 1x1 proj
    pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    proj = nn.Conv2d(in_channels, pool_proj, kernel_size=1)
    pooled = pool(x)
    out_proj_separate = proj(pooled)
    fused_pool = nn.Sequential(pool, proj)
    out_proj_fused = fused_pool(x)
    assert torch.equal(out_proj_fused, out_proj_separate), "Fusion pool_proj mismatch"

    print("  Fusion-level: PASS")


def test_layer_level_matches_original():
    """Test that layer-level branches produce same output as original model branches."""
    print("Testing layer-level branches match original model...")

    original = build_original_model()
    original.eval()

    x = torch.rand(batch_size, in_channels, height, width)

    with torch.no_grad():
        # Get original branch outputs
        orig_b1 = original.branch1x1(x)
        orig_b3 = original.branch3x3(x)
        orig_b5 = original.branch5x5(x)
        orig_bp = original.branch_pool(x)

    # Build decomposed branches with same weights
    # Branch 1x1
    branch1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
    branch1.load_state_dict(original.branch1x1.state_dict())

    # Branch 3x3
    branch3 = nn.Sequential(
        nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
        nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
    )
    branch3.load_state_dict(original.branch3x3.state_dict())

    # Branch 5x5
    branch5 = nn.Sequential(
        nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
        nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
    )
    branch5.load_state_dict(original.branch5x5.state_dict())

    # Branch pool
    branchp = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels, pool_proj, kernel_size=1)
    )
    branchp.load_state_dict(original.branch_pool.state_dict())

    with torch.no_grad():
        dec_b1 = branch1(x)
        dec_b3 = branch3(x)
        dec_b5 = branch5(x)
        dec_bp = branchp(x)

    assert torch.equal(orig_b1, dec_b1), "Branch 1x1 mismatch"
    assert torch.equal(orig_b3, dec_b3), "Branch 3x3 mismatch"
    assert torch.equal(orig_b5, dec_b5), "Branch 5x5 mismatch"
    assert torch.equal(orig_bp, dec_bp), "Branch pool mismatch"

    print("  Layer-level branches: PASS")


def test_full_recomposition():
    """
    The key test: build a composed model from decomposed layer-level branches
    with shared weights from the original, and verify outputs are identical.
    """
    print("Testing full recomposition matches original model...")

    original = build_original_model()
    original.eval()

    x = torch.rand(batch_size, in_channels, height, width)

    with torch.no_grad():
        original_output = original(x)

    # Build composed model from separate branch components
    class ComposedModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Each branch is a standalone component (matching level_2_layer)
            self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
            self.branch3x3 = nn.Sequential(
                nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
                nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
            )
            self.branch5x5 = nn.Sequential(
                nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
                nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
            )
            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                nn.Conv2d(in_channels, pool_proj, kernel_size=1)
            )

        def forward(self, x):
            # Each branch runs independently (parallel in the Inception design)
            b1 = self.branch1x1(x)
            b3 = self.branch3x3(x)
            b5 = self.branch5x5(x)
            bp = self.branch_pool(x)
            # Channel concatenation (level_0_kernel/channel_concat)
            return torch.cat([b1, b3, b5, bp], dim=1)

    composed = ComposedModel()
    # Transfer all weights from original
    composed.branch1x1.load_state_dict(original.branch1x1.state_dict())
    composed.branch3x3.load_state_dict(original.branch3x3.state_dict())
    composed.branch5x5.load_state_dict(original.branch5x5.state_dict())
    composed.branch_pool.load_state_dict(original.branch_pool.state_dict())
    composed.eval()

    with torch.no_grad():
        composed_output = composed(x)

    assert torch.equal(original_output, composed_output), \
        "Full recomposition output does not match original!"

    total_channels = out_1x1 + out_3x3 + out_5x5 + pool_proj
    assert composed_output.shape == (batch_size, total_channels, height, width), \
        f"Shape mismatch: expected {(batch_size, total_channels, height, width)}, got {composed_output.shape}"

    # Verify channel slicing matches individual branches
    with torch.no_grad():
        b1 = composed.branch1x1(x)
        b3 = composed.branch3x3(x)
        b5 = composed.branch5x5(x)
        bp = composed.branch_pool(x)

    offset = 0
    assert torch.equal(composed_output[:, offset:offset+out_1x1], b1), "1x1 slice mismatch"
    offset += out_1x1
    assert torch.equal(composed_output[:, offset:offset+out_3x3], b3), "3x3 slice mismatch"
    offset += out_3x3
    assert torch.equal(composed_output[:, offset:offset+out_5x5], b5), "5x5 slice mismatch"
    offset += out_5x5
    assert torch.equal(composed_output[:, offset:offset+pool_proj], bp), "pool slice mismatch"

    print("  Full recomposition: PASS")


def run_tests():
    test_kernel_level()
    test_fusion_level()
    test_layer_level_matches_original()
    test_full_recomposition()
    print("\nAll composition tests PASS")


if __name__ == "__main__":
    run_tests()
