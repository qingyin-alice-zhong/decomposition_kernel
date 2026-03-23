"""
Composition test: Proves that composing the kernel-level components
(linear_no_bias -> bias_add -> relu) produces identical output to
the original monolithic model.

This test:
1. Creates the original model
2. Creates the three kernel-level components
3. Copies weights from the original model to the components
4. Runs the same input through both paths
5. Asserts numerical equivalence
"""
import sys
import os
import torch
import torch.nn as nn

# Add parent directory to path so we can import components
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Import kernel-level components
sys.path.insert(0, os.path.join(BASE_DIR, "level_0_kernel"))
import linear_no_bias
import bias_add
import relu as relu_mod

# Import the model-level component
sys.path.insert(0, os.path.join(BASE_DIR, "level_2_model"))
import gemm_add_relu_model


def test_composition_equivalence():
    """Test that composing kernel components equals the original model."""
    torch.manual_seed(42)

    # Dimensions
    in_features = 512  # Use smaller dims for faster testing
    out_features = 256
    batch_size = 64
    bias_shape = (out_features,)

    # Create original model
    original = gemm_add_relu_model.Model(in_features, out_features, bias_shape)

    # Create kernel-level components
    k_linear = linear_no_bias.Model(in_features, out_features)
    k_bias = bias_add.Model(bias_shape)
    k_relu = relu_mod.Model()

    # Copy weights from original to components
    k_linear.gemm.weight.data.copy_(original.gemm.weight.data)
    k_bias.bias.data.copy_(original.bias.data)

    # Create a fixed input
    x = torch.rand(batch_size, in_features)

    # Run through original model
    original.eval()
    with torch.no_grad():
        y_original = original(x)

    # Run through composed kernel components
    k_linear.eval()
    k_bias.eval()
    k_relu.eval()
    with torch.no_grad():
        y_step1 = k_linear(x)
        y_step2 = k_bias(y_step1)
        y_composed = k_relu(y_step2)

    # Verify exact numerical equivalence
    assert torch.allclose(y_original, y_composed, atol=1e-6), (
        f"Composition mismatch! Max diff: {(y_original - y_composed).abs().max().item()}"
    )
    print("[PASS] Kernel composition matches original model")


def test_fusion_equivalence():
    """Test that the fusion-level component equals the original model."""
    torch.manual_seed(42)

    in_features = 512
    out_features = 256
    batch_size = 64
    bias_shape = (out_features,)

    # Import fusion-level component
    sys.path.insert(0, os.path.join(BASE_DIR, "level_1_fusion"))
    import gemm_bias_relu

    original = gemm_add_relu_model.Model(in_features, out_features, bias_shape)
    fused = gemm_bias_relu.Model(in_features, out_features, bias_shape)

    # Copy weights
    fused.gemm.weight.data.copy_(original.gemm.weight.data)
    fused.bias.data.copy_(original.bias.data)

    x = torch.rand(batch_size, in_features)

    original.eval()
    fused.eval()
    with torch.no_grad():
        y_original = original(x)
        y_fused = fused(x)

    assert torch.allclose(y_original, y_fused, atol=1e-6), (
        f"Fusion mismatch! Max diff: {(y_original - y_fused).abs().max().item()}"
    )
    print("[PASS] Fusion-level component matches original model")


def test_individual_kernels():
    """Test that each kernel component works correctly in isolation."""
    torch.manual_seed(42)

    # Test linear_no_bias
    lin = linear_no_bias.Model(64, 32)
    x = torch.rand(8, 64)
    y = lin(x)
    assert y.shape == (8, 32), f"Linear shape mismatch: {y.shape}"
    # Verify it's a matmul: y = x @ W^T
    expected = x @ lin.gemm.weight.data.T
    assert torch.allclose(y, expected, atol=1e-5), "Linear output incorrect"
    print("[PASS] linear_no_bias kernel correct")

    # Test bias_add
    ba = bias_add.Model((32,))
    x = torch.rand(8, 32)
    y = ba(x)
    expected = x + ba.bias
    assert torch.allclose(y, expected, atol=1e-6), "Bias add output incorrect"
    print("[PASS] bias_add kernel correct")

    # Test relu
    r = relu_mod.Model()
    x = torch.randn(8, 32)
    y = r(x)
    expected = torch.relu(x)
    assert torch.allclose(y, expected), "ReLU output incorrect"
    print("[PASS] relu kernel correct")


def test_shapes_and_dtypes():
    """Verify shapes and dtypes propagate correctly through composition."""
    torch.manual_seed(42)

    in_features = 128
    out_features = 64
    batch_size = 16

    k_linear = linear_no_bias.Model(in_features, out_features)
    k_bias = bias_add.Model((out_features,))
    k_relu = relu_mod.Model()

    x = torch.rand(batch_size, in_features)

    y1 = k_linear(x)
    assert y1.shape == (batch_size, out_features), f"After linear: {y1.shape}"
    assert y1.dtype == torch.float32

    y2 = k_bias(y1)
    assert y2.shape == (batch_size, out_features), f"After bias: {y2.shape}"
    assert y2.dtype == torch.float32

    y3 = k_relu(y2)
    assert y3.shape == (batch_size, out_features), f"After relu: {y3.shape}"
    assert y3.dtype == torch.float32
    assert (y3 >= 0).all()

    print("[PASS] Shapes and dtypes correct through composition")


if __name__ == "__main__":
    print("=" * 60)
    print("Composition Verification Tests")
    print("=" * 60)
    test_individual_kernels()
    test_composition_equivalence()
    test_fusion_equivalence()
    test_shapes_and_dtypes()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
