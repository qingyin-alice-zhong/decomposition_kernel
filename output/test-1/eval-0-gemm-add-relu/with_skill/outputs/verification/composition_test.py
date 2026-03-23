"""
Composition Verification: Gemm_Add_ReLU
Verifies that composing all L0 kernel components reproduces the original model output.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# =============================================================================
# STEP 1: Define the ORIGINAL model
# =============================================================================

class OriginalModel(nn.Module):
    """Exact copy of the original 76_Gemm_Add_ReLU model."""
    def __init__(self, in_features, out_features, bias_shape):
        super(OriginalModel, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.gemm(x)
        x = x + self.bias
        x = torch.relu(x)
        return x

# =============================================================================
# STEP 2: Import all decomposed kernel components
# =============================================================================

kernel_dir = str(Path(__file__).parent.parent / "level_0_kernel")
sys.path.insert(0, kernel_dir)

from linear_8192x8192_nobias_fp32 import Model as LinearNoBias
from bias_add_8192_fp32 import Model as BiasAdd
from relu_fp32 import Model as ReLU

# =============================================================================
# STEP 3: Build the composed model from kernels
# =============================================================================

class ComposedModel(nn.Module):
    """Model composed from L0 kernel components."""
    def __init__(self, in_features, out_features, bias_shape):
        super().__init__()
        self.gemm = LinearNoBias(in_features, out_features)
        self.bias_add = BiasAdd(bias_shape)
        self.relu = ReLU()

    def forward(self, x):
        x = self.gemm(x)
        x = self.bias_add(x)
        x = self.relu(x)
        return x

# =============================================================================
# STEP 4: Define test inputs
# =============================================================================

batch_size = 1024
in_features = 8192
out_features = 8192
bias_shape = (out_features,)

def get_test_inputs():
    return [torch.rand(batch_size, in_features)]

# =============================================================================
# STEP 5: Verification function
# =============================================================================

def verify_composition(rtol=1e-4, atol=1e-5):
    print("=" * 60)
    print("COMPOSITION VERIFICATION")
    print("=" * 60)

    # Create models
    original = OriginalModel(in_features, out_features, bias_shape)
    composed = ComposedModel(in_features, out_features, bias_shape)

    # Transfer weights: map original state_dict to composed
    orig_sd = original.state_dict()
    comp_sd = composed.state_dict()

    # Explicit weight mapping
    weight_map = {
        "gemm.weight": "gemm.linear.weight",
        "bias": "bias_add.bias",
    }

    for orig_key, comp_key in weight_map.items():
        comp_sd[comp_key] = orig_sd[orig_key].clone()

    composed.load_state_dict(comp_sd)
    print("[OK] Weights transferred")

    original.eval()
    composed.eval()

    # Run 3 trials
    all_pass = True
    max_diff_all = 0.0

    for trial in range(3):
        torch.manual_seed(42 + trial)
        test_inputs = get_test_inputs()

        with torch.no_grad():
            original_output = original(*test_inputs)
            composed_output = composed(*test_inputs)

        shape_match = original_output.shape == composed_output.shape
        max_diff = (original_output - composed_output).abs().max().item()
        max_diff_all = max(max_diff_all, max_diff)
        value_match = torch.allclose(original_output, composed_output, rtol=rtol, atol=atol)

        status = "PASS" if (shape_match and value_match) else "FAIL"
        print(f"Trial {trial}: {status} (max_diff={max_diff:.2e})")

        if not (shape_match and value_match):
            all_pass = False

    print()
    print("-" * 60)
    print(f"Max difference across trials: {max_diff_all:.2e}")
    print("-" * 60)

    if all_pass:
        print()
        print("[PASS] Composition verification PASSED!")
        print("       Decomposed components correctly reproduce original.")
        return True
    else:
        print()
        print("[FAIL] Composition verification FAILED!")
        return False


if __name__ == "__main__":
    success = verify_composition()
    sys.exit(0 if success else 1)
