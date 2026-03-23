"""
End-to-End Composition Verification
Builds full inception module from only L0 kernel components and verifies
numerical equivalence with the original model.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# =============================================================================
# STEP 1: Original model
# =============================================================================

class OriginalModel(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super(OriginalModel, self).__init__()
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
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)


# =============================================================================
# STEP 2: Import all L0 kernel components
# =============================================================================

kernel_dir = str(Path(__file__).parent.parent / "level_0_kernel")
sys.path.insert(0, kernel_dir)

from conv2d_480x192_1x1_fp32 import Model as Conv1x1_192
from conv2d_480x96_1x1_fp32 import Model as Conv1x1_96
from conv2d_96x208_3x3_fp32 import Model as Conv3x3_208
from conv2d_480x16_1x1_fp32 import Model as Conv1x1_16
from conv2d_16x48_5x5_fp32 import Model as Conv5x5_48
from maxpool2d_3x3_fp32 import Model as MaxPool3x3
from conv2d_480x64_1x1_fp32 import Model as Conv1x1_64


# =============================================================================
# STEP 3: Composed model from kernels
# =============================================================================

class ComposedModel(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        super().__init__()
        # Branch 1x1
        self.branch1x1 = Conv1x1_192(in_channels, out_1x1)
        # Branch 3x3
        self.branch3x3_reduce = Conv1x1_96(in_channels, reduce_3x3)
        self.branch3x3_conv = Conv3x3_208(reduce_3x3, out_3x3)
        # Branch 5x5
        self.branch5x5_reduce = Conv1x1_16(in_channels, reduce_5x5)
        self.branch5x5_conv = Conv5x5_48(reduce_5x5, out_5x5)
        # Branch pool
        self.branch_pool_pool = MaxPool3x3()
        self.branch_pool_proj = Conv1x1_64(in_channels, pool_proj)

    def forward(self, x):
        # Branch 1x1
        b1 = self.branch1x1(x)
        # Branch 3x3
        b3 = self.branch3x3_reduce(x)
        b3 = self.branch3x3_conv(b3)
        # Branch 5x5
        b5 = self.branch5x5_reduce(x)
        b5 = self.branch5x5_conv(b5)
        # Branch pool
        bp = self.branch_pool_pool(x)
        bp = self.branch_pool_proj(bp)
        return torch.cat([b1, b3, b5, bp], 1)


# =============================================================================
# STEP 4: Test inputs
# =============================================================================

in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_test_inputs():
    return [torch.rand(batch_size, in_channels, height, width)]

def get_init_args():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]


# =============================================================================
# STEP 5: Verification
# =============================================================================

def build_weight_map():
    """Map original state_dict keys to composed model keys."""
    return {
        "branch1x1.weight": "branch1x1.conv.weight",
        "branch1x1.bias": "branch1x1.conv.bias",
        "branch3x3.0.weight": "branch3x3_reduce.conv.weight",
        "branch3x3.0.bias": "branch3x3_reduce.conv.bias",
        "branch3x3.1.weight": "branch3x3_conv.conv.weight",
        "branch3x3.1.bias": "branch3x3_conv.conv.bias",
        "branch5x5.0.weight": "branch5x5_reduce.conv.weight",
        "branch5x5.0.bias": "branch5x5_reduce.conv.bias",
        "branch5x5.1.weight": "branch5x5_conv.conv.weight",
        "branch5x5.1.bias": "branch5x5_conv.conv.bias",
        "branch_pool.1.weight": "branch_pool_proj.conv.weight",
        "branch_pool.1.bias": "branch_pool_proj.conv.bias",
    }


def verify_composition(rtol=1e-4, atol=1e-5):
    print("=" * 60)
    print("COMPOSITION VERIFICATION")
    print("=" * 60)

    init_args = get_init_args()
    original = OriginalModel(*init_args)
    composed = ComposedModel(*init_args)

    # Transfer weights using explicit map
    weight_map = build_weight_map()
    orig_sd = original.state_dict()
    comp_sd = composed.state_dict()

    mapped = 0
    for orig_key, comp_key in weight_map.items():
        if orig_key in orig_sd and comp_key in comp_sd:
            comp_sd[comp_key] = orig_sd[orig_key].clone()
            mapped += 1

    composed.load_state_dict(comp_sd)
    print(f"[OK] Weights transferred: {mapped}/{len(weight_map)}")

    original.eval()
    composed.eval()

    test_inputs = get_test_inputs()
    print(f"[OK] Test inputs: {[x.shape for x in test_inputs]}")

    num_trials = 3
    all_pass = True
    max_diff_all = 0.0

    for trial in range(num_trials):
        torch.manual_seed(42 + trial)
        test_inputs = get_test_inputs()

        with torch.no_grad():
            original_output = original(*test_inputs)
            composed_output = composed(*test_inputs)

        shape_match = original_output.shape == composed_output.shape
        diff = (original_output - composed_output).abs().max().item()
        max_diff_all = max(max_diff_all, diff)
        value_match = torch.allclose(original_output, composed_output, rtol=rtol, atol=atol)

        status = "PASS" if (shape_match and value_match) else "FAIL"
        print(f"Trial {trial}: shape_match={shape_match}, max_diff={diff:.2e}, {status}")

        if not (shape_match and value_match):
            all_pass = False

    print()
    print("-" * 60)
    print(f"Max difference: {max_diff_all:.2e}")
    if all_pass:
        print("[PASS] Composition verification PASSED!")
    else:
        print("[FAIL] Composition verification FAILED!")
    print("-" * 60)

    return all_pass


if __name__ == "__main__":
    success = verify_composition()
    sys.exit(0 if success else 1)
