"""
Composition Verification: ResNet BasicBlock
Builds the full model from L0 kernel components and verifies output matches original.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# =============================================================================
# STEP 1: Define the ORIGINAL model
# =============================================================================

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


# =============================================================================
# STEP 2: Import all decomposed kernel components
# =============================================================================

kernel_dir = str(Path(__file__).parent.parent / "level_0_kernel")
sys.path.insert(0, kernel_dir)

from conv2d_3x64_3x3_fp32 import Model as Conv2d_3x64_3x3
from conv2d_64x64_3x3_fp32 import Model as Conv2d_64x64_3x3
from conv2d_3x64_1x1_fp32 import Model as Conv2d_3x64_1x1
from batchnorm2d_64_fp32 import Model as BatchNorm2d_64
from relu_fp32 import Model as ReLU_K


# =============================================================================
# STEP 3: Build the composed model from kernels
# =============================================================================

class ComposedModel(nn.Module):
    """
    ResNet BasicBlock composed entirely from L0 kernel components.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Main path: conv1 -> bn1 -> relu -> conv2 -> bn2
        self.conv1 = Conv2d_3x64_3x3(in_channels, out_channels, 3, stride, 1)
        self.bn1 = BatchNorm2d_64(out_channels)
        self.relu1 = ReLU_K()
        self.conv2 = Conv2d_64x64_3x3(out_channels, out_channels, 3, 1, 1)
        self.bn2 = BatchNorm2d_64(out_channels)
        # Downsample path: conv_ds -> bn_ds
        self.conv_ds = Conv2d_3x64_1x1(in_channels, out_channels, 1, stride)
        self.bn_ds = BatchNorm2d_64(out_channels)
        # Final relu after residual add
        self.relu_final = ReLU_K()

    def forward(self, x):
        identity = x

        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Downsample path
        identity = self.conv_ds(identity)
        identity = self.bn_ds(identity)

        # Residual add + final relu
        out = out + identity
        out = self.relu_final(out)

        return out


# =============================================================================
# STEP 4: Define test inputs
# =============================================================================

in_channels = 3
out_channels = 64
stride = 1
batch_size = 10

def get_test_inputs():
    return [torch.rand(batch_size, in_channels, 224, 224)]


# =============================================================================
# STEP 5: Verification function
# =============================================================================

def verify_composition(rtol=1e-4, atol=1e-5):
    print("=" * 60)
    print("COMPOSITION VERIFICATION")
    print("=" * 60)

    original = OriginalModel(in_channels, out_channels, stride)
    composed = ComposedModel(in_channels, out_channels, stride)

    # Build weight map: original state_dict keys -> composed state_dict keys
    weight_map = {
        "conv1.weight":        "conv1.conv.weight",
        "bn1.weight":          "bn1.bn.weight",
        "bn1.bias":            "bn1.bn.bias",
        "bn1.running_mean":    "bn1.bn.running_mean",
        "bn1.running_var":     "bn1.bn.running_var",
        "bn1.num_batches_tracked": "bn1.bn.num_batches_tracked",
        "conv2.weight":        "conv2.conv.weight",
        "bn2.weight":          "bn2.bn.weight",
        "bn2.bias":            "bn2.bn.bias",
        "bn2.running_mean":    "bn2.bn.running_mean",
        "bn2.running_var":     "bn2.bn.running_var",
        "bn2.num_batches_tracked": "bn2.bn.num_batches_tracked",
        "downsample.0.weight": "conv_ds.conv.weight",
        "downsample.1.weight": "bn_ds.bn.weight",
        "downsample.1.bias":   "bn_ds.bn.bias",
        "downsample.1.running_mean": "bn_ds.bn.running_mean",
        "downsample.1.running_var":  "bn_ds.bn.running_var",
        "downsample.1.num_batches_tracked": "bn_ds.bn.num_batches_tracked",
    }

    orig_sd = original.state_dict()
    comp_sd = composed.state_dict()

    for orig_key, comp_key in weight_map.items():
        if orig_key in orig_sd and comp_key in comp_sd:
            comp_sd[comp_key] = orig_sd[orig_key].clone()

    composed.load_state_dict(comp_sd)
    print("[OK] Weights transferred via explicit mapping")

    original.eval()
    composed.eval()

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
        diff = (original_output.float() - composed_output.float()).abs().max().item()
        max_diff_all = max(max_diff_all, diff)
        value_match = torch.allclose(original_output.float(), composed_output.float(),
                                     rtol=rtol, atol=atol)

        status = "PASS" if (shape_match and value_match) else "FAIL"
        if not (shape_match and value_match):
            all_pass = False
        print(f"Trial {trial} (seed={42+trial}): {status} "
              f"(max_diff={diff:.2e}, shape_match={shape_match})")

    print()
    print("-" * 60)
    if all_pass:
        print(f"[PASS] Composition verification PASSED! (max_diff={max_diff_all:.2e})")
    else:
        print(f"[FAIL] Composition verification FAILED! (max_diff={max_diff_all:.2e})")
    print("-" * 60)

    return all_pass


if __name__ == "__main__":
    success = verify_composition()
    sys.exit(0 if success else 1)
