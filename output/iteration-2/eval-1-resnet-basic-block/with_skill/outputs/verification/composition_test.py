"""
Composition Verification: ResNet BasicBlock
End-to-end test: build model from L0 kernels and compare to original.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# ---- Import original model ----
# Resolve workspace root: go up from verification/ to outputs/, then find workspace root
_outputs_dir = Path(__file__).resolve().parent.parent
_workspace_root = _outputs_dir
while _workspace_root.name != "decomposition_workspace" and _workspace_root != _workspace_root.parent:
    _workspace_root = _workspace_root.parent

_original_file = _workspace_root / "data" / "kernelbench" / "level3" / "8_ResNetBasicBlock.py"

import importlib.util
orig_spec = importlib.util.spec_from_file_location(
    "original_resnet",
    str(_original_file)
)
orig_module = importlib.util.module_from_spec(orig_spec)
orig_spec.loader.exec_module(orig_module)
OriginalModel = orig_module.Model

# ---- Import L0 kernel components ----
kernel_dir = str(Path(__file__).parent.parent / "level_0_kernel")
sys.path.insert(0, kernel_dir)

from conv2d_3x64_3x3_fp32 import Model as Conv2d_3x64_3x3
from conv2d_64x64_3x3_fp32 import Model as Conv2d_64x64_3x3
from conv2d_3x64_1x1_fp32 import Model as Conv2d_3x64_1x1
from batchnorm2d_64_fp32 import Model as BatchNorm2d_64
from relu_fp32 import Model as ReLU


# ---- Build composed model from L0 kernels ----
class ComposedModel(nn.Module):
    """
    ResNet BasicBlock composed entirely from L0 kernel components.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Main path: conv1 -> bn1 -> relu -> conv2 -> bn2
        self.conv1 = Conv2d_3x64_3x3(in_channels, out_channels, 3, stride, 1)
        self.bn1 = BatchNorm2d_64(out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2d_64x64_3x3(out_channels, out_channels, 3, 1, 1)
        self.bn2 = BatchNorm2d_64(out_channels)

        # Downsample path: conv_ds -> bn_ds
        self.conv_ds = Conv2d_3x64_1x1(in_channels, out_channels, 1, stride)
        self.bn_ds = BatchNorm2d_64(out_channels)

        # Final relu after residual add
        self.final_relu = ReLU()

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
        out += identity
        out = self.final_relu(out)

        return out


def verify_composition(rtol=1e-4, atol=1e-5):
    print("=" * 60)
    print("COMPOSITION VERIFICATION: ResNet BasicBlock")
    print("=" * 60)

    in_channels = 3
    out_channels = 64
    stride = 1

    original = OriginalModel(in_channels, out_channels, stride)
    composed = ComposedModel(in_channels, out_channels, stride)

    # Weight transfer: map original state_dict to composed
    orig_sd = original.state_dict()
    comp_sd = composed.state_dict()

    weight_map = {
        "conv1.weight": "conv1.conv.weight",
        "bn1.weight": "bn1.bn.weight",
        "bn1.bias": "bn1.bn.bias",
        "bn1.running_mean": "bn1.bn.running_mean",
        "bn1.running_var": "bn1.bn.running_var",
        "bn1.num_batches_tracked": "bn1.bn.num_batches_tracked",
        "conv2.weight": "conv2.conv.weight",
        "bn2.weight": "bn2.bn.weight",
        "bn2.bias": "bn2.bn.bias",
        "bn2.running_mean": "bn2.bn.running_mean",
        "bn2.running_var": "bn2.bn.running_var",
        "bn2.num_batches_tracked": "bn2.bn.num_batches_tracked",
        "downsample.0.weight": "conv_ds.conv.weight",
        "downsample.1.weight": "bn_ds.bn.weight",
        "downsample.1.bias": "bn_ds.bn.bias",
        "downsample.1.running_mean": "bn_ds.bn.running_mean",
        "downsample.1.running_var": "bn_ds.bn.running_var",
        "downsample.1.num_batches_tracked": "bn_ds.bn.num_batches_tracked",
    }

    for orig_key, comp_key in weight_map.items():
        if orig_key in orig_sd and comp_key in comp_sd:
            comp_sd[comp_key] = orig_sd[orig_key].clone()

    composed.load_state_dict(comp_sd)
    print("[OK] Weights transferred")

    original.eval()
    composed.eval()

    # Run comparison over 3 trials
    all_pass = True
    max_diff_all = 0.0

    for trial in range(3):
        torch.manual_seed(42 + trial)
        test_input = torch.rand(10, 3, 224, 224)

        with torch.no_grad():
            orig_out = original(test_input)
            comp_out = composed(test_input)

        diff = (orig_out.float() - comp_out.float()).abs().max().item()
        max_diff_all = max(max_diff_all, diff)
        matches = torch.allclose(orig_out, comp_out, rtol=rtol, atol=atol)

        if not matches:
            all_pass = False

        print(f"Trial {trial}: max_diff={diff:.2e} {'PASS' if matches else 'FAIL'}")

    print()
    print("-" * 60)
    print(f"Shape match:    {orig_out.shape == comp_out.shape}")
    print(f"Max difference: {max_diff_all:.2e}")
    print("-" * 60)

    if all_pass:
        print()
        print("[PASS] Composition verification PASSED!")
        return True
    else:
        print()
        print("[FAIL] Composition verification FAILED!")
        return False


if __name__ == "__main__":
    success = verify_composition()
    sys.exit(0 if success else 1)
