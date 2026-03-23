"""
Step 3b Refactored: MLPBlock (MoE) -> Kernels
Decomposes MLPBlock into:
  - RMSNorm (L0)
  - Gate Linear (L0)
  - MoE Expert Routing (L0) - topk + softmax
  - MoE MLP1 (L0) - expert einsum + bias
  - SwiGLU (L0)
  - MoE MLP2 (L0) - expert einsum + bias
  - MoE Expert Combine (L0) - weighted sum
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "children"))

from rms_norm_2880_fp32 import Model as RMSNorm
from linear_2880x4_bf16 import Model as GateLinear
from moe_expert_routing_bf16 import Model as ExpertRouting
from moe_mlp1_4x5760x2880_bf16 import Model as MoEMLP1
from swiglu_bf16 import Model as SwiGLU
from moe_mlp2_4x2880x2880_bf16 import Model as MoEMLP2
from moe_expert_combine_bf16 import Model as ExpertCombine


class RefactoredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(2880)
        self.gate = GateLinear(2880, 4)
        self.routing = ExpertRouting(2)
        self.mlp1 = MoEMLP1(4, 2880, 2880)
        self.swiglu = SwiGLU(7.0, 1.702)
        self.mlp2 = MoEMLP2(4, 2880, 2880)
        self.combine = ExpertCombine()

    def forward(self, x):
        t = self.norm(x)
        g = self.gate(t)
        expert_weights, expert_indices = self.routing(g)
        t = self.mlp1(t, expert_indices)
        t = self.swiglu(t)
        t = self.mlp2(t, expert_indices)
        t = self.combine(t, expert_weights)
        return x + t


def get_inputs():
    return [torch.randn(32, 2880, dtype=torch.bfloat16)]


def get_init_inputs():
    return []


if __name__ == "__main__":
    model = RefactoredModel()
    model.eval()
    with torch.no_grad():
        inputs = get_inputs()
        output = model(*inputs)
        print(f"Output shape: {output.shape}")
        print("PASS" if output.shape == (32, 2880) else "FAIL")
