"""
End-to-End Composition Test: GPT-OSS Transformer
Builds a full model from ONLY L0 kernel components and verifies
it matches the original model output.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add kernel directory
KERNEL_DIR = str(Path(__file__).parent.parent / "level_0_kernel")
sys.path.insert(0, KERNEL_DIR)

# Import all L0 kernels
from embedding_201088x2880_bf16 import Model as EmbeddingKernel
from rms_norm_2880_fp32 import Model as RMSNormKernel
from linear_2880x201088_bf16 import Model as UnembeddingKernel
from linear_2880x5120_bf16 import Model as QKVLinearKernel
from rotary_embedding_64_fp32 import Model as RoPEKernel
from sdpa_gqa_64_bf16 import Model as SDPAKernel
from linear_4096x2880_bf16 import Model as OutLinearKernel
from linear_2880x4_bf16 import Model as GateLinearKernel
from moe_expert_routing_bf16 import Model as ExpertRoutingKernel
from moe_mlp1_4x5760x2880_bf16 import Model as MoEMLP1Kernel
from swiglu_bf16 import Model as SwiGLUKernel
from moe_mlp2_4x2880x2880_bf16 import Model as MoEMLP2Kernel
from moe_expert_combine_bf16 import Model as ExpertCombineKernel

# Import original model
MODEL_DIR = str(Path(__file__).parent.parent / "level_3_model")
sys.path.insert(0, MODEL_DIR)
from gpt_oss import Model as OriginalModel, get_inputs


class ComposedAttentionBlock(nn.Module):
    def __init__(self, layer_idx=0):
        super().__init__()
        self.num_attention_heads = 64
        self.num_key_value_heads = 8
        self.head_dim = 64
        self.norm = RMSNormKernel(2880)
        self.qkv = QKVLinearKernel(2880, 5120)
        self.rope = RoPEKernel(64, 150000.0, 4096, 32.0, 1.0, 32.0)
        sliding_window = 128 if layer_idx % 2 == 0 else 0
        self.sdpa = SDPAKernel(64, 8, 64, sliding_window)
        self.out = OutLinearKernel(4096, 2880)

    def forward(self, x):
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[:, self.num_attention_heads * self.head_dim : (self.num_attention_heads + self.num_key_value_heads) * self.head_dim].contiguous()
        v = qkv[:, (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim].contiguous()
        q = q.view(-1, self.num_key_value_heads, self.num_attention_heads // self.num_key_value_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        t = self.sdpa(q, k, v)
        t = self.out(t)
        t = x + t
        return t


class ComposedMLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNormKernel(2880)
        self.gate = GateLinearKernel(2880, 4)
        self.routing = ExpertRoutingKernel(2)
        self.mlp1 = MoEMLP1Kernel(4, 2880, 2880)
        self.swiglu = SwiGLUKernel(7.0, 1.702)
        self.mlp2 = MoEMLP2Kernel(4, 2880, 2880)
        self.combine = ExpertCombineKernel()

    def forward(self, x):
        t = self.norm(x)
        g = self.gate(t)
        expert_weights, expert_indices = self.routing(g)
        t = self.mlp1(t, expert_indices)
        t = self.swiglu(t)
        t = self.mlp2(t, expert_indices)
        t = self.combine(t, expert_weights)
        return x + t


class ComposedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingKernel(201088, 2880)
        self.attn_0 = ComposedAttentionBlock(0)
        self.mlp_0 = ComposedMLPBlock()
        self.attn_1 = ComposedAttentionBlock(1)
        self.mlp_1 = ComposedMLPBlock()
        self.norm = RMSNormKernel(2880)
        self.unembedding = UnembeddingKernel(2880, 201088)

    def forward(self, x):
        x = self.embedding(x)
        # Block 0
        x = self.attn_0(x)
        x = self.mlp_0(x)
        # Block 1
        x = self.attn_1(x)
        x = self.mlp_1(x)
        # Final
        x = self.norm(x)
        x = self.unembedding(x)
        return x


def transfer_weights(original, composed):
    """Transfer weights from original to composed model with explicit mapping."""
    orig_sd = original.state_dict()
    comp_sd = composed.state_dict()

    # Build explicit weight map
    weight_map = {
        # Embedding
        "model.embedding.weight": "embedding.embedding.weight",
        # Block 0 - Attention
        "model.block.0.attn.norm.scale": "attn_0.norm.scale",
        "model.block.0.attn.qkv.weight": "attn_0.qkv.linear.weight",
        "model.block.0.attn.qkv.bias": "attn_0.qkv.linear.bias",
        "model.block.0.attn.sinks": "attn_0.sdpa.sinks",
        "model.block.0.attn.out.weight": "attn_0.out.linear.weight",
        "model.block.0.attn.out.bias": "attn_0.out.linear.bias",
        # Block 0 - MLP
        "model.block.0.mlp.norm.scale": "mlp_0.norm.scale",
        "model.block.0.mlp.gate.weight": "mlp_0.gate.linear.weight",
        "model.block.0.mlp.gate.bias": "mlp_0.gate.linear.bias",
        "model.block.0.mlp.mlp1_weight": "mlp_0.mlp1.mlp1_weight",
        "model.block.0.mlp.mlp1_bias": "mlp_0.mlp1.mlp1_bias",
        "model.block.0.mlp.mlp2_weight": "mlp_0.mlp2.mlp2_weight",
        "model.block.0.mlp.mlp2_bias": "mlp_0.mlp2.mlp2_bias",
        # Block 1 - Attention
        "model.block.1.attn.norm.scale": "attn_1.norm.scale",
        "model.block.1.attn.qkv.weight": "attn_1.qkv.linear.weight",
        "model.block.1.attn.qkv.bias": "attn_1.qkv.linear.bias",
        "model.block.1.attn.sinks": "attn_1.sdpa.sinks",
        "model.block.1.attn.out.weight": "attn_1.out.linear.weight",
        "model.block.1.attn.out.bias": "attn_1.out.linear.bias",
        # Block 1 - MLP
        "model.block.1.mlp.norm.scale": "mlp_1.norm.scale",
        "model.block.1.mlp.gate.weight": "mlp_1.gate.linear.weight",
        "model.block.1.mlp.gate.bias": "mlp_1.gate.linear.bias",
        "model.block.1.mlp.mlp1_weight": "mlp_1.mlp1.mlp1_weight",
        "model.block.1.mlp.mlp1_bias": "mlp_1.mlp1.mlp1_bias",
        "model.block.1.mlp.mlp2_weight": "mlp_1.mlp2.mlp2_weight",
        "model.block.1.mlp.mlp2_bias": "mlp_1.mlp2.mlp2_bias",
        # Final norm + unembedding
        "model.norm.scale": "norm.scale",
        "model.unembedding.weight": "unembedding.linear.weight",
    }

    mapped = 0
    for orig_key, comp_key in weight_map.items():
        if orig_key in orig_sd and comp_key in comp_sd:
            comp_sd[comp_key] = orig_sd[orig_key].clone()
            mapped += 1
        else:
            if orig_key not in orig_sd:
                print(f"  Missing in original: {orig_key}")
            if comp_key not in comp_sd:
                print(f"  Missing in composed: {comp_key}")

    composed.load_state_dict(comp_sd)
    print(f"  Mapped {mapped}/{len(weight_map)} parameters")
    return mapped


def main():
    print("=" * 60)
    print("END-TO-END COMPOSITION TEST: GPT-OSS")
    print("=" * 60)

    # Load original
    print("\n[1/4] Loading original model...")
    original = OriginalModel()
    # Initialize all parameters with small random values (torch.empty leaves NaN/garbage)
    torch.manual_seed(123)
    for p in original.parameters():
        if p.dtype == torch.bfloat16:
            p.data = torch.randn_like(p) * 0.02
        elif p.dtype == torch.float32:
            p.data = torch.randn_like(p) * 0.1
    original.eval()
    print(f"  Original params: {sum(p.numel() for p in original.parameters())}")

    # Load composed
    print("\n[2/4] Building composed model from L0 kernels...")
    composed = ComposedModel()
    composed.eval()
    print(f"  Composed params: {sum(p.numel() for p in composed.parameters())}")

    # Transfer weights
    print("\n[3/4] Transferring weights...")
    mapped = transfer_weights(original, composed)

    # Compare
    print("\n[4/4] Numerical comparison (3 trials)...")
    all_pass = True
    max_diff_all = 0.0

    for trial in range(3):
        torch.manual_seed(42 + trial)
        inputs = get_inputs()

        with torch.no_grad():
            orig_out = original(*inputs)
            comp_out = composed(*inputs)

        diff = (orig_out.float() - comp_out.float()).abs().max().item()
        max_diff_all = max(max_diff_all, diff)

        # bf16 tolerance
        matches = torch.allclose(orig_out.float(), comp_out.float(), rtol=1e-3, atol=1e-3)
        if not matches:
            all_pass = False

        status = "PASS" if matches else "FAIL"
        print(f"  Trial {trial}: max_diff={diff:.2e} {status}")

    print()
    print("-" * 60)
    if all_pass:
        print(f"PASS (max_diff={max_diff_all:.2e})")
    else:
        print(f"FAIL (max_diff={max_diff_all:.2e})")
    print("-" * 60)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
