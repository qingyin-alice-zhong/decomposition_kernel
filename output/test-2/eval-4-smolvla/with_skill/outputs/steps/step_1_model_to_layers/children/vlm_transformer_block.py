"""
Component: VLM Transformer Block (VLlama3)
Abstraction Level: layer
Parent: SmolVLA
Children: [rms_norm, qkv_projection, rope, attention, o_projection, mlp]

Operations: RMSNorm, Q/K/V linear projections, RoPE, scaled dot-product attention,
            output projection, residual add, RMSNorm, SwiGLU MLP, residual add

Input Shapes:
  - hidden_states: (1, L, 960) dtype=float32

Output Shapes:
  - output: (1, L, 960) dtype=float32

Weight Shapes:
  - input_layernorm.weight: (960,)
  - q_proj.weight: (960, 960)
  - k_proj.weight: (320, 960)
  - v_proj.weight: (320, 960)
  - o_proj.weight: (960, 960)
  - post_attention_layernorm.weight: (960,)
  - mlp.gate_proj.weight: (2560, 960)
  - mlp.up_proj.weight: (2560, 960)
  - mlp.down_proj.weight: (960, 2560)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_HIDDEN_SIZE = 960
TEXT_NUM_HEADS = 15
TEXT_NUM_KV_HEADS = 5
TEXT_HEAD_DIM = 64
TEXT_INTERMEDIATE_SIZE = 2560
TEXT_RMS_NORM_EPS = 1e-5
SEQ_LEN = 113  # default prefix length (64 img + 48 lang + 1 state)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=TEXT_RMS_NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class LlamaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_INTERMEDIATE_SIZE, bias=False)
        self.up_proj = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_INTERMEDIATE_SIZE, bias=False)
        self.down_proj = nn.Linear(TEXT_INTERMEDIATE_SIZE, TEXT_HIDDEN_SIZE, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Model(nn.Module):
    """Single VLM Transformer Block with self-attention."""
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(TEXT_HIDDEN_SIZE)
        self.q_proj = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_NUM_HEADS * TEXT_HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(TEXT_NUM_HEADS * TEXT_HEAD_DIM, TEXT_HIDDEN_SIZE, bias=False)
        self.post_attention_layernorm = RMSNorm(TEXT_HIDDEN_SIZE)
        self.mlp = LlamaMLP()
        self.head_dim = TEXT_HEAD_DIM

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # Pre-attention norm + QKV projection
        normed = self.input_layernorm(hidden_states)
        normed = normed.to(dtype=self.q_proj.weight.dtype)
        B, L, _ = normed.shape

        q = self.q_proj(normed).view(B, L, -1, self.head_dim)
        k = self.k_proj(normed).view(B, L, -1, self.head_dim)
        v = self.v_proj(normed).view(B, L, -1, self.head_dim)

        # RoPE
        if position_ids is not None:
            q = self._apply_rope(q, position_ids)
            k = self._apply_rope(k, position_ids)

        # GQA attention
        num_kv_groups = TEXT_NUM_HEADS // TEXT_NUM_KV_HEADS
        k = k[:, :, :, None, :].expand(B, L, TEXT_NUM_KV_HEADS, num_kv_groups, self.head_dim)
        k = k.reshape(B, L, TEXT_NUM_HEADS, self.head_dim)
        v = v[:, :, :, None, :].expand(B, L, TEXT_NUM_KV_HEADS, num_kv_groups, self.head_dim)
        v = v.reshape(B, L, TEXT_NUM_HEADS, self.head_dim)

        q = q.to(torch.float32).transpose(1, 2)
        k = k.to(torch.float32).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(2, 3)) * (self.head_dim ** -0.5)

        if attention_mask is not None:
            big_neg = torch.finfo(attn.dtype).min
            attn = torch.where(attention_mask[:, None, :, :], attn, big_neg)

        probs = F.softmax(attn, dim=-1).to(v.dtype)
        att_out = torch.matmul(probs, v.permute(0, 2, 1, 3))
        att_out = att_out.permute(0, 2, 1, 3).reshape(B, -1, TEXT_NUM_HEADS * self.head_dim)

        # Output projection + residual
        if att_out.dtype != self.o_proj.weight.dtype:
            att_out = att_out.to(self.o_proj.weight.dtype)
        out = self.o_proj(att_out) + hidden_states
        residual = out.clone()

        # Post-attention norm + MLP + residual
        out = self.post_attention_layernorm(out)
        out = self.mlp(out)
        out = out + residual
        return out

    def _apply_rope(self, x, positions, max_wavelength=10_000):
        d_half = x.shape[-1] // 2
        device = x.device
        dtype = x.dtype
        x = x.to(torch.float32)
        freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
        timescale = max_wavelength ** freq_exponents
        radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
        radians = radians[..., None, :]
        sin = torch.sin(radians)
        cos = torch.cos(radians)
        x1, x2 = x.split(d_half, dim=-1)
        res = torch.empty_like(x)
        res[..., :d_half] = x1 * cos - x2 * sin
        res[..., d_half:] = x2 * cos + x1 * sin
        return res.to(dtype)


def get_inputs():
    B, L = 1, SEQ_LEN
    hidden_states = torch.randn(B, L, TEXT_HIDDEN_SIZE)
    attention_mask = torch.ones(B, L, L, dtype=torch.bool)
    position_ids = torch.arange(L).unsqueeze(0)
    return [hidden_states, attention_mask, position_ids]

def get_init_inputs():
    return []

def get_expected_output_shape():
    return [(1, SEQ_LEN, TEXT_HIDDEN_SIZE)]

def run_tests():
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            expected = get_expected_output_shape()
            assert tuple(output.shape) == tuple(expected[0]), f"Got {output.shape}, expected {expected[0]}"
            print(f"Input shape(s): {[x.shape for x in inputs]}")
            print(f"Output shape(s): [{output.shape}]")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback; traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
