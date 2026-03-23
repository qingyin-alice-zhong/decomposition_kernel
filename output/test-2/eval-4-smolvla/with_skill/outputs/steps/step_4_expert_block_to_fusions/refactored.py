"""
Refactored: Expert Transformer Block -> [rms_norm_720 x2, linear projections, expert_mlp]
Attention arithmetic (matmul, softmax, RoPE, GQA) stays as data plumbing (--skip-anticheat).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "children"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rms_norm_720_fp32 import Model as RMSNorm720
from linear_720x960_fp32 import Model as Linear720x960
from linear_720x320_fp32 import Model as Linear720x320
from linear_320x320_fp32 import Model as Linear320x320
from linear_960x720_fp32 import Model as Linear960x720
from expert_mlp import Model as ExpertMLP

EXPERT_HIDDEN_SIZE = 720
EXPERT_NUM_HEADS = 15
EXPERT_NUM_KV_HEADS = 5
EXPERT_HEAD_DIM = 64
EXPERT_RMS_NORM_EPS = 1e-5
CHUNK_SIZE = 50


class RefactoredModel(nn.Module):
    """Expert Transformer Block refactored to use child modules."""
    def __init__(self, is_cross_attn=False):
        super().__init__()
        self.is_cross_attn = is_cross_attn

        # RMSNorm kernels
        self.input_layernorm = RMSNorm720(EXPERT_HIDDEN_SIZE)
        self.post_attention_layernorm = RMSNorm720(EXPERT_HIDDEN_SIZE)

        # Q projection: always 720 -> 960
        _q = Linear720x960(EXPERT_HIDDEN_SIZE, EXPERT_NUM_HEADS * EXPERT_HEAD_DIM)
        self.q_proj = _q.linear

        # K/V projections: differ based on cross-attn mode
        if is_cross_attn:
            kv_input_dim = 320
            _k = Linear320x320(kv_input_dim, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM)
            self.k_proj = _k.linear
            _v = Linear320x320(kv_input_dim, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM)
            self.v_proj = _v.linear
        else:
            _k = Linear720x320(EXPERT_HIDDEN_SIZE, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM)
            self.k_proj = _k.linear
            _v = Linear720x320(EXPERT_HIDDEN_SIZE, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM)
            self.v_proj = _v.linear

        # O projection: 960 -> 720
        _o = Linear960x720(EXPERT_NUM_HEADS * EXPERT_HEAD_DIM, EXPERT_HIDDEN_SIZE)
        self.o_proj = _o.linear

        # MLP fusion
        self.mlp = ExpertMLP()

        self.head_dim = EXPERT_HEAD_DIM

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        normed = self.input_layernorm(hidden_states)
        normed = normed.to(dtype=self.q_proj.weight.dtype)
        B, L, _ = normed.shape
        q = self.q_proj(normed).view(B, L, -1, self.head_dim)
        k = self.k_proj(normed).view(B, L, -1, self.head_dim)
        v = self.v_proj(normed).view(B, L, -1, self.head_dim)

        if position_ids is not None:
            q = self._apply_rope(q, position_ids)
            k = self._apply_rope(k, position_ids)

        num_kv_groups = EXPERT_NUM_HEADS // EXPERT_NUM_KV_HEADS
        k = k[:, :, :, None, :].expand(B, L, EXPERT_NUM_KV_HEADS, num_kv_groups, self.head_dim)
        k = k.reshape(B, L, EXPERT_NUM_HEADS, self.head_dim)
        v = v[:, :, :, None, :].expand(B, L, EXPERT_NUM_KV_HEADS, num_kv_groups, self.head_dim)
        v = v.reshape(B, L, EXPERT_NUM_HEADS, self.head_dim)

        q = q.to(torch.float32).transpose(1, 2)
        k = k.to(torch.float32).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(2, 3)) * (self.head_dim ** -0.5)

        if attention_mask is not None:
            big_neg = torch.finfo(attn.dtype).min
            attn = torch.where(attention_mask[:, None, :, :], attn, big_neg)

        probs = F.softmax(attn, dim=-1).to(v.dtype)
        att_out = torch.matmul(probs, v.permute(0, 2, 1, 3))
        att_out = att_out.permute(0, 2, 1, 3).reshape(B, -1, EXPERT_NUM_HEADS * self.head_dim)

        if att_out.dtype != self.o_proj.weight.dtype:
            att_out = att_out.to(self.o_proj.weight.dtype)
        out = self.o_proj(att_out) + hidden_states
        residual = out.clone()
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
    B, L = 1, CHUNK_SIZE
    hidden_states = torch.randn(B, L, EXPERT_HIDDEN_SIZE)
    attention_mask = torch.ones(B, L, L, dtype=torch.bool)
    position_ids = torch.arange(L).unsqueeze(0)
    return [hidden_states, attention_mask, position_ids]

def get_init_inputs():
    return [False]  # is_cross_attn=False

def get_expected_output_shape():
    return [(1, CHUNK_SIZE, EXPERT_HIDDEN_SIZE)]
