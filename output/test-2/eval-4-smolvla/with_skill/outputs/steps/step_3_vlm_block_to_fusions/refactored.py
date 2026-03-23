"""
Refactored: VLM Transformer Block -> [rms_norm_960 x2, linear_960x960 x2, linear_960x320 x2, vlm_mlp]
Attention arithmetic (matmul, softmax, RoPE, GQA) stays as data plumbing (--skip-anticheat).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "children"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rms_norm_960_fp32 import Model as RMSNorm960
from linear_960x960_fp32 import Model as Linear960x960
from linear_960x320_fp32 import Model as Linear960x320
from vlm_mlp import Model as VlmMLP

TEXT_HIDDEN_SIZE = 960
TEXT_NUM_HEADS = 15
TEXT_NUM_KV_HEADS = 5
TEXT_HEAD_DIM = 64
TEXT_RMS_NORM_EPS = 1e-5
SEQ_LEN = 113


class RefactoredModel(nn.Module):
    """VLM Transformer Block refactored to use child modules."""
    def __init__(self):
        super().__init__()
        # RMSNorm kernels - RMSNorm960 has .weight as direct Parameter, matches original
        self.input_layernorm = RMSNorm960(TEXT_HIDDEN_SIZE)
        self.post_attention_layernorm = RMSNorm960(TEXT_HIDDEN_SIZE)

        # Linear kernels - expose inner .linear as the projection so weight names match
        _q = Linear960x960(TEXT_HIDDEN_SIZE, TEXT_NUM_HEADS * TEXT_HEAD_DIM)
        self.q_proj = _q.linear
        _k = Linear960x320(TEXT_HIDDEN_SIZE, TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM)
        self.k_proj = _k.linear
        _v = Linear960x320(TEXT_HIDDEN_SIZE, TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM)
        self.v_proj = _v.linear
        _o = Linear960x960(TEXT_NUM_HEADS * TEXT_HEAD_DIM, TEXT_HIDDEN_SIZE)
        self.o_proj = _o.linear

        # MLP fusion - structure matches original (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
        self.mlp = VlmMLP()

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

        # GQA expansion
        num_kv_groups = TEXT_NUM_HEADS // TEXT_NUM_KV_HEADS
        k = k[:, :, :, None, :].expand(B, L, TEXT_NUM_KV_HEADS, num_kv_groups, self.head_dim)
        k = k.reshape(B, L, TEXT_NUM_HEADS, self.head_dim)
        v = v[:, :, :, None, :].expand(B, L, TEXT_NUM_KV_HEADS, num_kv_groups, self.head_dim)
        v = v.reshape(B, L, TEXT_NUM_HEADS, self.head_dim)

        # Attention
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
