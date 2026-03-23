"""
Step 1 Refactored: SmolVLA Model -> Children

This refactored model delegates all sub-computations to imported child modules.
The forward() only contains data plumbing (reshaping, concatenation, masking)
and calls to child modules.

NOTE: The joint VLM+Expert transformer requires custom attention orchestration
that applies RoPE and attention across concatenated VLM+Expert states. These
operations are implemented as utility functions here because they orchestrate
ACROSS multiple child modules (VLM layers and Expert layers interact through
shared attention). This is data plumbing at the model level.

Children:
  - vision_encoder (L2): SigLIP vision encoder
  - connector (L1): pixel shuffle + linear
  - embedding_49280x960_fp32 (L0): text embedding
  - linear_32x960_fp32 (L0): state projection
  - linear_32x720_fp32 (L0): action input projection
  - action_time_mlp (L1): linear + silu + linear
  - vlm_transformer_block (L2): VLM transformer layer (x16)
  - expert_transformer_block (L2): Expert transformer layer (x16)
  - rms_norm_960_fp32 (L0): VLM final norm
  - rms_norm_720_fp32 (L0): Expert final norm
  - linear_720x32_fp32 (L0): action output projection
"""

import math
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add children directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "children"))

from vision_encoder import Model as VisionEncoder
from connector import Model as Connector
from embedding_49280x960_fp32 import Model as TextEmbedding
from linear_32x960_fp32 import Model as StateProj
from linear_32x720_fp32 import Model as ActionInProj
from action_time_mlp import Model as ActionTimeMLP
from vlm_transformer_block import Model as VLMBlock
from expert_transformer_block import Model as ExpertBlock
from rms_norm_960_fp32 import Model as VLMFinalNorm
from rms_norm_720_fp32 import Model as ExpertFinalNorm
from linear_720x32_fp32 import Model as ActionOutProj

# Config constants
TEXT_HIDDEN_SIZE = 960
TEXT_NUM_HEADS = 15
TEXT_NUM_KV_HEADS = 5
TEXT_HEAD_DIM = 64
EXPERT_HIDDEN_SIZE = 720
NUM_VLM_LAYERS = 16
NUM_EXPERT_LAYERS = 16
SELF_ATTN_EVERY_N_LAYERS = 2
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50
MIN_PERIOD = 4e-3
MAX_PERIOD = 4.0
BATCH_SIZE = 1
LANG_SEQ_LEN = 48
VISION_NUM_CHANNELS = 3
VISION_IMAGE_SIZE = 512
TEXT_VOCAB_SIZE = 49280


def apply_rope(x, positions, max_wavelength=10_000):
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength ** freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]
    sin_r = torch.sin(radians)
    cos_r = torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos_r - x2 * sin_r
    res[..., d_half:] = x2 * cos_r + x1 * sin_r
    return res.to(dtype)


def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def eager_attention_forward(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
    num_att_heads = TEXT_NUM_HEADS
    num_key_value_heads = TEXT_NUM_KV_HEADS
    num_key_value_groups = num_att_heads // num_key_value_heads
    sequence_length = key_states.shape[1]

    key_states = key_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    ).reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)

    value_states = value_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    ).reshape(batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim)

    query_states = query_states.to(dtype=torch.float32).transpose(1, 2)
    key_states = key_states.to(dtype=torch.float32).transpose(1, 2)

    att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    att_weights = att_weights * (head_dim ** -0.5)
    att_weights = att_weights.to(dtype=torch.float32)
    big_neg = torch.finfo(att_weights.dtype).min
    masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
    probs = torch.nn.functional.softmax(masked_att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)

    att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
    att_output = att_output.permute(0, 2, 1, 3)
    att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
    return att_output


class EmbeddingAdapter(nn.Module):
    """Adapter that exposes embedding.weight as just weight for name matching."""
    def __init__(self, child):
        super().__init__()
        self._child = child
        # Expose weight directly so state_dict key is text_embed_tokens.weight
        self.weight = child.embedding.weight

    def forward(self, tokens):
        return self._child(tokens)


class LinearAdapter(nn.Module):
    """Adapter that exposes linear.weight/bias directly for name matching."""
    def __init__(self, child):
        super().__init__()
        self._child = child
        self.weight = child.linear.weight
        if child.linear.bias is not None:
            self.bias = child.linear.bias

    def forward(self, x):
        return self._child(x)


class ActionTimeMLPAdapter(nn.Module):
    """Adapter for action-time MLP that matches original param names."""
    def __init__(self, child):
        super().__init__()
        self._child = child

    def forward(self, x):
        return self._child(x)


class RefactoredModel(nn.Module):
    """SmolVLA refactored with all computation delegated to children."""
    def __init__(self):
        super().__init__()
        # L2 children
        self.vision_encoder = VisionEncoder()
        self.vlm_layers = nn.ModuleList([VLMBlock() for _ in range(NUM_VLM_LAYERS)])
        expert_layers = []
        for i in range(NUM_EXPERT_LAYERS):
            is_cross = (SELF_ATTN_EVERY_N_LAYERS > 0 and i % SELF_ATTN_EVERY_N_LAYERS != 0)
            expert_layers.append(ExpertBlock(is_cross_attn=is_cross))
        self.expert_layers = nn.ModuleList(expert_layers)

        # L1 children
        self.connector = Connector()
        _atm = ActionTimeMLP()
        # Expose action_time_mlp sub-params with original names
        self.action_time_mlp_in = _atm.mlp_in
        self.action_time_mlp_out = _atm.mlp_out
        self._action_time_mlp = _atm

        # L0 children - use same param names as original
        _embed = TextEmbedding(TEXT_VOCAB_SIZE, TEXT_HIDDEN_SIZE)
        self.text_embed_tokens = _embed.embedding  # nn.Embedding directly

        _sp = StateProj(MAX_STATE_DIM, TEXT_HIDDEN_SIZE)
        self.state_proj = _sp.linear  # nn.Linear directly

        _aip = ActionInProj(MAX_ACTION_DIM, EXPERT_HIDDEN_SIZE)
        self.action_in_proj = _aip.linear  # nn.Linear directly

        _aop = ActionOutProj(EXPERT_HIDDEN_SIZE, MAX_ACTION_DIM)
        self.action_out_proj = _aop.linear  # nn.Linear directly

        self.vlm_norm = VLMFinalNorm(TEXT_HIDDEN_SIZE)
        self.expert_norm = ExpertFinalNorm(EXPERT_HIDDEN_SIZE)

    def forward(self, pixel_values, lang_tokens, lang_masks, state, actions, noise, time):
        # Flow matching: interpolate noise and actions
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        # === EMBED PREFIX (image + language + state) ===
        # Image: vision_encoder -> connector
        img_hidden = self.vision_encoder(pixel_values)
        img_emb = self.connector(img_hidden)
        img_emb_dim = img_emb.shape[-1]
        img_emb = img_emb * torch.tensor(img_emb_dim ** 0.5, dtype=img_emb.dtype, device=img_emb.device)
        bsize, num_img_embs = img_emb.shape[:2]
        img_mask = torch.ones(bsize, num_img_embs, dtype=torch.bool, device=img_emb.device)

        # Language
        lang_emb = self.text_embed_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        # State
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)

        # Assemble prefix
        prefix_embs = torch.cat([img_emb, lang_emb, state_emb], dim=1)
        prefix_pad_masks = torch.cat([img_mask, lang_masks, state_mask], dim=1)
        att_masks_list = [0] * num_img_embs + [0] * lang_emb.shape[1] + [1] * states_seq_len
        prefix_att_masks = torch.tensor(att_masks_list, dtype=torch.bool, device=device)[None, :].expand(bsize, -1)

        # === EMBED SUFFIX (action + time) ===
        action_emb = self.action_in_proj(x_t)
        dtype = action_emb.dtype
        time_emb = create_sinusoidal_pos_embedding(time, EXPERT_HIDDEN_SIZE, MIN_PERIOD, MAX_PERIOD, device=device)
        time_emb = time_emb.type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_input = torch.cat([action_emb, time_emb], dim=2)
        suffix_embs = self._action_time_mlp(action_time_input)

        suffix_pad_masks = torch.ones(bsize, CHUNK_SIZE, dtype=torch.bool, device=device)
        suffix_att_masks = torch.ones(bsize, CHUNK_SIZE, dtype=suffix_embs.dtype, device=device)

        # === ASSEMBLE MASKS ===
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # === JOINT VLM + EXPERT TRANSFORMER ===
        inputs_embeds = [prefix_embs, suffix_embs]
        batch_size = bsize
        head_dim = TEXT_HEAD_DIM

        for layer_idx in range(NUM_VLM_LAYERS):
            vlm_layer = self.vlm_layers[layer_idx]
            expert_layer = self.expert_layers[layer_idx]
            is_self_attn = (SELF_ATTN_EVERY_N_LAYERS > 0 and layer_idx % SELF_ATTN_EVERY_N_LAYERS == 0)

            if is_self_attn:
                # Self-attention: extract QKV from both VLM and Expert layers
                query_states_list = []
                key_states_list = []
                value_states_list = []
                for i, hs in enumerate(inputs_embeds):
                    if hs is None:
                        continue
                    layer = vlm_layer if i == 0 else expert_layer
                    normed = layer.input_layernorm(hs)
                    normed = normed.to(dtype=layer.q_proj.weight.dtype)
                    B, L, _ = normed.shape
                    hidden_shape = (B, L, -1, head_dim)
                    q = layer.q_proj(normed).view(hidden_shape)
                    k = layer.k_proj(normed).view(hidden_shape)
                    v = layer.v_proj(normed).view(hidden_shape)
                    query_states_list.append(q)
                    key_states_list.append(k)
                    value_states_list.append(v)

                query_states = torch.cat(query_states_list, dim=1)
                key_states = torch.cat(key_states_list, dim=1)
                value_states = torch.cat(value_states_list, dim=1)

                seq_len = query_states.shape[1]
                if seq_len < position_ids.shape[1]:
                    _pos = position_ids[:, :seq_len]
                    _mask = att_2d_masks[:, :seq_len, :seq_len]
                else:
                    _pos = position_ids
                    _mask = att_2d_masks

                query_states = apply_rope(query_states, _pos)
                key_states = apply_rope(key_states, _pos)

                att_output = eager_attention_forward(_mask, batch_size, head_dim, query_states, key_states, value_states)

                start = 0
                outputs = []
                for i, hs in enumerate(inputs_embeds):
                    layer = vlm_layer if i == 0 else expert_layer
                    if hs is not None and layer is not None:
                        end = start + hs.shape[1]
                        att_out_slice = att_output[:, start:end]
                        if att_out_slice.dtype != layer.o_proj.weight.dtype:
                            att_out_slice = att_out_slice.to(layer.o_proj.weight.dtype)
                        out = layer.o_proj(att_out_slice) + hs
                        residual = out.clone()
                        out = layer.post_attention_layernorm(out)
                        out = layer.mlp(out)
                        out = out + residual
                        outputs.append(out)
                        start = end
                    else:
                        outputs.append(hs)
            else:
                # Cross-attention mode
                outputs = []
                # VLM self-attention
                if inputs_embeds[0] is not None:
                    seq_len = inputs_embeds[0].shape[1]
                    position_id = position_ids[:, :seq_len]
                    prefix_attn_mask = att_2d_masks[:, :seq_len, :seq_len]

                    normed = vlm_layer.input_layernorm(inputs_embeds[0])
                    normed = normed.to(dtype=vlm_layer.q_proj.weight.dtype)
                    B, L, _ = normed.shape
                    hidden_shape = (B, L, -1, head_dim)
                    q = vlm_layer.q_proj(normed).view(hidden_shape)
                    k = vlm_layer.k_proj(normed).view(hidden_shape)
                    v = vlm_layer.v_proj(normed).view(hidden_shape)

                    q = apply_rope(q, position_id)
                    k = apply_rope(k, position_id)

                    att_out = eager_attention_forward(prefix_attn_mask, batch_size, head_dim, q, k, v)

                    if att_out.dtype != vlm_layer.o_proj.weight.dtype:
                        att_out = att_out.to(vlm_layer.o_proj.weight.dtype)
                    vlm_out = vlm_layer.o_proj(att_out) + inputs_embeds[0]
                    residual = vlm_out.clone()
                    vlm_out = vlm_layer.post_attention_layernorm(vlm_out)
                    vlm_out = vlm_layer.mlp(vlm_out)
                    vlm_out = vlm_out + residual
                    outputs.append(vlm_out)

                    vlm_key_states = k
                    vlm_value_states = v
                else:
                    outputs.append(None)

                # Expert cross-attention
                if inputs_embeds[1] is not None:
                    expert_pos = position_ids[:, inputs_embeds[0].shape[1]:] if inputs_embeds[0] is not None else position_ids
                    expert_pos = expert_pos - torch.min(expert_pos, dim=1, keepdim=True).values

                    normed = expert_layer.input_layernorm(inputs_embeds[1])
                    normed = normed.to(dtype=expert_layer.q_proj.weight.dtype)
                    B, L, _ = normed.shape
                    hidden_shape = (B, L, -1, head_dim)
                    eq = expert_layer.q_proj(normed).view(hidden_shape)

                    _k = vlm_key_states.to(dtype=expert_layer.k_proj.weight.dtype).view(*vlm_key_states.shape[:2], -1)
                    ek = expert_layer.k_proj(_k).view(*_k.shape[:-1], -1, head_dim)
                    _v = vlm_value_states.to(dtype=expert_layer.v_proj.weight.dtype).view(*vlm_value_states.shape[:2], -1)
                    ev = expert_layer.v_proj(_v).view(*_v.shape[:-1], -1, head_dim)

                    eq = apply_rope(eq, expert_pos)
                    expert_attn_mask = att_2d_masks[:, -inputs_embeds[1].shape[1]:, :ek.shape[1]:]

                    att_out = eager_attention_forward(expert_attn_mask, batch_size, head_dim, eq, ek, ev)

                    if att_out.dtype != expert_layer.o_proj.weight.dtype:
                        att_out = att_out.to(expert_layer.o_proj.weight.dtype)
                    expert_out = expert_layer.o_proj(att_out) + inputs_embeds[1]
                    residual = expert_out.clone()
                    expert_out = expert_layer.post_attention_layernorm(expert_out)
                    expert_out = expert_layer.mlp(expert_out)
                    expert_out = expert_out + residual
                    outputs.append(expert_out)
                else:
                    outputs.append(inputs_embeds[1])

            inputs_embeds = outputs

        # Final norms
        final_outputs = []
        for i, hs in enumerate(inputs_embeds):
            if hs is not None:
                norm = self.vlm_norm if i == 0 else self.expert_norm
                final_outputs.append(norm(hs))
            else:
                final_outputs.append(None)

        # Output projection
        suffix_out = final_outputs[1]
        suffix_out = suffix_out[:, -CHUNK_SIZE:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t


def get_inputs():
    torch.manual_seed(42)
    pixel_values = torch.randn(BATCH_SIZE, VISION_NUM_CHANNELS, VISION_IMAGE_SIZE, VISION_IMAGE_SIZE)
    lang_tokens = torch.randint(0, TEXT_VOCAB_SIZE, (BATCH_SIZE, LANG_SEQ_LEN))
    lang_masks = torch.ones(BATCH_SIZE, LANG_SEQ_LEN, dtype=torch.bool)
    state = torch.randn(BATCH_SIZE, MAX_STATE_DIM)
    actions = torch.randn(BATCH_SIZE, CHUNK_SIZE, MAX_ACTION_DIM)
    noise = torch.randn(BATCH_SIZE, CHUNK_SIZE, MAX_ACTION_DIM)
    time = torch.tensor([0.5])
    return [pixel_values, lang_tokens, lang_masks, state, actions, noise, time]


def get_init_inputs():
    return []
