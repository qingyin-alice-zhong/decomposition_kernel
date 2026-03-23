"""
Component: SmolVLA VLAFlowMatching
Abstraction Level: model
Parent: root
Children: [vision_encoder, connector, text_embedding, vlm_transformer_block, expert_transformer_block,
           state_proj, action_in_proj, action_time_mlp, action_out_proj, vlm_final_norm, expert_final_norm]

Operations: Vision encoding (SigLIP), language embedding, state projection, action embedding,
            sinusoidal positional encoding, time-action MLP fusion, joint VLM+Expert transformer
            with self-attention and cross-attention, RoPE, output projection

Input Shapes:
  - pixel_values: (1, 3, 512, 512) dtype=float32
  - lang_tokens: (1, 48) dtype=int64
  - lang_masks: (1, 48) dtype=bool
  - state: (1, 32) dtype=float32
  - actions: (1, 50, 32) dtype=float32
  - noise: (1, 50, 32) dtype=float32
  - time: (1,) dtype=float32

Output Shapes:
  - v_t: (1, 50, 32) dtype=float32

Weight Shapes: ~500M parameters across vision encoder, text model, expert, projections
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Config constants (from SmolVLM2-500M-Video-Instruct)
# ============================================================================

# Vision (SigLIP)
VISION_HIDDEN_SIZE = 768
VISION_NUM_HEADS = 12
VISION_INTERMEDIATE_SIZE = 3072
VISION_NUM_LAYERS = 12
VISION_PATCH_SIZE = 16
VISION_IMAGE_SIZE = 512
VISION_NUM_PATCHES = (VISION_IMAGE_SIZE // VISION_PATCH_SIZE) ** 2  # 1024
VISION_NUM_POSITIONS = VISION_NUM_PATCHES  # 1024
VISION_NUM_CHANNELS = 3
VISION_LAYER_NORM_EPS = 1e-6

# Text / LLM (VLlama3)
TEXT_HIDDEN_SIZE = 960
TEXT_NUM_HEADS = 15
TEXT_NUM_KV_HEADS = 5
TEXT_HEAD_DIM = 64
TEXT_INTERMEDIATE_SIZE = 2560
TEXT_NUM_LAYERS = 32  # full model has 32 but we use 16
TEXT_RMS_NORM_EPS = 1e-5
TEXT_VOCAB_SIZE = 49280
TEXT_ROPE_THETA = 100000

# VLM subset
NUM_VLM_LAYERS = 16

# Expert
EXPERT_HIDDEN_SIZE = 720  # int(960 * 0.75)
EXPERT_INTERMEDIATE_SIZE = 2048
EXPERT_NUM_HEADS = 15
EXPERT_NUM_KV_HEADS = 5
EXPERT_HEAD_DIM = 64
NUM_EXPERT_LAYERS = 16
SELF_ATTN_EVERY_N_LAYERS = 2

# Connector
SCALE_FACTOR = 4
CONNECTOR_IN_SIZE = VISION_HIDDEN_SIZE * SCALE_FACTOR * SCALE_FACTOR  # 768 * 16 = 12288
CONNECTOR_OUT_SIZE = TEXT_HIDDEN_SIZE  # 960

# SmolVLA policy
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32
CHUNK_SIZE = 50
MIN_PERIOD = 4e-3
MAX_PERIOD = 4.0
PREFIX_LENGTH = -1  # no padding
ADD_IMAGE_SPECIAL_TOKENS = False

# Batch/sequence
BATCH_SIZE = 1
LANG_SEQ_LEN = 48

# Derived
NUM_IMG_TOKENS = VISION_NUM_PATCHES // (SCALE_FACTOR * SCALE_FACTOR)  # 1024 / 16 = 64


# ============================================================================
# Utility functions
# ============================================================================

def apply_rope(x, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
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


def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    dtype = torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Create 2D attention masks from padding and attention masks."""
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def pad_tensor(tensor, max_len, pad_value=0):
    """Pads a tensor along sequence dimension to match max_len."""
    b, d = tensor.shape[:2]
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor
    return padded_tensor


# ============================================================================
# Vision Encoder (SigLIP)
# ============================================================================

class SigLIPVisionEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            VISION_NUM_CHANNELS, VISION_HIDDEN_SIZE,
            kernel_size=VISION_PATCH_SIZE, stride=VISION_PATCH_SIZE, padding=0
        )
        self.position_embedding = nn.Embedding(VISION_NUM_POSITIONS, VISION_HIDDEN_SIZE)
        self.register_buffer("position_ids", torch.arange(VISION_NUM_POSITIONS).unsqueeze(0))

    def forward(self, pixel_values):
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        patch_embeds = patch_embeds + self.position_embedding(self.position_ids)
        return patch_embeds


class SigLIPAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = VISION_NUM_HEADS
        self.head_dim = VISION_HIDDEN_SIZE // VISION_NUM_HEADS
        self.q_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.k_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.v_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.out_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, VISION_HIDDEN_SIZE)
        return self.out_proj(out)


class SigLIPMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(VISION_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE, bias=True)
        self.fc2 = nn.Linear(VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE, bias=True)

    def forward(self, x):
        # gelu_pytorch_tanh
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class SigLIPEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SigLIPAttention()
        self.layer_norm1 = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)
        self.mlp = SigLIPMLP()
        self.layer_norm2 = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = x + residual
        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class SigLIPVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = SigLIPVisionEmbeddings()
        self.layers = nn.ModuleList([SigLIPEncoderLayer() for _ in range(VISION_NUM_LAYERS)])
        self.post_layernorm = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)
        for layer in self.layers:
            x = layer(x)
        x = self.post_layernorm(x)
        return x


# ============================================================================
# Connector (pixel shuffle + linear projection)
# ============================================================================

class SmolVLMConnector(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(CONNECTOR_IN_SIZE, CONNECTOR_OUT_SIZE, bias=False)

    def forward(self, image_hidden_states):
        # Pixel shuffle: (B, L, D) -> (B, L/scale^2, D*scale^2)
        B, L, D = image_hidden_states.shape
        h = w = int(L ** 0.5)
        image_hidden_states = image_hidden_states.view(B, h, w, D)
        # Reshape for pixel shuffle
        image_hidden_states = image_hidden_states.view(
            B, h // SCALE_FACTOR, SCALE_FACTOR, w // SCALE_FACTOR, SCALE_FACTOR, D
        )
        image_hidden_states = image_hidden_states.permute(0, 1, 3, 2, 4, 5)
        image_hidden_states = image_hidden_states.reshape(
            B, h // SCALE_FACTOR * w // SCALE_FACTOR, D * SCALE_FACTOR * SCALE_FACTOR
        )
        return self.proj(image_hidden_states)


# ============================================================================
# LLM Components (RMSNorm, MLP, Attention for VLM and Expert)
# ============================================================================

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
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class VLMTransformerLayer(nn.Module):
    """A single VLM (text model) transformer layer."""
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(TEXT_HIDDEN_SIZE)
        self.q_proj = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_NUM_HEADS * TEXT_HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(TEXT_HIDDEN_SIZE, TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(TEXT_NUM_HEADS * TEXT_HEAD_DIM, TEXT_HIDDEN_SIZE, bias=False)
        self.post_attention_layernorm = RMSNorm(TEXT_HIDDEN_SIZE)
        self.mlp = LlamaMLP(TEXT_HIDDEN_SIZE, TEXT_INTERMEDIATE_SIZE)
        self.head_dim = TEXT_HEAD_DIM

    def get_qkv(self, hidden_states):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(dtype=self.q_proj.weight.dtype)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape)
        k = self.k_proj(hidden_states).view(hidden_shape)
        v = self.v_proj(hidden_states).view(hidden_shape)
        return q, k, v, hidden_states

    def post_attention(self, att_output, hidden_states_input):
        if att_output.dtype != self.o_proj.weight.dtype:
            att_output = att_output.to(self.o_proj.weight.dtype)
        out = self.o_proj(att_output)
        out = out + hidden_states_input
        residual = out.clone()
        out = self.post_attention_layernorm(out)
        out = self.mlp(out)
        out = out + residual
        return out


class ExpertTransformerLayer(nn.Module):
    """A single Expert transformer layer."""
    def __init__(self, is_cross_attn=False):
        super().__init__()
        self.input_layernorm = RMSNorm(EXPERT_HIDDEN_SIZE)
        self.q_proj = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_NUM_HEADS * EXPERT_HEAD_DIM, bias=False)
        if is_cross_attn:
            # Cross-attention: k, v take input from VLM KV dimension
            kv_input_dim = TEXT_NUM_KV_HEADS * TEXT_HEAD_DIM  # 320
            self.k_proj = nn.Linear(kv_input_dim, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False)
            self.v_proj = nn.Linear(kv_input_dim, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False)
        else:
            self.k_proj = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False)
            self.v_proj = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_NUM_KV_HEADS * EXPERT_HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(EXPERT_NUM_HEADS * EXPERT_HEAD_DIM, EXPERT_HIDDEN_SIZE, bias=False)
        self.post_attention_layernorm = RMSNorm(EXPERT_HIDDEN_SIZE)
        self.mlp = LlamaMLP(EXPERT_HIDDEN_SIZE, EXPERT_INTERMEDIATE_SIZE)
        self.head_dim = EXPERT_HEAD_DIM
        self.is_cross_attn = is_cross_attn

    def get_qkv(self, hidden_states):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(dtype=self.q_proj.weight.dtype)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape)
        k = self.k_proj(hidden_states).view(hidden_shape)
        v = self.v_proj(hidden_states).view(hidden_shape)
        return q, k, v, hidden_states

    def get_cross_qkv(self, hidden_states, vlm_key_states, vlm_value_states):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = hidden_states.to(dtype=self.q_proj.weight.dtype)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape)

        # Reshape VLM KV states for cross-attention
        _k = vlm_key_states.to(dtype=self.k_proj.weight.dtype).view(*vlm_key_states.shape[:2], -1)
        k = self.k_proj(_k).view(*_k.shape[:-1], -1, self.head_dim)

        _v = vlm_value_states.to(dtype=self.v_proj.weight.dtype).view(*vlm_value_states.shape[:2], -1)
        v = self.v_proj(_v).view(*_v.shape[:-1], -1, self.head_dim)

        return q, k, v, hidden_states

    def post_attention(self, att_output, hidden_states_input):
        if att_output.dtype != self.o_proj.weight.dtype:
            att_output = att_output.to(self.o_proj.weight.dtype)
        out = self.o_proj(att_output)
        out = out + hidden_states_input
        residual = out.clone()
        out = self.post_attention_layernorm(out)
        out = self.mlp(out)
        out = out + residual
        return out


# ============================================================================
# Eager Attention
# ============================================================================

def eager_attention_forward(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
    """Eager attention implementation matching SmolVLMWithExpertModel."""
    num_att_heads = TEXT_NUM_HEADS
    num_key_value_heads = TEXT_NUM_KV_HEADS
    num_key_value_groups = num_att_heads // num_key_value_heads

    sequence_length = key_states.shape[1]

    key_states = key_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    key_states = key_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )

    value_states = value_states[:, :, :, None, :].expand(
        batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
    )
    value_states = value_states.reshape(
        batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
    )

    query_states = query_states.to(dtype=torch.float32)
    key_states = key_states.to(dtype=torch.float32)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)

    att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
    att_weights *= head_dim ** -0.5

    att_weights = att_weights.to(dtype=torch.float32)
    big_neg = torch.finfo(att_weights.dtype).min
    masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
    probs = F.softmax(masked_att_weights, dim=-1)
    probs = probs.to(dtype=value_states.dtype)

    att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
    att_output = att_output.permute(0, 2, 1, 3)
    att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)
    return att_output


# ============================================================================
# Full Model
# ============================================================================

class Model(nn.Module):
    """
    Self-contained SmolVLA (VLAFlowMatching) model.
    Wraps the full forward pass: vision encoding, language embedding,
    state/action embedding, joint VLM+Expert transformer, output projection.
    """
    def __init__(self):
        super().__init__()

        # Vision encoder
        self.vision_encoder = SigLIPVisionEncoder()

        # Connector
        self.connector = SmolVLMConnector()

        # Text embeddings
        self.text_embed_tokens = nn.Embedding(TEXT_VOCAB_SIZE, TEXT_HIDDEN_SIZE)

        # VLM transformer layers (first 16 of 32)
        self.vlm_layers = nn.ModuleList([VLMTransformerLayer() for _ in range(NUM_VLM_LAYERS)])

        # Expert transformer layers (16 layers, alternating self-attn and cross-attn)
        expert_layers = []
        for i in range(NUM_EXPERT_LAYERS):
            is_cross = (SELF_ATTN_EVERY_N_LAYERS > 0 and i % SELF_ATTN_EVERY_N_LAYERS != 0)
            expert_layers.append(ExpertTransformerLayer(is_cross_attn=is_cross))
        self.expert_layers = nn.ModuleList(expert_layers)

        # Final norms
        self.vlm_norm = RMSNorm(TEXT_HIDDEN_SIZE)
        self.expert_norm = RMSNorm(EXPERT_HIDDEN_SIZE)

        # Projection layers
        self.state_proj = nn.Linear(MAX_STATE_DIM, TEXT_HIDDEN_SIZE)
        self.action_in_proj = nn.Linear(MAX_ACTION_DIM, EXPERT_HIDDEN_SIZE)
        self.action_out_proj = nn.Linear(EXPERT_HIDDEN_SIZE, MAX_ACTION_DIM)
        self.action_time_mlp_in = nn.Linear(EXPERT_HIDDEN_SIZE * 2, EXPERT_HIDDEN_SIZE)
        self.action_time_mlp_out = nn.Linear(EXPERT_HIDDEN_SIZE, EXPERT_HIDDEN_SIZE)

    def embed_image(self, pixel_values):
        """Encode image through SigLIP vision encoder + connector."""
        image_hidden_states = self.vision_encoder(pixel_values)
        image_hidden_states = self.connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens):
        """Embed language tokens via text model embedding layer."""
        return self.text_embed_tokens(tokens)

    def embed_prefix(self, pixel_values, lang_tokens, lang_masks, state):
        """Embed all prefix inputs: image, language, state."""
        embs = []
        pad_masks = []
        att_masks_list = []

        # Image embedding
        img_emb = self.embed_image(pixel_values)
        img_emb_dim = img_emb.shape[-1]
        img_emb = img_emb * torch.tensor(img_emb_dim ** 0.5, dtype=img_emb.dtype, device=img_emb.device)

        bsize, num_img_embs = img_emb.shape[:2]
        img_mask = torch.ones(bsize, num_img_embs, dtype=torch.bool, device=img_emb.device)

        embs.append(img_emb)
        pad_masks.append(img_mask)
        att_masks_list += [0] * num_img_embs

        # Language embedding
        lang_emb = self.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks_list += [0] * num_lang_embs

        # State embedding
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        device = state_emb.device
        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks_list += [1] * states_seq_len

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks_list, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy actions + timestep for expert processing."""
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep, EXPERT_HIDDEN_SIZE, MIN_PERIOD, MAX_PERIOD, device=device
        )
        time_emb = time_emb.type(dtype=dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs = action_time_emb
        bsize, action_time_dim = embs.shape[:2]
        pad_masks = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        att_masks = torch.ones(bsize, CHUNK_SIZE, dtype=embs.dtype, device=device)

        return embs, pad_masks, att_masks

    def forward_joint_transformer(self, prefix_embs, suffix_embs, attention_mask, position_ids):
        """Run joint VLM + Expert transformer layers."""
        inputs_embeds = [prefix_embs, suffix_embs]
        batch_size = prefix_embs.shape[0] if prefix_embs is not None else suffix_embs.shape[0]
        head_dim = TEXT_HEAD_DIM

        for layer_idx in range(NUM_VLM_LAYERS):
            vlm_layer = self.vlm_layers[layer_idx]
            expert_layer = self.expert_layers[layer_idx]

            is_self_attn_step = (SELF_ATTN_EVERY_N_LAYERS > 0 and layer_idx % SELF_ATTN_EVERY_N_LAYERS == 0)

            if is_self_attn_step:
                # Self-attention: concatenate all embeddings
                query_states = []
                key_states = []
                value_states = []

                for i, hidden_states in enumerate(inputs_embeds):
                    if hidden_states is None:
                        continue
                    layer = vlm_layer if i == 0 else expert_layer
                    if layer is None:
                        continue
                    q, k, v, _ = layer.get_qkv(hidden_states)
                    query_states.append(q)
                    key_states.append(k)
                    value_states.append(v)

                query_states = torch.cat(query_states, dim=1)
                key_states = torch.cat(key_states, dim=1)
                value_states = torch.cat(value_states, dim=1)

                seq_len = query_states.shape[1]
                if seq_len < position_ids.shape[1]:
                    _position_ids = position_ids[:, :seq_len]
                    _attention_mask = attention_mask[:, :seq_len, :seq_len]
                else:
                    _position_ids = position_ids
                    _attention_mask = attention_mask

                query_states = apply_rope(query_states, _position_ids)
                key_states = apply_rope(key_states, _position_ids)

                att_output = eager_attention_forward(
                    _attention_mask, batch_size, head_dim, query_states, key_states, value_states
                )

                # Post-attention processing
                start = 0
                outputs = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = vlm_layer if i == 0 else expert_layer
                    if hidden_states is not None and layer is not None:
                        end = start + hidden_states.shape[1]
                        out = layer.post_attention(att_output[:, start:end], hidden_states)
                        outputs.append(out)
                        start = end
                    else:
                        outputs.append(hidden_states)
            else:
                # Cross-attention mode
                outputs = []

                # VLM self-attention first (if prefix_embs present)
                if inputs_embeds[0] is not None:
                    seq_len = inputs_embeds[0].shape[1]
                    position_id = position_ids[:, :seq_len]
                    prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

                    q, k, v, _ = vlm_layer.get_qkv(inputs_embeds[0])
                    q = apply_rope(q, position_id)
                    k = apply_rope(k, position_id)

                    att_output = eager_attention_forward(
                        prefix_attention_mask, batch_size, head_dim, q, k, v
                    )

                    vlm_out = vlm_layer.post_attention(att_output, inputs_embeds[0])
                    outputs.append(vlm_out)

                    # Store KV for cross-attention
                    vlm_key_states = k
                    vlm_value_states = v
                else:
                    outputs.append(None)
                    # Would need cached KV - not implementing cache for training forward

                # Expert cross-attention
                if inputs_embeds[1] is not None and expert_layer is not None:
                    expert_position_id = position_ids[:, inputs_embeds[0].shape[1]:] if inputs_embeds[0] is not None else position_ids
                    expert_position_id = expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values

                    q, k, v, _ = expert_layer.get_cross_qkv(inputs_embeds[1], vlm_key_states, vlm_value_states)
                    q = apply_rope(q, expert_position_id)

                    expert_attn_mask = attention_mask[
                        :, -inputs_embeds[1].shape[1]:, :k.shape[1]:
                    ]

                    att_output = eager_attention_forward(
                        expert_attn_mask, batch_size, head_dim, q, k, v
                    )

                    expert_out = expert_layer.post_attention(att_output, inputs_embeds[1])
                    outputs.append(expert_out)
                else:
                    outputs.append(inputs_embeds[1])

            inputs_embeds = outputs

        # Final norms
        final_outputs = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                norm = self.vlm_norm if i == 0 else self.expert_norm
                final_outputs.append(norm(hidden_states))
            else:
                final_outputs.append(None)

        return final_outputs

    def forward(self, pixel_values, lang_tokens, lang_masks, state, actions, noise, time):
        """Full training forward pass."""
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, lang_tokens, lang_masks, state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        outputs = self.forward_joint_transformer(
            prefix_embs, suffix_embs, att_2d_masks, position_ids
        )

        suffix_out = outputs[1]
        suffix_out = suffix_out[:, -CHUNK_SIZE:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        return v_t


def get_inputs():
    """Generate test inputs matching the training forward pass."""
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


def get_expected_output_shape():
    return [(BATCH_SIZE, CHUNK_SIZE, MAX_ACTION_DIM)]


def run_tests():
    """Verify this component executes correctly."""
    try:
        model = Model(*get_init_inputs())
        model.eval()
        with torch.no_grad():
            inputs = get_inputs()
            output = model(*inputs)
            assert output is not None, "Output is None"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"
            expected_shapes = get_expected_output_shape()
            actual_shapes = [output.shape] if isinstance(output, torch.Tensor) else [o.shape for o in output]
            for i, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes)):
                assert tuple(actual) == tuple(expected), \
                    f"Output {i} shape mismatch: got {actual}, expected {expected}"
            print(f"Input shape(s): {[x.shape if hasattr(x, 'shape') else type(x) for x in inputs]}")
            print(f"Output shape(s): {actual_shapes}")
            print("PASS")
            return True
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
