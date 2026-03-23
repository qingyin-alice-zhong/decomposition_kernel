# GPT-OSS Decomposition Analysis

## Model Overview

GPT-OSS is a GPT-style decoder-only transformer with several advanced features:
- **Mixture of Experts (MoE)** MLP blocks with configurable number of experts and experts-per-token
- **RMSNorm** instead of LayerNorm
- **Rotary Position Embeddings (RoPE)** with YaRN scaling for extended context
- **Grouped Query Attention (GQA)** with 64 query heads and 8 key-value heads
- **Sliding window attention** applied to every other layer
- **Attention sink tokens** (learnable per-head bias in softmax)
- **SwiGLU activation** with clamping in the MLP

## Configuration Used

Test config (reduced for memory feasibility):
- num_hidden_layers: 2 (original: 36)
- num_experts: 4 (original: 128)
- experts_per_token: 2 (original: 4)
- vocab_size: 201088 (preserved)
- hidden_size: 2880 (preserved)
- intermediate_size: 2880 (preserved)
- head_dim: 64 (preserved)
- num_attention_heads: 64 (preserved)
- num_key_value_heads: 8 (preserved)
- sliding_window: 128 (preserved)
- rope_theta: 150000.0, rope_scaling_factor: 32.0

## Module Hierarchy

```
Transformer (L3)
  +-- nn.Embedding(201088, 2880, bf16)           [L0 kernel]
  +-- TransformerBlock x N (L2 layer)
  |     +-- AttentionBlock (L1 fusion)
  |     |     +-- RMSNorm(2880)                   [L0 kernel]
  |     |     +-- nn.Linear(2880, 5120, bf16)     [L0 kernel] (QKV projection)
  |     |     +-- RotaryEmbedding(head_dim=64)    [L0 kernel] (YaRN RoPE)
  |     |     +-- SDPA with GQA + Sinks           [L0 kernel] (sliding window on even layers)
  |     |     +-- nn.Linear(4096, 2880, bf16)     [L0 kernel] (output projection)
  |     |     +-- residual add
  |     +-- MLPBlock MoE (L1 fusion)
  |           +-- RMSNorm(2880)                   [L0 kernel]
  |           +-- nn.Linear(2880, 4, bf16)        [L0 kernel] (gate)
  |           +-- ExpertRouting(topk=2)            [L0 kernel] (topk + softmax)
  |           +-- MoE MLP1 (einsum + bias)         [L0 kernel] (4x5760x2880)
  |           +-- SwiGLU(limit=7.0)               [L0 kernel]
  |           +-- MoE MLP2 (einsum + bias)         [L0 kernel] (4x2880x2880)
  |           +-- ExpertCombine (einsum)           [L0 kernel]
  |           +-- residual add
  +-- RMSNorm(2880)                               [L0 kernel] (final norm)
  +-- nn.Linear(2880, 201088, bf16, no bias)      [L0 kernel] (unembedding)
```

## Data Flow (with shapes)

Input: [32] (int32 - token indices)
  -> Embedding: [32] -> [32, 2880] bf16
  -> TransformerBlock 0:
     -> AttentionBlock:
        -> RMSNorm: [32, 2880] -> [32, 2880] bf16
        -> QKV Linear: [32, 2880] -> [32, 5120] bf16
        -> Split Q/K/V: Q=[32, 4096], K=[32, 512], V=[32, 512]
        -> Reshape Q: [32, 8, 8, 64]  (GQA: 8 KV heads, 8 Q per KV)
        -> Reshape K: [32, 8, 64]
        -> Reshape V: [32, 8, 64]
        -> RoPE: Q=[32, 8, 8, 64], K=[32, 8, 64] (same shapes)
        -> SDPA: Q+K+V+Sinks -> [32, 4096] bf16 (sliding_window=128)
        -> Out Linear: [32, 4096] -> [32, 2880] bf16
        -> Residual add: [32, 2880]
     -> MLPBlock MoE:
        -> RMSNorm: [32, 2880] -> [32, 2880]
        -> Gate: [32, 2880] -> [32, 4] (4 experts)
        -> Routing: topk(2) -> weights=[32, 2], indices=[32, 2]
        -> MLP1: einsum [32, 2, 5760, 2880] x [32, 2880] -> [32, 2, 5760]
        -> SwiGLU: [32, 2, 5760] -> [32, 2, 2880]
        -> MLP2: einsum [32, 2, 2880, 2880] x [32, 2, 2880] -> [32, 2, 2880]
        -> Combine: einsum [32, 2, 2880] x [32, 2] -> [32, 2880]
        -> Residual add: [32, 2880]
  -> TransformerBlock 1: (same structure, sliding_window=0)
  -> Final RMSNorm: [32, 2880] -> [32, 2880]
  -> Unembedding: [32, 2880] -> [32, 201088] bf16
Output: [32, 201088] bf16

## External Dependencies Handled

1. `from gpt_oss.torch.weights import Checkpoint` - Removed (checkpoint loading not needed for architecture decomposition)
2. `torch.distributed` - Removed (set world_size=1 for single-process testing)

## Unique Kernel Files (13)

| File | Operation | Instance Count |
|------|-----------|---------------|
| embedding_201088x2880_bf16.py | nn.Embedding | 1 |
| rms_norm_2880_fp32.py | RMSNorm | 5 (2 attn + 2 mlp + 1 final) |
| linear_2880x5120_bf16.py | QKV Linear | 2 |
| rotary_embedding_64_fp32.py | RoPE with YaRN | 2 |
| sdpa_gqa_64_bf16.py | SDPA with GQA + Sinks | 2 |
| linear_4096x2880_bf16.py | Attention output projection | 2 |
| linear_2880x4_bf16.py | MoE gate | 2 |
| moe_expert_routing_bf16.py | TopK + Softmax routing | 2 |
| moe_mlp1_4x5760x2880_bf16.py | Expert MLP1 (einsum) | 2 |
| swiglu_bf16.py | SwiGLU activation | 2 |
| moe_mlp2_4x2880x2880_bf16.py | Expert MLP2 (einsum) | 2 |
| moe_expert_combine_bf16.py | Expert weighted sum | 2 |
| linear_2880x201088_bf16.py | Unembedding (no bias) | 1 |

## Verification Results

| Step | Status | Max Diff |
|------|--------|----------|
| Step 1: Model -> Layers | PASS | 0.00e+00 |
| Step 2: TransformerBlock -> Fusions | PASS | 0.00e+00 |
| Step 3a: AttentionBlock -> Kernels | PASS | 0.00e+00 |
| Step 3b: MLPBlock -> Kernels | PASS | 0.00e+00 |
| Composition Test | PASS | 0.00e+00 |
