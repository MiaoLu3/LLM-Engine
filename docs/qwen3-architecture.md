# Qwen3 Model Architecture

This document explains the custom Qwen3 model implementation and how the different modules under `llm_engine/model/` are combined.

## Model Directory Structure

```
llm_engine/model/
├── __init__.py      # Exports all public APIs
├── rope.py          # Rotary Position Embeddings
├── layers.py        # RMSNorm, SwiGLU MLP
├── attention.py     # FlashAttention, PagedAttention, Qwen3Attention
├── qwen3.py         # Full model: Config, DecoderLayer, Model, ForCausalLM
├── loader.py        # Weight loading from HuggingFace
└── executor.py      # Forward pass execution wrapper
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Qwen3ForCausalLM                                 │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                         Qwen3Model                                │  │
│  │  ┌─────────────────┐                                              │  │
│  │  │  embed_tokens   │  [vocab_size=151936, hidden_size=2560]       │  │
│  │  └────────┬────────┘                                              │  │
│  │           ▼                                                       │  │
│  │  ┌─────────────────┐                                              │  │
│  │  │  RotaryEmbed    │  Precomputes cos/sin for positions           │  │
│  │  └────────┬────────┘                                              │  │
│  │           ▼                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │              Qwen3DecoderLayer × 40                         │  │  │
│  │  │  ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐  │  │  │
│  │  │  │ RMSNorm     │───►│ Qwen3Attention  │───►│  Residual   │  │  │  │
│  │  │  │ (input)     │    │ (GQA + RoPE)    │    │  Add        │  │  │  │
│  │  │  └─────────────┘    └─────────────────┘    └──────┬──────┘  │  │  │
│  │  │                                                    │        │  │  │
│  │  │  ┌─────────────┐    ┌─────────────────┐    ┌──────▼──────┐  │  │  │
│  │  │  │ RMSNorm     │───►│   Qwen3MLP      │───►│  Residual   │  │  │  │
│  │  │  │ (post_attn) │    │   (SwiGLU)      │    │  Add        │  │  │  │
│  │  │  └─────────────┘    └─────────────────┘    └─────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │           ▼                                                       │  │
│  │  ┌─────────────────┐                                              │  │
│  │  │  RMSNorm (final)│                                              │  │
│  │  └────────┬────────┘                                              │  │
│  └───────────┼───────────────────────────────────────────────────────┘  │
│              ▼                                                          │
│  ┌─────────────────┐                                                    │
│  │    lm_head      │  [hidden_size → vocab_size] (tied to embed_tokens) │
│  └────────┬────────┘                                                    │
│           ▼                                                             │
│        logits                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. `rope.py` - Rotary Position Embeddings

RoPE encodes position information by rotating query and key vectors, allowing the attention mechanism to be aware of relative positions between tokens.

**Key formula:**
```python
x_rotated = x * cos(position * freq) + rotate_half(x) * sin(position * freq)
```

**Components:**
- `compute_rope_frequencies()`: Computes inverse frequencies θ_i = θ^(-2i/d)
- `compute_rope_cos_sin()`: Precomputes cos/sin for given positions
- `rotate_half()`: Rotation operation [-x2, x1] used in RoPE
- `apply_rope()`: Applies rotation to a single tensor
- `apply_rope_to_qk()`: Applies rotation to both Q and K
- `RotaryEmbedding`: Class that caches cos/sin for efficiency

**Usage:**
```python
rope = RotaryEmbedding(head_dim=128, rope_theta=1000000.0)
q_rotated, k_rotated = rope(query, key, position_ids)
```

### 2. `layers.py` - Building Blocks

#### RMSNorm (Root Mean Square Normalization)

Simpler than LayerNorm - only normalizes by RMS, no mean subtraction:

```python
def forward(self, x):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + self.eps) * self.weight
```

#### Qwen3MLP (SwiGLU)

Gated Linear Unit with SiLU activation for better training dynamics:

```python
def forward(self, x):
    gate = self.gate_proj(x)      # [hidden → intermediate]
    up = self.up_proj(x)          # [hidden → intermediate]
    hidden = F.silu(gate) * up    # Element-wise gating
    return self.down_proj(hidden) # [intermediate → hidden]
```

### 3. `attention.py` - Attention Mechanisms

This module provides **three attention implementations**:

| Function | Phase | Input Shape | Purpose |
|----------|-------|-------------|---------|
| `flash_attention_prefill` | Prefill | `[total_tokens, heads, head_dim]` | Efficient packed sequences |
| `paged_attention_decode` | Decode | `[batch, heads, head_dim]` | Block-based KV cache access |
| `naive_attention` | Fallback | `[batch, heads, seq, head_dim]` | CPU testing, padded batches |

#### FlashAttention Prefill

Uses `cu_seqlens` (cumulative sequence lengths) to pack variable-length sequences without padding:

```
Sequences: [10 tokens] [8 tokens] [12 tokens]
Packed:    [----30 total tokens----]
cu_seqlens: [0, 10, 18, 30]
```

```python
output = flash_attn_varlen_func(
    q=query,                    # [total_tokens, num_heads, head_dim]
    k=key,                      # [total_tokens, num_kv_heads, head_dim]
    v=value,
    cu_seqlens_q=cu_seqlens,    # [batch_size + 1]
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=True,
)
```

#### PagedAttention Decode

Block-based KV cache access using PyTorch gather operations:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  KV Cache Pool: [num_blocks, block_size, num_kv_heads, head_dim]        │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐               │
│  │ B0  │ │ B1  │ │ B2  │ │ B3  │ │ B4  │ │ B5  │ │ B6  │ ...           │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘               │
└─────────────────────────────────────────────────────────────────────────┘
                              │
         block_tables: [[0,2], [1,5]]  (Seq0 uses B0,B2; Seq1 uses B1,B5)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. Gather: k_cache[block_tables] → per-sequence KV                     │
│  2. Reshape: flatten blocks into sequence dimension                     │
│  3. GQA expand: repeat KV heads to match query heads                    │
│  4. Attention: Q @ K^T → scores → softmax → @ V                         │
│  5. Mask: zero out positions beyond context_lens                        │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Qwen3Attention

Wraps both attention implementations and handles GQA:

```python
def forward(self, hidden_states, position_embeddings, is_prefill=True, ...):
    cos, sin = position_embeddings

    # Project Q, K, V
    q = self.q_proj(hidden_states)  # [hidden → num_heads * head_dim]
    k = self.k_proj(hidden_states)  # [hidden → num_kv_heads * head_dim]
    v = self.v_proj(hidden_states)

    # Apply RoPE
    q, k = apply_rope(q, k, cos, sin)

    # Route to appropriate attention
    if is_prefill:
        output = flash_attention_prefill(q, k, v, cu_seqlens, max_seqlen)
    else:
        output = paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens)

    return self.o_proj(output), (k, v)
```

### 4. `qwen3.py` - Full Model Assembly

#### Qwen3Config

Dataclass holding model hyperparameters (Qwen3-4B defaults):

```python
@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 40
    num_attention_heads: int = 20
    num_key_value_heads: int = 4      # GQA: 5:1 ratio
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
```

#### Qwen3DecoderLayer

Single transformer layer with pre-norm architecture:

```python
def forward(self, hidden_states, position_embeddings, ...):
    # Self-attention with residual
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    attn_output, kv = self.self_attn(hidden_states, position_embeddings, ...)
    hidden_states = residual + attn_output

    # MLP with residual
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, kv
```

#### Qwen3Model

Stacks embedding, N decoder layers, and final norm:

```python
def forward(self, input_ids, position_ids=None, is_prefill=True, ...):
    # 1. Embed tokens
    hidden_states = self.embed_tokens(input_ids)

    # 2. Compute RoPE
    cos, sin = compute_rope_cos_sin(position_ids, self.config.head_dim, ...)

    # 3. Process through layers
    new_kv_list = []
    for i, layer in enumerate(self.layers):
        hidden_states, kv = layer(hidden_states, (cos, sin), ...)
        new_kv_list.append(kv)

    # 4. Final norm
    hidden_states = self.norm(hidden_states)

    return hidden_states, new_kv_list
```

#### Qwen3ForCausalLM

Complete model with lm_head for next-token prediction:

```python
def forward(self, input_ids, ...):
    hidden_states, kv_list = self.model(input_ids, ...)
    logits = self.lm_head(hidden_states)
    return logits, kv_list

@classmethod
def from_pretrained(cls, model_path, device=None, dtype=None):
    config = Qwen3Config.from_pretrained(model_path)
    model = cls(config)
    load_qwen3_weights(model, model_path)
    return model
```

### 5. `loader.py` - Weight Loading

Loads weights from HuggingFace checkpoints (safetensors or pytorch_model.bin):

```python
def load_qwen3_weights(model, model_path):
    # Build weight name mapping
    weight_map = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.layers.{i}.input_layernorm.weight": "model.layers.{i}.input_layernorm.weight",
        "model.layers.{i}.self_attn.q_proj.weight": "model.layers.{i}.self_attn.q_proj.weight",
        # ... etc
    }

    # Load from safetensors
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt") as f:
            for hf_name in f.keys():
                tensor = f.get_tensor(hf_name)
                set_weight(model, weight_map[hf_name], tensor)
```

## Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: input_ids [batch, seq_len] or [total_tokens] (packed)               │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. Embedding: embed_tokens(input_ids)                                      │
│     [batch, seq, hidden_size] or [total_tokens, hidden_size]                │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. RoPE: compute_rope_cos_sin(position_ids)                                │
│     Returns (cos, sin) tensors for position encoding                        │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
    ┌────────────────────────────┴───────────────────────────┐
    │                                                        │
    ▼                                                        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  3. FOR EACH LAYER (×40 for Qwen3-4B):                                    │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  a. RMSNorm(input_layernorm)                                        │  │
│  │     hidden = hidden / sqrt(mean(hidden²) + eps) * weight            │  │
│  └───────────────────────────────┬─────────────────────────────────────┘  │
│                                  │                                        │
│  ┌───────────────────────────────▼─────────────────────────────────────┐  │
│  │  b. Qwen3Attention                                                  │  │
│  │     ┌──────────────────────────────────────────────────────────┐    │  │
│  │     │  Q = q_proj(hidden)   [hidden → num_heads * head_dim]    │    │  │
│  │     │  K = k_proj(hidden)   [hidden → num_kv_heads * head_dim] │    │  │
│  │     │  V = v_proj(hidden)   [hidden → num_kv_heads * head_dim] │    │  │
│  │     ├──────────────────────────────────────────────────────────┤    │  │
│  │     │  Q, K = apply_rope(Q, K, cos, sin)  # Position encoding  │    │  │
│  │     ├──────────────────────────────────────────────────────────┤    │  │
│  │     │  if is_prefill:                                          │    │  │
│  │     │      output = flash_attention_prefill(Q, K, V, cu_seqlens)│   │  │
│  │     │  else:                                                   │    │  │
│  │     │      output = paged_attention_decode(Q, k_cache, v_cache)│    │  │
│  │     ├──────────────────────────────────────────────────────────┤    │  │
│  │     │  output = o_proj(output)  [num_heads * head_dim → hidden]│    │  │
│  │     └──────────────────────────────────────────────────────────┘    │  │
│  │     Returns: (attn_output, (K, V))  # K,V for caching              │  │
│  └───────────────────────────────┬─────────────────────────────────────┘  │
│                                  │                                        │
│  ┌───────────────────────────────▼─────────────────────────────────────┐  │
│  │  c. Residual: hidden = hidden + attn_output                         │  │
│  └───────────────────────────────┬─────────────────────────────────────┘  │
│                                  │                                        │
│  ┌───────────────────────────────▼─────────────────────────────────────┐  │
│  │  d. RMSNorm(post_attention_layernorm)                               │  │
│  └───────────────────────────────┬─────────────────────────────────────┘  │
│                                  │                                        │
│  ┌───────────────────────────────▼─────────────────────────────────────┐  │
│  │  e. Qwen3MLP (SwiGLU)                                               │  │
│  │     gate = gate_proj(hidden)    # [hidden → intermediate]          │  │
│  │     up = up_proj(hidden)        # [hidden → intermediate]          │  │
│  │     hidden = down_proj(silu(gate) * up)  # [intermediate → hidden] │  │
│  └───────────────────────────────┬─────────────────────────────────────┘  │
│                                  │                                        │
│  ┌───────────────────────────────▼─────────────────────────────────────┐  │
│  │  f. Residual: hidden = hidden + mlp_output                          │  │
│  └───────────────────────────────┬─────────────────────────────────────┘  │
└──────────────────────────────────┼────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. Final RMSNorm                                                           │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. lm_head: Linear [hidden_size → vocab_size]                              │
│     Output: logits [batch, seq_len, vocab_size]                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Aspect | Choice | Reason |
|--------|--------|--------|
| **Attention** | GQA (5:1) | 5× KV cache savings (20 Q heads share 4 KV heads) |
| **Normalization** | RMSNorm | Simpler, no mean subtraction, fewer params |
| **Activation** | SwiGLU | Better training dynamics than ReLU/GELU |
| **Position** | RoPE θ=1e6 | Relative positions, supports long context |
| **Prefill** | FlashAttention | O(N) memory, fused kernels |
| **Decode** | PagedAttention | Block-based KV, no memory fragmentation |

## Module Dependencies

```
qwen3.py
├── imports from layers.py (RMSNorm, Qwen3MLP)
├── imports from attention.py (Qwen3Attention)
└── imports from rope.py (compute_rope_cos_sin, RotaryEmbedding)

attention.py
├── imports from rope.py (apply_rope_to_qk)
└── imports flash_attn (external, optional)

layers.py
└── standalone (no internal dependencies)

rope.py
└── standalone (no internal dependencies)

loader.py
├── imports from qwen3.py (Qwen3ForCausalLM - type hint only)
└── imports safetensors (external)
```

## Qwen3-4B Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocab_size` | 151,936 | Tokenizer vocabulary |
| `hidden_size` | 2,560 | Model width |
| `intermediate_size` | 6,912 | MLP expansion (~2.7×) |
| `num_hidden_layers` | 40 | Transformer depth |
| `num_attention_heads` | 20 | Query heads |
| `num_key_value_heads` | 4 | KV heads (GQA 5:1) |
| `head_dim` | 128 | Per-head dimension |
| `max_position_embeddings` | 32,768 | Max context length |
| `rope_theta` | 1,000,000 | RoPE base frequency |

**Total Parameters:** ~4 billion

## Usage Example

```python
from llm_engine.model import Qwen3ForCausalLM, Qwen3Config

# Create model with custom config
config = Qwen3Config(
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=4,
)
model = Qwen3ForCausalLM(config)

# Or load from HuggingFace checkpoint
model = Qwen3ForCausalLM.from_pretrained("/path/to/Qwen3-4B")

# Forward pass (prefill)
input_ids = torch.randint(0, config.vocab_size, (2, 10))
logits, kv_list = model(input_ids, is_prefill=True)

# Forward pass (decode with KV cache)
new_token = torch.randint(0, config.vocab_size, (2, 1))
logits, kv_list = model(
    new_token,
    is_prefill=False,
    kv_caches=kv_caches,
    block_tables=block_tables,
    context_lens=context_lens,
)
```
