# Qwen3 Model Architecture

This document explains the custom Qwen3 model implementation and how the different modules under `llm_engine/model/` are combined.

## Model Directory Structure

```
llm_engine/model/
├── __init__.py           # Exports all public APIs
├── rope.py               # Rotary Position Embeddings
├── layers.py             # RMSNorm, SwiGLU MLP
├── attention.py          # Unified attention (prefill/chunked/decode)
├── attention_metadata.py # AttentionMetadata for unified forward
├── qwen3.py              # Full model: Config, DecoderLayer, Model, ForCausalLM
├── loader.py             # Weight loading from HuggingFace
└── executor.py           # Forward pass execution wrapper
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
│  │  │              Qwen3DecoderLayer × 36                         │  │  │
│  │  │  ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐  │  │  │
│  │  │  │ RMSNorm     │───►│ Qwen3Attention  │───►│  Residual   │  │  │  │
│  │  │  │ (input)     │    │ (Unified GQA)   │    │  Add        │  │  │  │
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

## Class Hierarchy

```
Qwen3ForCausalLM
├── model: Qwen3Model
│   ├── embed_tokens: nn.Embedding
│   ├── layers: nn.ModuleList[Qwen3DecoderLayer]
│   │   └── [0..35]: Qwen3DecoderLayer
│   │       ├── input_layernorm: RMSNorm
│   │       ├── self_attn: Qwen3Attention
│   │       │   ├── q_proj, k_proj, v_proj, o_proj: nn.Linear
│   │       │   ├── q_norm, k_norm: RMSNorm
│   │       │   └── forward() → unified attention
│   │       ├── post_attention_layernorm: RMSNorm
│   │       └── mlp: Qwen3MLP
│   │           └── gate_proj, up_proj, down_proj: nn.Linear
│   ├── norm: RMSNorm (final)
│   └── rotary_emb: RotaryEmbedding
└── lm_head: nn.Linear (tied to embed_tokens)
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

### 3. `attention_metadata.py` - Unified Attention Metadata

**AttentionMetadata** describes batch composition for the unified forward pass:

```python
@dataclass
class AttentionMetadata:
    # Batch composition
    num_prefill_tokens: int      # Total prefill tokens in batch
    num_decode_tokens: int       # Total decode tokens (= num_decode_seqs)
    num_prefill_seqs: int
    num_decode_seqs: int

    # For prefill (FlashAttention varlen)
    prefill_seq_lens: Tensor         # [num_prefill_seqs]
    prefill_context_lens: Tensor     # [num_prefill_seqs] - 0 for pure prefill
    prefill_cu_seqlens_q: Tensor     # [num_prefill_seqs + 1]
    prefill_cu_seqlens_kv: Tensor    # [num_prefill_seqs + 1]
    max_prefill_seq_len: int
    prefill_block_tables: Tensor     # For chunked prefill

    # For decode (PagedAttention)
    decode_seq_lens: Tensor          # [num_decode_seqs]
    decode_block_tables: Tensor      # [num_decode_seqs, max_blocks]

    # KV cache write positions
    slot_mapping: Tensor             # [num_tokens]
```

**Token Layout:**
```
[<-- prefill tokens -->|<-- decode tokens -->]
```

**Helper Functions:**
- `create_prefill_metadata()`: Create metadata for pure prefill
- `create_decode_metadata()`: Create metadata for decode-only
- `AttentionMetadataBuilder`: Build metadata from scheduled sequences

### 4. `attention.py` - Unified Attention

This module provides **unified attention** that handles all modes in a single forward pass:

| Mode | Condition | Kernel |
|------|-----------|--------|
| Pure prefill | `num_prefill > 0`, `context_lens = 0` | FlashAttention varlen |
| Chunked prefill | `num_prefill > 0`, `context_lens > 0` | SDPA + gather cached KV |
| Decode | `num_decode > 0` | PagedAttention |
| Mixed | Both prefill and decode | Split processing |

#### Qwen3Attention - Unified Forward

```python
def forward(
    self,
    hidden_states: Tensor,           # [total_tokens, hidden_size]
    position_embeddings: Tuple,      # (cos, sin)
    kv_cache: Optional[Tuple],       # (k_cache, v_cache) for this layer
    attn_metadata: Optional[AttentionMetadata],
) -> Tensor:
    # If no metadata, use legacy path (for HF comparison)
    if attn_metadata is None:
        return self._forward_legacy_prefill(hidden_states, cos, sin)

    # Unified forward
    return self._forward_unified(hidden_states, cos, sin, kv_cache, attn_metadata)
```

#### Unified Forward Flow

```python
def _forward_unified(self, hidden_states, cos, sin, kv_cache, attn_metadata):
    num_prefill = attn_metadata.num_prefill_tokens
    num_decode = attn_metadata.num_decode_tokens

    # 1. Project all tokens together
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # 2. Apply QK-norm and RoPE
    q, k = self.q_norm(q), self.k_norm(k)
    q, k = apply_rope(q, k, cos, sin)

    # 3. Write new KV to cache
    if kv_cache is not None and slot_mapping is not None:
        write_to_kv_cache(k, v, kv_cache, slot_mapping)

    # 4. Split and process
    output = torch.empty(...)

    if num_prefill > 0:
        if has_chunked_prefill:
            output[:num_prefill] = _attention_chunked_prefill(...)
        else:
            output[:num_prefill] = _attention_pure_prefill(...)  # FlashAttn

    if num_decode > 0:
        output[num_prefill:] = _attention_decode(...)  # PagedAttn

    # 5. Output projection
    return self.o_proj(output)
```

#### FlashAttention Prefill

Uses `cu_seqlens` (cumulative sequence lengths) to pack variable-length sequences:

```
Sequences: [10 tokens] [8 tokens] [12 tokens]
Packed:    [----30 total tokens----]
cu_seqlens: [0, 10, 18, 30]
```

#### PagedAttention Decode

Block-based KV cache access:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  KV Cache Pool: [num_blocks, block_size, num_kv_heads, head_dim]        │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                       │
│  │ B0  │ │ B1  │ │ B2  │ │ B3  │ │ B4  │ │ B5  │ ...                   │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
         block_tables: [[0,2], [1,5]]  (Seq0 uses B0,B2; Seq1 uses B1,B5)
                              │
                              ▼
         1. Gather KV from block_tables
         2. Apply length mask (context_lens)
         3. Compute attention: Q @ K^T → softmax → @ V
```

#### Chunked Prefill Attention

For sequences with cached KV from previous chunks:

```python
def _attention_chunked_prefill(q, k_new, v_new, kv_cache, attn_metadata):
    for each sequence:
        # 1. Gather cached KV from paged cache
        k_cached = gather_from_cache(block_tables[i], context_lens[i])
        v_cached = gather_from_cache(...)

        # 2. Concat cached + new
        k_full = concat(k_cached, k_new[seq_slice])
        v_full = concat(v_cached, v_new[seq_slice])

        # 3. SDPA with custom causal mask
        # Q positions: [ctx_len, ctx_len+1, ..., ctx_len+query_len-1]
        # K positions: [0, 1, ..., total_len-1]
        output[seq_slice] = SDPA(q[seq_slice], k_full, v_full, mask)
```

### 5. `qwen3.py` - Full Model Assembly

#### Qwen3Config

Dataclass holding model hyperparameters (Qwen3-4B defaults):

```python
@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 36
    num_attention_heads: int = 20
    num_key_value_heads: int = 4      # GQA: 5:1 ratio
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
```

#### Method Signatures

| Class | Method | Input | Output | Use Case |
|-------|--------|-------|--------|----------|
| `Qwen3ForCausalLM` | `forward()` | `[total_tokens]` | `[total_tokens, vocab]` | Production |
| `Qwen3ForCausalLM` | `forward_legacy()` | `[batch, seq]` | `[batch, seq, vocab]` | HF comparison |
| `Qwen3Model` | `forward()` | `[total_tokens]` | `[total_tokens, hidden]` | Production |
| `Qwen3Model` | `forward_legacy()` | `[batch, seq]` | `[batch, seq, hidden]` | HF comparison |

#### Qwen3ForCausalLM.forward() - Production API

```python
def forward(
    input_ids: Tensor,           # [total_tokens]
    positions: Tensor,           # [total_tokens]
    kv_caches: List[Tuple],      # [(k_cache, v_cache)] per layer
    attn_metadata: AttentionMetadata,
) -> Tensor:                     # [total_tokens, vocab_size]
```

#### Qwen3ForCausalLM.forward_legacy() - HF Comparison

```python
def forward_legacy(
    input_ids: Tensor,           # [batch, seq_len]
    position_ids: Optional[Tensor] = None,
) -> Tensor:                     # [batch, seq_len, vocab_size]
```

### 6. `loader.py` - Weight Loading

Loads weights from HuggingFace checkpoints (safetensors or pytorch_model.bin):

```python
def load_qwen3_weights(model, model_path):
    # Build weight name mapping (HF → our model)
    weight_map = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.layers.{i}.self_attn.q_proj.weight": "...",
        "model.layers.{i}.self_attn.q_norm.weight": "...",  # Qwen3 Q/K norms
        # ... etc
    }

    # Load from safetensors
    for sf_path in safetensor_files:
        with safe_open(sf_path, framework="pt") as f:
            for hf_name in f.keys():
                set_weight(model, weight_map[hf_name], tensor)
```

## Complete Data Flow

```
forward() - Production Mode (Unified Attention)
═══════════════════════════════════════════════════════════════════

input_ids [total_tokens]    positions [total_tokens]
    │                           │
    ▼                           ▼
┌─────────────────┐    ┌─────────────────┐
│  embed_tokens   │    │ compute_rope    │
└────────┬────────┘    └────────┬────────┘
         │                      │
         ▼                      ▼
hidden_states              (cos, sin)
[total_tokens, hidden]     [total_tokens, head_dim]
         │                      │
         └──────────┬───────────┘
                    │
    ┌───────────────┼───────────────┐
    │               ▼               │
    │  ┌─────────────────────────┐  │
    │  │  FOR EACH LAYER (×36)   │  │
    │  │                         │  │
    │  │  1. input_layernorm     │◄─┼── kv_caches[i]
    │  │  2. self_attn (unified) │◄─┼── attn_metadata
    │  │     - write to KV cache │  │
    │  │     - split prefill/dec │  │
    │  │     - FlashAttn/PagedAt │  │
    │  │  3. residual add        │  │
    │  │  4. post_attn_layernorm │  │
    │  │  5. MLP (SwiGLU)        │  │
    │  │  6. residual add        │  │
    │  └─────────────────────────┘  │
    └───────────────┼───────────────┘
                    │
                    ▼
           ┌────────────────┐
           │  final norm    │
           └───────┬────────┘
                   │
                   ▼
           ┌────────────────┐
           │    lm_head     │
           └───────┬────────┘
                   │
                   ▼
            logits [total_tokens, vocab_size]
```

## Key Design Decisions

| Aspect | Choice | Reason |
|--------|--------|--------|
| **Attention API** | Unified (Option C) | Single forward handles prefill/chunked/decode/mixed |
| **Attention** | GQA (5:1) | 5× KV cache savings (20 Q heads share 4 KV heads) |
| **Normalization** | RMSNorm | Simpler, no mean subtraction, fewer params |
| **Activation** | SwiGLU | Better training dynamics than ReLU/GELU |
| **Position** | RoPE θ=1e6 | Relative positions, supports long context |
| **Prefill** | FlashAttention | O(N) memory, fused kernels |
| **Decode** | PagedAttention | Block-based KV, no memory fragmentation |
| **Chunked Prefill** | SDPA + gather | Concat cached + new KV, custom causal mask |

## Module Dependencies

```
qwen3.py
├── imports from layers.py (RMSNorm, Qwen3MLP)
├── imports from attention.py (Qwen3Attention)
├── imports from attention_metadata.py (AttentionMetadata) [TYPE_CHECKING]
└── imports from rope.py (compute_rope_cos_sin, RotaryEmbedding)

attention.py
├── imports from rope.py (apply_rope_to_qk)
├── imports from layers.py (RMSNorm)
├── imports from attention_metadata.py (AttentionMetadata) [TYPE_CHECKING]
└── imports flash_attn (external, optional)

attention_metadata.py
└── imports from data_structures.sequence (Sequence) [TYPE_CHECKING]

layers.py
└── standalone (no internal dependencies)

rope.py
└── standalone (no internal dependencies)

loader.py
├── imports from qwen3.py (Qwen3ForCausalLM) [TYPE_CHECKING]
└── imports safetensors (external)
```

## Qwen3-4B Dimensions

| Parameter | Value | Notes |
|-----------|-------|-------|
| `vocab_size` | 151,936 | Tokenizer vocabulary |
| `hidden_size` | 2,560 | Model width |
| `intermediate_size` | 6,912 | MLP expansion (~2.7×) |
| `num_hidden_layers` | 36 | Transformer depth |
| `num_attention_heads` | 20 | Query heads |
| `num_key_value_heads` | 4 | KV heads (GQA 5:1) |
| `head_dim` | 128 | Per-head dimension |
| `max_position_embeddings` | 32,768 | Max context length |
| `rope_theta` | 1,000,000 | RoPE base frequency |

**Total Parameters:** ~4 billion

## Usage Examples

### Production Forward (Unified API)

```python
from llm_engine.model import (
    Qwen3ForCausalLM,
    AttentionMetadata,
    create_prefill_metadata,
)

# Load model
model = Qwen3ForCausalLM.from_pretrained("/path/to/Qwen3-4B")

# Pure prefill (no KV cache)
input_ids = torch.tensor([1, 2, 3, 4, 5])  # [total_tokens]
positions = torch.tensor([0, 1, 2, 3, 4])
attn_metadata = create_prefill_metadata(seq_lens=[5], device="cuda")

logits = model(input_ids, positions, kv_caches=None, attn_metadata=attn_metadata)
# logits: [5, vocab_size]
```

### Decode with KV Cache

```python
from llm_engine.model import create_decode_metadata

# Allocate KV cache (Step 4)
kv_caches = allocate_kv_cache(num_layers=36, num_blocks=100, ...)

# Decode step
input_ids = torch.tensor([100, 200])  # 2 sequences, 1 token each
positions = torch.tensor([10, 15])     # Current positions
attn_metadata = create_decode_metadata(
    context_lens=[10, 15],
    block_tables=[[0, 1], [2, 3, 4]],
    slot_mapping=[80, 121],  # Where to write new KV
    device="cuda",
)

logits = model(input_ids, positions, kv_caches, attn_metadata)
# logits: [2, vocab_size]
```

### Mixed Batch (Prefill + Decode)

```python
# Token layout: [prefill_tokens | decode_tokens]
input_ids = torch.tensor([1, 2, 3, 4, 5, 100, 200])
#                        [--- prefill ---] [decode]
positions = torch.tensor([0, 1, 2, 3, 4, 10, 15])

attn_metadata = AttentionMetadata(
    num_prefill_tokens=5,
    num_decode_tokens=2,
    # ... prefill and decode metadata
)

logits = model(input_ids, positions, kv_caches, attn_metadata)
# logits: [7, vocab_size]
```

### Legacy Forward (HF Comparison)

```python
# For comparing against HuggingFace
input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # [batch=1, seq=5]

logits = model.forward_legacy(input_ids)
# logits: [1, 5, vocab_size]
```
