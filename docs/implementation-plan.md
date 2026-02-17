# LLM Engine Implementation Plan

A minimal LLM serving engine built from scratch, implementing core vLLM concepts for educational purposes.

**Model:** Qwen3-4B
**Environment:** `self_evolution` conda
**Repository:** `/scratch/m000069/miaolu/llm-engine`

---

## Progress Overview

| Phase | Steps | Status |
|-------|-------|--------|
| Foundation | 1-3 | Completed |
| Custom Model | 3.5a-f | Completed & Validated |
| Unified Attention | 3.6 | Completed (Option C) |
| KV Cache | 4a-b | Completed |
| Scheduling | 5-6 | Next |
| API & Features | 7-10 | Pending |
| Production | 11-12 | Pending |

---

## Phase 1: Foundation (Completed)

### Step 1: Project Setup
**Status:** Completed

- Created directory structure (`llm_engine/`, `tests/`, `scripts/`, `docs/`)
- Set up `pyproject.toml` with dependencies
- Configured pytest for testing

### Step 2: Data Structures
**Status:** Completed

Core abstractions for sequence and memory management:

| File | Classes | Purpose |
|------|---------|---------|
| `data_structures/sequence.py` | `Sequence`, `SequenceGroup` | Track token sequences and their states |
| `data_structures/block.py` | `Block`, `BlockTable` | KV cache block references |

### Step 3: Model Integration (Basic)
**Status:** Completed

- `model/loader.py`: Load HuggingFace models and extract architecture info
- `model/executor.py`: Basic model execution wrapper
- `config.py`: `ModelConfig` dataclass

---

## Phase 2: Custom Model Implementation (Completed)

### Step 3.5a: Rotary Position Embeddings
**Status:** Completed

**File:** `llm_engine/model/rope.py`

- `compute_rope_cos_sin()`: Precompute cos/sin for positions
- `apply_rope_to_qk()`: Apply rotation to Q and K tensors
- Supports batched and packed (variable-length) sequences

### Step 3.5b: Model Building Blocks
**Status:** Completed

**File:** `llm_engine/model/layers.py`

| Class | Description |
|-------|-------------|
| `RMSNorm` | Root Mean Square normalization (eps=1e-6) |
| `Qwen3MLP` | SwiGLU MLP (gate_proj, up_proj, down_proj) |

### Step 3.5c: Attention Mechanisms
**Status:** Completed

**File:** `llm_engine/model/attention.py`

| Function/Class | Description |
|----------------|-------------|
| `flash_attention_prefill()` | FlashAttention varlen for packed prefill |
| `paged_attention_decode()` | Block-based KV cache access for decode |
| `Qwen3Attention` | Full attention module with 3 forward paths |

**Forward Paths:**
1. Packed prefill (FlashAttention varlen)
2. Padded prefill (SDPA)
3. Decode (PagedAttention)

### Step 3.5d: Full Model Definition
**Status:** Completed

**File:** `llm_engine/model/qwen3.py`

| Class | Description |
|-------|-------------|
| `Qwen3Config` | Model configuration (matches HF) |
| `Qwen3DecoderLayer` | Single transformer layer |
| `Qwen3Model` | Stack of decoder layers |
| `Qwen3ForCausalLM` | Full model with lm_head |

**Qwen3-4B Architecture:**
- 36 layers, hidden_size=2560
- 20 attention heads, 4 KV heads (GQA 5:1)
- head_dim=128, rope_theta=1000000

### Step 3.5e: Weight Loading
**Status:** Completed

**File:** `llm_engine/model/loader.py`

- `load_qwen3_weights()`: Load from HuggingFace checkpoint
- Supports safetensors and pytorch_model.bin formats
- Weight name mapping between HF and custom model

### Step 3.5f: Validation Against HuggingFace
**Status:** Completed

**Script:** `scripts/compare_models.py`

**Fixes Applied:**
1. Initialize Q/K norms as `RMSNorm(head_dim)` (was `None`)
2. Use custom RMSNorm with eps=1e-6 (not nn.RMSNorm eps=1e-8)
3. Replace naive_attention with SDPA + correct GQA expansion
4. Cast RoPE cos/sin to model dtype for bf16/fp16

**Validation Results:**
```
float32: max_diff=2.3e-05, top-1 match ✓, top-5 match ✓
bf16:    bit-exact match ✓
fp16:    bit-exact match ✓
```

See: `docs/qwen3-hf-comparison-fixes.md` for detailed analysis.

### Step 3.6: Unified Attention Architecture (Option C)
**Status:** Completed

Refactored attention to use vLLM-style unified forward pass with metadata-driven routing.

**New Files:**

| File | Purpose |
|------|---------|
| `model/attention_metadata.py` | `AttentionMetadata`, `AttentionMetadataBuilder` |

**Key Changes:**

1. **AttentionMetadata**: Describes batch composition (prefill vs decode tokens)
2. **Unified forward**: Single entry point handles all modes
3. **Split processing**: Prefill and decode tokens processed separately
4. **KV cache integration**: `write_to_kv_cache()`, `gather_from_kv_cache()`

**Token Layout:**
```
[<-- prefill tokens -->|<-- decode tokens -->]
```

**Forward Flow:**
```python
def forward(hidden_states, position_embeddings, kv_cache, attn_metadata):
    # 1. Project all tokens together
    q, k, v = project_qkv(hidden_states)

    # 2. Write new KV to cache
    if kv_cache and slot_mapping:
        write_to_kv_cache(k, v, kv_cache, slot_mapping)

    # 3. Split and process
    if num_prefill > 0:
        output[:num_prefill] = prefill_attention(q[:num_prefill], ...)
    if num_decode > 0:
        output[num_prefill:] = decode_attention(q[num_prefill:], ...)

    return o_proj(output)
```

**Supports:**
- Pure prefill (FlashAttention varlen)
- Chunked prefill (SDPA with cached + new KV)
- Decode (PagedAttention)
- Mixed batches (prefill + decode in same forward)

---

## Phase 3: KV Cache (Completed)

### Step 4a: KVCache Tensor Allocation
**Status:** Completed

**File:** `llm_engine/memory/kv_cache.py`

GPU tensor allocation for K/V across all layers:

| Class/Function | Description |
|----------------|-------------|
| `KVCacheConfig` | Configuration dataclass |
| `KVCache` | Block-based cache storage |
| `create_kv_cache()` | Convenience constructor |
| `compute_slot_mapping()` | Compute write positions |

**Cache Layout:**
```
k_cache[layer]: [num_blocks, block_size, num_kv_heads, head_dim]
v_cache[layer]: [num_blocks, block_size, num_kv_heads, head_dim]
```

**Key Methods:**
- `get_layer_cache(layer_idx)` → (k_cache, v_cache)
- `get_all_layer_caches()` → List of (k, v) per layer
- `copy_blocks(src_to_dst)` → Copy-on-write support
- `clear_blocks(block_ids)` → Zero specific blocks

### Step 4b: BlockManager
**Status:** Completed

**File:** `llm_engine/memory/block_manager.py`

Manages physical block allocation for sequences:

| Class/Function | Description |
|----------------|-------------|
| `BlockManagerConfig` | Configuration dataclass |
| `BlockManager` | Block allocation manager |
| `create_block_manager()` | Convenience constructor |

**Key Methods:**
- `allocate_sequence(seq_id, num_tokens)` → Allocate blocks for prefill
- `allocate_token(seq_id)` → Allocate for decode (1 token)
- `free_sequence(seq_id)` → Free all blocks
- `fork_sequence(src, dst)` → Copy-on-write fork
- `copy_on_write(seq_id, block_idx)` → Duplicate shared block
- `get_slot_mapping(seq_id, context_len, num_new_tokens)` → Cache write positions
- `get_block_table(seq_id)` → Physical block IDs
- `can_allocate(num_blocks)` → Check memory availability

---

## Phase 4: Scheduling

### Step 5: Basic Scheduler
**Status:** Pending

FCFS (First-Come-First-Served) scheduling:

```python
class Scheduler:
    def __init__(self, block_manager: BlockManager):
        self.waiting: List[SequenceGroup] = []
        self.running: List[SequenceGroup] = []

    def schedule(self) -> SchedulerOutput:
        """Decide which sequences to run in the next step."""
```

### Step 6: Token Packing
**Status:** Pending

Pack multiple sequences into a single batch for efficient prefill:

```python
def pack_sequences(sequences: List[Sequence]) -> PackedBatch:
    """
    Pack variable-length sequences for FlashAttention varlen.

    Returns:
        input_ids: [total_tokens]
        cu_seqlens: [num_seqs + 1]
        position_ids: [total_tokens]
    """
```

---

## Phase 5: API & Features

### Step 7: Offline Batch API
**Status:** Pending

Simple batch inference API:

```python
class LLMEngine:
    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[RequestOutput]:
        """Generate completions for a batch of prompts."""
```

### Step 8: Prefix Caching
**Status:** Pending

Cache and reuse KV for common prefixes:

- Hash-based prefix matching
- Shared block references across sequences
- LRU eviction for cached blocks

### Step 9: Chunked Prefill
**Status:** Pending

Split long prefills into chunks to interleave with decodes.

**Note:** The unified attention (Step 3.6) already supports chunked prefill. This step only requires:

- Scheduler changes to split large prefills into chunks
- Build `AttentionMetadata` with `prefill_context_lens > 0`
- Configurable `max_num_batched_tokens` (default: 8192)

The attention layer handles chunked prefill via `_attention_chunked_prefill()`:
1. Gather cached KV from paged cache
2. Concatenate with new chunk KV
3. Apply causal attention with custom mask

### Step 10: Preemption
**Status:** Pending

Handle memory pressure by preempting sequences:

- Swap KV cache to CPU when GPU memory exhausted
- Resume preempted sequences when memory available
- Priority-based preemption policy

---

## Phase 6: Production

### Step 11: Async Engine
**Status:** Pending

Asynchronous request handling:

```python
class AsyncLLMEngine:
    async def add_request(self, request_id: str, prompt: str) -> None
    async def get_output(self, request_id: str) -> RequestOutput
    async def run_engine_loop(self) -> None
```

### Step 12: Testing & Documentation
**Status:** Pending

- Comprehensive test coverage for all components
- Performance benchmarks
- API documentation
- Example scripts

---

## File Structure

```
llm-engine/
├── llm_engine/
│   ├── __init__.py
│   ├── config.py              # ModelConfig, SamplingParams ✓
│   ├── data_structures/
│   │   ├── sequence.py        # Sequence, SequenceGroup ✓
│   │   └── block.py           # Block, BlockTable ✓
│   ├── model/
│   │   ├── rope.py            # Rotary position embeddings ✓
│   │   ├── layers.py          # RMSNorm, MLP ✓
│   │   ├── attention.py       # Unified attention (prefill/decode) ✓
│   │   ├── attention_metadata.py  # AttentionMetadata ✓
│   │   ├── qwen3.py           # Full Qwen3 model ✓
│   │   ├── loader.py          # Weight loading ✓
│   │   └── executor.py        # Model execution ✓
│   ├── memory/                 # Step 4: KV Cache ✓
│   │   ├── kv_cache.py        # KVCache, KVCacheConfig ✓
│   │   └── block_manager.py   # BlockManager ✓
│   ├── scheduler/              # Step 5-6 (pending)
│   │   └── scheduler.py
│   ├── sampling/
│   │   └── sampler.py         # Sampler ✓
│   └── engine/                 # Steps 7-11 (pending)
│       ├── llm_engine.py
│       ├── async_engine.py
│       └── output.py          # RequestOutput ✓
├── tests/
├── scripts/
│   ├── compare_models.py      # HF comparison ✓
│   ├── test_unified_prefill.py  # Unified attention test ✓
│   └── test_kv_cache.py       # KVCache/BlockManager test ✓
└── docs/
    ├── implementation-plan.md
    ├── qwen3-architecture.md
    └── qwen3-hf-comparison-fixes.md
```

---

## References

- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [PagedAttention Blog](https://blog.vllm.ai/2023/06/20/vllm.html)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-4B)
