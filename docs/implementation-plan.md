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
| KV Cache | 4a-c | Pending |
| Scheduling | 5-6 | Pending |
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

---

## Phase 3: KV Cache (Next)

### Step 4a: KVCache Tensor Allocation
**Status:** Pending

Create GPU tensors for storing K/V across all layers:

```python
class KVCache:
    """
    Block-based KV cache storage.

    Shape per layer: [num_blocks, block_size, num_kv_heads, head_dim]
    """
    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype):
        self.k_cache = [torch.zeros(...) for _ in range(num_layers)]
        self.v_cache = [torch.zeros(...) for _ in range(num_layers)]
```

### Step 4b: BlockManager
**Status:** Pending

Manage block allocation and deallocation:

```python
class BlockManager:
    """
    Manages physical block allocation for sequences.

    Features:
    - Allocate blocks for new sequences
    - Free blocks when sequences complete
    - Copy-on-write for beam search / prefix sharing
    """
    def allocate(self, seq_id: int, num_blocks: int) -> List[int]
    def free(self, seq_id: int) -> None
    def fork(self, src_seq_id: int, dst_seq_id: int) -> None  # CoW
```

### Step 4c: Integrate with Attention
**Status:** Pending

- Update `Qwen3Attention.forward()` to read/write KV cache during decode
- Pass `KVCache` and `BlockTable` to attention layers
- Update `paged_attention_decode()` to use actual block indices

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

Split long prefills into chunks to interleave with decodes:

- Configurable chunk size (e.g., 512 tokens)
- Allows decode tokens to proceed during long prefills
- Improves latency for concurrent requests

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
│   ├── config.py              # ModelConfig, SamplingParams
│   ├── data_structures/
│   │   ├── sequence.py        # Sequence, SequenceGroup
│   │   └── block.py           # Block, BlockTable
│   ├── model/
│   │   ├── rope.py            # Rotary position embeddings
│   │   ├── layers.py          # RMSNorm, MLP
│   │   ├── attention.py       # Flash/Paged attention
│   │   ├── qwen3.py           # Full Qwen3 model
│   │   ├── loader.py          # Weight loading
│   │   └── executor.py        # Model execution
│   ├── memory/                 # Step 4: KV Cache
│   │   ├── kv_cache.py
│   │   └── block_manager.py
│   ├── scheduler/              # Step 5-6
│   │   └── scheduler.py
│   ├── sampling/
│   │   └── sampler.py
│   └── engine/                 # Steps 7-11
│       ├── llm_engine.py
│       ├── async_engine.py
│       └── output.py
├── tests/
├── scripts/
│   └── compare_models.py      # HF comparison script
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
