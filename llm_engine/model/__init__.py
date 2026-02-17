"""Model loading and execution."""

from llm_engine.model.loader import (
    ModelLoader,
    compute_kv_cache_size_per_token,
    compute_available_kv_blocks,
    load_qwen3_weights,
)
from llm_engine.model.executor import ModelExecutor, PackedInput, ModelOutput
from llm_engine.model.rope import (
    compute_rope_frequencies,
    compute_rope_cos_sin,
    rotate_half,
    apply_rope,
    apply_rope_to_qk,
    RotaryEmbedding,
)
from llm_engine.model.layers import (
    RMSNorm,
    Qwen3MLP,
    Qwen3DecoderLayer,
)
from llm_engine.model.attention import (
    flash_attention_prefill,
    paged_attention_decode,
    naive_attention,
    write_to_kv_cache,
    gather_from_kv_cache,
    Qwen3Attention,
)
from llm_engine.model.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilder,
    create_empty_metadata,
    create_prefill_metadata,
    create_decode_metadata,
)
from llm_engine.model.qwen3 import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3ForCausalLM,
)

__all__ = [
    # Loader
    "ModelLoader",
    "compute_kv_cache_size_per_token",
    "compute_available_kv_blocks",
    "load_qwen3_weights",
    # Executor
    "ModelExecutor",
    "PackedInput",
    "ModelOutput",
    # RoPE
    "compute_rope_frequencies",
    "compute_rope_cos_sin",
    "rotate_half",
    "apply_rope",
    "apply_rope_to_qk",
    "RotaryEmbedding",
    # Layers
    "RMSNorm",
    "Qwen3MLP",
    "Qwen3DecoderLayer",
    # Attention
    "flash_attention_prefill",
    "paged_attention_decode",
    "naive_attention",
    "write_to_kv_cache",
    "gather_from_kv_cache",
    "Qwen3Attention",
    # Attention Metadata
    "AttentionMetadata",
    "AttentionMetadataBuilder",
    "create_empty_metadata",
    "create_prefill_metadata",
    "create_decode_metadata",
    # Qwen3 Model
    "Qwen3Config",
    "Qwen3Model",
    "Qwen3ForCausalLM",
]
