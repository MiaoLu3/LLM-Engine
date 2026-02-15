"""
LLM Engine - A minimal LLM serving engine implementing vLLM concepts from scratch.

This package provides a complete, educational implementation of modern LLM serving
techniques including:
- Continuous batching
- Token packing
- PagedAttention with block-based KV cache
- Prefix caching
- Chunked prefill
- Preemption and eviction
- Async engine with streaming

Target model: Qwen3 family
"""

__version__ = "0.1.0"

from llm_engine.config import EngineConfig, SamplingParams

__all__ = [
    "EngineConfig",
    "SamplingParams",
]
