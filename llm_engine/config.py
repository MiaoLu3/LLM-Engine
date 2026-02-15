"""
Engine configuration dataclasses.

These mirror the key configuration options from vLLM:
- Model configuration
- Scheduler configuration
- Memory (KV cache) configuration
- Sampling parameters
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Configuration for the model to serve."""

    model_name_or_path: str
    """HuggingFace model name or local path (e.g., 'Qwen/Qwen2.5-7B-Instruct')."""

    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    """Data type for model weights."""

    trust_remote_code: bool = True
    """Whether to trust remote code from HuggingFace."""

    max_model_len: Optional[int] = None
    """Maximum context length. If None, uses model's default."""

    # Derived attributes (set after model loading)
    num_layers: int = field(default=0, init=False)
    num_heads: int = field(default=0, init=False)
    num_kv_heads: int = field(default=0, init=False)
    head_dim: int = field(default=0, init=False)
    hidden_size: int = field(default=0, init=False)
    vocab_size: int = field(default=0, init=False)


@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""

    max_num_seqs: int = 256
    """Maximum number of sequences that can be processed in a single iteration.
    This is a hard cap on concurrent sequences."""

    max_num_batched_tokens: int = 8192
    """Maximum number of tokens processed in a single forward pass.
    Includes both prefill tokens and decode tokens."""

    enable_chunked_prefill: bool = False
    """Whether to split long prompts into chunks.
    When enabled, long prefills are interleaved with decodes."""

    max_prefill_tokens: Optional[int] = None
    """Maximum tokens per prefill chunk. If None, uses max_num_batched_tokens."""


@dataclass
class CacheConfig:
    """Configuration for KV cache management."""

    block_size: int = 16
    """Number of tokens per KV cache block.
    Smaller = less memory waste, but more block table overhead."""

    gpu_memory_utilization: float = 0.9
    """Fraction of GPU memory to use for KV cache.
    The rest is reserved for model weights and activations."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching (APC).
    When enabled, shared prefixes across sequences reuse KV cache blocks."""

    swap_space_gb: float = 0.0
    """CPU swap space in GB for preempted sequences.
    If 0, use recompute strategy instead of swap."""


@dataclass
class EngineConfig:
    """Complete engine configuration combining all sub-configs."""

    model: ModelConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    device: str = "cuda"
    """Device to run on (cuda or cpu)."""

    seed: int = 42
    """Random seed for reproducibility."""

    @classmethod
    def from_model_name(
        cls,
        model_name_or_path: str,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool = False,
        **kwargs,
    ) -> "EngineConfig":
        """Convenience constructor from model name with common options."""
        return cls(
            model=ModelConfig(
                model_name_or_path=model_name_or_path,
                **{k: v for k, v in kwargs.items() if k in ModelConfig.__dataclass_fields__}
            ),
            scheduler=SchedulerConfig(
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=enable_chunked_prefill,
            ),
            cache=CacheConfig(
                gpu_memory_utilization=gpu_memory_utilization,
                enable_prefix_caching=enable_prefix_caching,
            ),
        )


@dataclass
class SamplingParams:
    """Parameters for token sampling during generation."""

    n: int = 1
    """Number of output sequences to generate per prompt."""

    max_tokens: int = 256
    """Maximum number of tokens to generate per sequence."""

    temperature: float = 1.0
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 1.0
    """Nucleus sampling threshold. 1.0 = disabled."""

    top_k: int = -1
    """Top-k sampling. -1 = disabled."""

    repetition_penalty: float = 1.0
    """Penalty for token repetition. 1.0 = disabled."""

    stop_token_ids: list[int] = field(default_factory=list)
    """Token IDs that trigger generation stop."""

    skip_special_tokens: bool = True
    """Whether to skip special tokens when decoding."""

    # Log probability options
    logprobs: bool = True
    """Whether to return log probabilities for sampled tokens.
    Always True - log probs are always computed and returned."""

    top_logprobs: int = 0
    """Number of top alternative tokens to return at each position.
    0 = only return sampled token's logprob.
    >0 = also return top-k alternatives with their logprobs.
    Max value typically 20 (like OpenAI API)."""

    @property
    def is_greedy(self) -> bool:
        """Check if sampling is deterministic (greedy)."""
        return self.temperature == 0.0 or self.top_k == 1

    def __post_init__(self):
        """Validate parameters."""
        # Log probs are always enabled
        self.logprobs = True

        # Clamp top_logprobs to reasonable range
        if self.top_logprobs < 0:
            self.top_logprobs = 0
        elif self.top_logprobs > 20:
            self.top_logprobs = 20

        # Validate temperature
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")

        # Validate n
        if self.n < 1:
            raise ValueError("n must be >= 1")
