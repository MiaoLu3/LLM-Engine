"""Tests for configuration dataclasses."""

import pytest
from llm_engine.config import (
    ModelConfig,
    SchedulerConfig,
    CacheConfig,
    EngineConfig,
    SamplingParams,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_basic_creation(self):
        """Test creating a basic model config."""
        config = ModelConfig(model_name_or_path="Qwen/Qwen2.5-7B")
        assert config.model_name_or_path == "Qwen/Qwen2.5-7B"
        assert config.dtype == "bfloat16"
        assert config.trust_remote_code is True
        assert config.max_model_len is None

    def test_custom_dtype(self):
        """Test setting custom dtype."""
        config = ModelConfig(
            model_name_or_path="test-model",
            dtype="float16",
        )
        assert config.dtype == "float16"

    def test_derived_attributes_default(self):
        """Test that derived attributes start at 0."""
        config = ModelConfig(model_name_or_path="test-model")
        assert config.num_layers == 0
        assert config.num_heads == 0
        assert config.num_kv_heads == 0
        assert config.head_dim == 0
        assert config.hidden_size == 0
        assert config.vocab_size == 0


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_default_values(self):
        """Test default scheduler config values."""
        config = SchedulerConfig()
        assert config.max_num_seqs == 256
        assert config.max_num_batched_tokens == 8192
        assert config.enable_chunked_prefill is False
        assert config.max_prefill_tokens is None

    def test_custom_values(self):
        """Test custom scheduler config."""
        config = SchedulerConfig(
            max_num_seqs=512,
            max_num_batched_tokens=16384,
            enable_chunked_prefill=True,
            max_prefill_tokens=4096,
        )
        assert config.max_num_seqs == 512
        assert config.max_num_batched_tokens == 16384
        assert config.enable_chunked_prefill is True
        assert config.max_prefill_tokens == 4096


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Test default cache config values."""
        config = CacheConfig()
        assert config.block_size == 16
        assert config.gpu_memory_utilization == 0.9
        assert config.enable_prefix_caching is False
        assert config.swap_space_gb == 0.0

    def test_custom_values(self):
        """Test custom cache config."""
        config = CacheConfig(
            block_size=32,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,
            swap_space_gb=4.0,
        )
        assert config.block_size == 32
        assert config.gpu_memory_utilization == 0.85
        assert config.enable_prefix_caching is True
        assert config.swap_space_gb == 4.0


class TestEngineConfig:
    """Tests for EngineConfig dataclass."""

    def test_basic_creation(self):
        """Test creating engine config with model config."""
        model_config = ModelConfig(model_name_or_path="test-model")
        engine_config = EngineConfig(model=model_config)

        assert engine_config.model.model_name_or_path == "test-model"
        assert engine_config.device == "cuda"
        assert engine_config.seed == 42

    def test_from_model_name(self):
        """Test convenience constructor from model name."""
        config = EngineConfig.from_model_name(
            model_name_or_path="Qwen/Qwen2.5-7B",
            max_num_seqs=128,
            max_num_batched_tokens=4096,
            gpu_memory_utilization=0.85,
            enable_prefix_caching=True,
        )

        assert config.model.model_name_or_path == "Qwen/Qwen2.5-7B"
        assert config.scheduler.max_num_seqs == 128
        assert config.scheduler.max_num_batched_tokens == 4096
        assert config.cache.gpu_memory_utilization == 0.85
        assert config.cache.enable_prefix_caching is True

    def test_default_sub_configs(self):
        """Test that sub-configs use defaults when not specified."""
        model_config = ModelConfig(model_name_or_path="test-model")
        engine_config = EngineConfig(model=model_config)

        # Should use default SchedulerConfig
        assert engine_config.scheduler.max_num_seqs == 256

        # Should use default CacheConfig
        assert engine_config.cache.block_size == 16


class TestSamplingParams:
    """Tests for SamplingParams dataclass."""

    def test_default_values(self):
        """Test default sampling params."""
        params = SamplingParams()
        assert params.n == 1
        assert params.max_tokens == 256
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == -1
        assert params.repetition_penalty == 1.0
        assert params.stop_token_ids == []
        assert params.skip_special_tokens is True

    def test_logprobs_always_true(self):
        """Test that logprobs is always True."""
        params = SamplingParams()
        assert params.logprobs is True

        # Even if explicitly set to False, should be True after __post_init__
        params = SamplingParams(logprobs=False)
        assert params.logprobs is True

    def test_top_logprobs_default(self):
        """Test default top_logprobs value."""
        params = SamplingParams()
        assert params.top_logprobs == 0

    def test_top_logprobs_clamping(self):
        """Test that top_logprobs is clamped to 0-20."""
        # Negative should be clamped to 0
        params = SamplingParams(top_logprobs=-5)
        assert params.top_logprobs == 0

        # Above 20 should be clamped to 20
        params = SamplingParams(top_logprobs=50)
        assert params.top_logprobs == 20

        # Valid value should be preserved
        params = SamplingParams(top_logprobs=10)
        assert params.top_logprobs == 10

    def test_is_greedy_with_zero_temp(self):
        """Test is_greedy with temperature=0."""
        params = SamplingParams(temperature=0.0)
        assert params.is_greedy is True

    def test_is_greedy_with_top_k_one(self):
        """Test is_greedy with top_k=1."""
        params = SamplingParams(top_k=1)
        assert params.is_greedy is True

    def test_is_not_greedy(self):
        """Test is_greedy is False for stochastic sampling."""
        params = SamplingParams(temperature=0.7)
        assert params.is_greedy is False

    def test_invalid_temperature(self):
        """Test that negative temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be >= 0"):
            SamplingParams(temperature=-0.5)

    def test_invalid_n(self):
        """Test that n < 1 raises error."""
        with pytest.raises(ValueError, match="n must be >= 1"):
            SamplingParams(n=0)

    def test_custom_stop_tokens(self):
        """Test setting custom stop token IDs."""
        params = SamplingParams(stop_token_ids=[100, 101, 102])
        assert params.stop_token_ids == [100, 101, 102]
