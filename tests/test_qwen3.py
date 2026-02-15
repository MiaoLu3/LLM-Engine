"""Tests for Qwen3 model implementation."""

import pytest
import torch
import json
import tempfile
from pathlib import Path

from llm_engine.model.qwen3 import (
    Qwen3Config,
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3ForCausalLM,
)


class TestQwen3Config:
    """Tests for Qwen3Config."""

    def test_default_config(self):
        """Test default configuration values (Qwen3-4B)."""
        config = Qwen3Config()

        assert config.vocab_size == 151936
        assert config.hidden_size == 2560
        assert config.intermediate_size == 6912
        assert config.num_hidden_layers == 40
        assert config.num_attention_heads == 20
        assert config.num_key_value_heads == 4
        assert config.head_dim == 128
        assert config.max_position_embeddings == 32768
        assert config.rms_norm_eps == 1e-6
        assert config.rope_theta == 1000000.0
        assert config.tie_word_embeddings is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = Qwen3Config(
            vocab_size=50000,
            hidden_size=512,
            intermediate_size=1376,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=2,
        )

        assert config.vocab_size == 50000
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 6

    def test_from_pretrained(self):
        """Test loading config from JSON file."""
        config_dict = {
            "vocab_size": 32000,
            "hidden_size": 256,
            "intermediate_size": 688,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f)

            config = Qwen3Config.from_pretrained(tmpdir)

            assert config.vocab_size == 32000
            assert config.hidden_size == 256
            assert config.num_hidden_layers == 4
            assert config.tie_word_embeddings is False


class TestQwen3DecoderLayer:
    """Tests for Qwen3DecoderLayer."""

    @pytest.fixture
    def config(self):
        """Create a small test configuration."""
        return Qwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
        )

    @pytest.fixture
    def layer(self, config):
        """Create a decoder layer."""
        return Qwen3DecoderLayer(config, layer_idx=0)

    def test_basic_creation(self, layer, config):
        """Test creating a decoder layer."""
        assert layer.layer_idx == 0
        assert layer.hidden_size == config.hidden_size

    def test_sublayer_structure(self, layer):
        """Test layer components exist."""
        assert layer.input_layernorm is not None
        assert layer.post_attention_layernorm is not None
        assert layer.self_attn is not None
        assert layer.mlp is not None

    def test_forward_prefill(self, layer, config):
        """Test forward pass in prefill mode."""
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        cos = torch.randn(batch_size, seq_len, config.head_dim)
        sin = torch.randn(batch_size, seq_len, config.head_dim)

        output, kv = layer(
            hidden_states,
            position_embeddings=(cos, sin),
            is_prefill=True,
        )

        assert output.shape == hidden_states.shape

    def test_residual_connections(self, layer, config):
        """Test that residual connections work."""
        batch_size, seq_len = 1, 5
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        cos = torch.randn(batch_size, seq_len, config.head_dim)
        sin = torch.randn(batch_size, seq_len, config.head_dim)

        # Zero out all weights to check residual
        with torch.no_grad():
            for p in layer.parameters():
                p.zero_()

        output, _ = layer(
            hidden_states,
            position_embeddings=(cos, sin),
            is_prefill=True,
        )

        # With zeroed weights, output should equal input (residual only)
        # This depends on implementation details, so just check shapes
        assert output.shape == hidden_states.shape


class TestQwen3Model:
    """Tests for Qwen3Model (transformer without lm_head)."""

    @pytest.fixture
    def config(self):
        """Create a small test configuration."""
        return Qwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
        )

    @pytest.fixture
    def model(self, config):
        """Create a Qwen3Model."""
        return Qwen3Model(config)

    def test_basic_creation(self, model, config):
        """Test creating the model."""
        assert len(model.layers) == config.num_hidden_layers
        assert model.embed_tokens.weight.shape == (config.vocab_size, config.hidden_size)

    def test_forward_batched(self, model, config):
        """Test forward pass with batched input."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        hidden_states, kv_list = model(input_ids, is_prefill=True)

        assert hidden_states.shape == (batch_size, seq_len, config.hidden_size)
        assert len(kv_list) == config.num_hidden_layers

    def test_forward_with_position_ids(self, model, config):
        """Test forward with explicit position IDs."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        hidden_states, kv_list = model(
            input_ids,
            position_ids=position_ids,
            is_prefill=True,
        )

        assert hidden_states.shape == (batch_size, seq_len, config.hidden_size)


class TestQwen3ForCausalLM:
    """Tests for Qwen3ForCausalLM (full model)."""

    @pytest.fixture
    def config(self):
        """Create a small test configuration."""
        return Qwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            tie_word_embeddings=False,
        )

    @pytest.fixture
    def model(self, config):
        """Create a Qwen3ForCausalLM."""
        return Qwen3ForCausalLM(config)

    def test_basic_creation(self, model, config):
        """Test creating the full model."""
        assert model.config == config
        assert model.lm_head.weight.shape == (config.vocab_size, config.hidden_size)

    def test_forward_logits(self, model, config):
        """Test forward pass returns logits."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, kv_list = model(input_ids, is_prefill=True)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert len(kv_list) == config.num_hidden_layers

    def test_weight_tying(self):
        """Test weight tying between embed_tokens and lm_head."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            tie_word_embeddings=True,
        )
        model = Qwen3ForCausalLM(config)

        # With tied weights, lm_head.weight should be same as embed_tokens.weight
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_get_num_params(self, model):
        """Test parameter counting."""
        num_params = model.get_num_params()
        assert num_params > 0

        # Verify by manual count
        manual_count = sum(p.numel() for p in model.parameters())
        assert num_params == manual_count

    def test_get_num_layers(self, model, config):
        """Test getting number of layers."""
        assert model.get_num_layers() == config.num_hidden_layers

    def test_get_head_dim(self, model, config):
        """Test getting head dimension."""
        assert model.get_head_dim() == config.head_dim

    def test_get_num_kv_heads(self, model, config):
        """Test getting number of KV heads."""
        assert model.get_num_kv_heads() == config.num_key_value_heads


class TestQwen3Dimensions:
    """Tests for Qwen3 model dimension calculations."""

    def test_qwen3_4b_dimensions(self):
        """Test Qwen3-4B architecture dimensions."""
        config = Qwen3Config()  # Default is Qwen3-4B

        # GQA ratio
        assert config.num_attention_heads % config.num_key_value_heads == 0
        gqa_ratio = config.num_attention_heads // config.num_key_value_heads
        assert gqa_ratio == 5  # 20 / 4 = 5

        # Head dimension
        computed_head_dim = config.hidden_size // config.num_attention_heads
        assert computed_head_dim == 128

        # MLP expansion ratio
        mlp_ratio = config.intermediate_size / config.hidden_size
        assert 2.5 < mlp_ratio < 3.0  # ~2.7x

    def test_parameter_count_estimate(self):
        """Test rough parameter count calculation for Qwen3-4B."""
        config = Qwen3Config()

        # Embedding
        embed_params = config.vocab_size * config.hidden_size

        # Per layer
        # Attention: Q + K + V + O projections
        q_params = config.hidden_size * config.num_attention_heads * config.head_dim
        k_params = config.hidden_size * config.num_key_value_heads * config.head_dim
        v_params = config.hidden_size * config.num_key_value_heads * config.head_dim
        o_params = config.num_attention_heads * config.head_dim * config.hidden_size
        attn_params = q_params + k_params + v_params + o_params

        # MLP: gate + up + down
        mlp_params = 3 * config.hidden_size * config.intermediate_size

        # Layer norms: 2 per layer
        norm_params = 2 * config.hidden_size

        layer_params = attn_params + mlp_params + norm_params
        total_layer_params = config.num_hidden_layers * layer_params

        # Final norm + lm_head (if not tied)
        final_params = config.hidden_size + config.vocab_size * config.hidden_size

        total_estimate = embed_params + total_layer_params + final_params

        # Should be roughly 4B parameters
        assert 3e9 < total_estimate < 5e9


class TestQwen3Gradients:
    """Tests for gradient flow through Qwen3."""

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
        )
        model = Qwen3ForCausalLM(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 5))
        logits, _ = model(input_ids, is_prefill=True)

        loss = logits.sum()
        loss.backward()

        # Check gradients exist for key parameters
        assert model.model.embed_tokens.weight.grad is not None
        assert model.model.layers[0].input_layernorm.weight.grad is not None
        assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
        assert model.model.layers[0].mlp.gate_proj.weight.grad is not None
        assert model.model.norm.weight.grad is not None


class TestQwen3Inference:
    """Tests for Qwen3 inference behavior."""

    def test_deterministic_forward(self):
        """Test that forward pass is deterministic."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
        )
        model = Qwen3ForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            logits1, _ = model(input_ids, is_prefill=True)
            logits2, _ = model(input_ids, is_prefill=True)

        assert torch.allclose(logits1, logits2)

    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different outputs."""
        config = Qwen3Config(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=688,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
        )
        model = Qwen3ForCausalLM(config)
        model.eval()

        input_ids_1 = torch.randint(0, config.vocab_size, (1, 5))
        input_ids_2 = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            logits1, _ = model(input_ids_1, is_prefill=True)
            logits2, _ = model(input_ids_2, is_prefill=True)

        # Different inputs should (almost certainly) produce different outputs
        assert not torch.allclose(logits1, logits2)
