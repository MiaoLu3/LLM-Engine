"""Tests for attention mechanisms (FlashAttention, PagedAttention)."""

import pytest
import torch
import math

from llm_engine.model.attention import (
    naive_attention,
    paged_attention_decode,
    Qwen3Attention,
)


# Skip FlashAttention tests if not available or no CUDA
try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True and torch.cuda.is_available()
except ImportError:
    HAS_FLASH_ATTN = False


class TestNaiveAttention:
    """Tests for naive attention implementation."""

    def test_output_shape(self):
        """Test output shape matches expected."""
        batch, num_heads, seq_len, head_dim = 2, 8, 10, 64
        query = torch.randn(batch, num_heads, seq_len, head_dim)
        key = torch.randn(batch, num_heads, seq_len, head_dim)
        value = torch.randn(batch, num_heads, seq_len, head_dim)

        output = naive_attention(query, key, value)
        assert output.shape == (batch, num_heads, seq_len, head_dim)

    def test_causal_mask(self):
        """Test that causal mask prevents attending to future tokens."""
        batch, num_heads, seq_len, head_dim = 1, 1, 5, 64
        query = torch.randn(batch, num_heads, seq_len, head_dim)
        key = torch.randn(batch, num_heads, seq_len, head_dim)
        value = torch.randn(batch, num_heads, seq_len, head_dim)

        output = naive_attention(query, key, value, causal=True)

        # For causal attention, output at position i should only depend on j <= i
        # We can verify this by modifying future values and checking output doesn't change
        value_modified = value.clone()
        value_modified[:, :, -1, :] *= 2  # Modify last position

        output_modified = naive_attention(query, key, value_modified, causal=True)

        # Outputs for positions 0-3 should be identical (they can't see position 4)
        assert torch.allclose(output[:, :, :-1, :], output_modified[:, :, :-1, :], atol=1e-5)

    def test_non_causal(self):
        """Test non-causal attention."""
        batch, num_heads, seq_len, head_dim = 1, 1, 5, 64
        query = torch.randn(batch, num_heads, seq_len, head_dim)
        key = torch.randn(batch, num_heads, seq_len, head_dim)
        value = torch.randn(batch, num_heads, seq_len, head_dim)

        output = naive_attention(query, key, value, causal=False)
        assert output.shape == (batch, num_heads, seq_len, head_dim)

    def test_gqa_support(self):
        """Test Grouped Query Attention (different num_kv_heads)."""
        batch, q_heads, kv_heads, seq_len, head_dim = 2, 8, 4, 10, 64
        query = torch.randn(batch, q_heads, seq_len, head_dim)
        key = torch.randn(batch, kv_heads, seq_len, head_dim)
        value = torch.randn(batch, kv_heads, seq_len, head_dim)

        output = naive_attention(query, key, value)
        assert output.shape == (batch, q_heads, seq_len, head_dim)

    def test_softmax_scale(self):
        """Test custom softmax scale."""
        batch, num_heads, seq_len, head_dim = 2, 8, 10, 64
        query = torch.randn(batch, num_heads, seq_len, head_dim)
        key = torch.randn(batch, num_heads, seq_len, head_dim)
        value = torch.randn(batch, num_heads, seq_len, head_dim)

        output1 = naive_attention(query, key, value, softmax_scale=1.0)
        output2 = naive_attention(query, key, value, softmax_scale=0.5)

        # Different scales should give different outputs
        assert not torch.allclose(output1, output2)


class TestPagedAttentionDecode:
    """Tests for PagedAttention decode."""

    def test_output_shape(self):
        """Test output shape."""
        batch_size = 2
        num_heads, num_kv_heads, head_dim = 8, 4, 64
        num_blocks, block_size = 10, 16
        max_blocks_per_seq = 4

        query = torch.randn(batch_size, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        block_tables = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq))
        context_lens = torch.tensor([32, 48])

        output = paged_attention_decode(query, k_cache, v_cache, block_tables, context_lens)
        assert output.shape == (batch_size, num_heads, head_dim)

    def test_context_length_masking(self):
        """Test that context length is properly masked."""
        batch_size = 2
        num_heads, num_kv_heads, head_dim = 4, 2, 32
        num_blocks, block_size = 5, 8
        max_blocks_per_seq = 2

        query = torch.randn(batch_size, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        block_tables = torch.tensor([[0, 1], [2, 3]])

        # Different context lengths
        context_lens_1 = torch.tensor([8, 8])
        context_lens_2 = torch.tensor([4, 4])  # Shorter context

        output1 = paged_attention_decode(query, k_cache, v_cache, block_tables, context_lens_1)
        output2 = paged_attention_decode(query, k_cache, v_cache, block_tables, context_lens_2)

        # Different context lengths should give different outputs
        assert not torch.allclose(output1, output2)

    def test_gqa_expansion(self):
        """Test GQA head expansion in PagedAttention."""
        batch_size = 1
        num_heads, num_kv_heads, head_dim = 8, 2, 32  # 4:1 ratio
        num_blocks, block_size = 4, 8
        max_blocks_per_seq = 2

        query = torch.randn(batch_size, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        block_tables = torch.tensor([[0, 1]])
        context_lens = torch.tensor([16])

        output = paged_attention_decode(query, k_cache, v_cache, block_tables, context_lens)
        assert output.shape == (batch_size, num_heads, head_dim)

    def test_single_sequence(self):
        """Test with single sequence."""
        batch_size = 1
        num_heads, num_kv_heads, head_dim = 4, 4, 32
        num_blocks, block_size = 4, 8
        max_blocks_per_seq = 2

        query = torch.randn(batch_size, num_heads, head_dim)
        k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim)
        block_tables = torch.tensor([[0, 1]])
        context_lens = torch.tensor([12])

        output = paged_attention_decode(query, k_cache, v_cache, block_tables, context_lens)
        assert output.shape == (batch_size, num_heads, head_dim)


class TestQwen3Attention:
    """Tests for Qwen3Attention module."""

    @pytest.fixture
    def attention(self):
        """Create a Qwen3Attention instance."""
        return Qwen3Attention(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=32,
            layer_idx=0,
        )

    def test_basic_creation(self, attention):
        """Test creating Qwen3Attention."""
        assert attention.hidden_size == 256
        assert attention.num_heads == 8
        assert attention.num_kv_heads == 4
        assert attention.head_dim == 32
        assert attention.num_key_value_groups == 2  # 8 / 4

    def test_projection_shapes(self, attention):
        """Test projection layer shapes."""
        # Q: hidden -> num_heads * head_dim
        assert attention.q_proj.weight.shape == (8 * 32, 256)
        # K: hidden -> num_kv_heads * head_dim
        assert attention.k_proj.weight.shape == (4 * 32, 256)
        # V: hidden -> num_kv_heads * head_dim
        assert attention.v_proj.weight.shape == (4 * 32, 256)
        # O: num_heads * head_dim -> hidden
        assert attention.o_proj.weight.shape == (256, 8 * 32)

    def test_prefill_padded_forward(self, attention):
        """Test prefill forward with padded batches."""
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, 256)

        # Create position embeddings
        cos = torch.randn(batch_size, seq_len, 32)
        sin = torch.randn(batch_size, seq_len, 32)

        output, (k, v) = attention(
            hidden_states,
            position_embeddings=(cos, sin),
            is_prefill=True,
        )

        assert output.shape == (batch_size, seq_len, 256)

    def test_decode_forward(self, attention):
        """Test decode forward with PagedAttention."""
        batch_size = 2
        num_blocks, block_size = 10, 16
        max_blocks_per_seq = 4

        hidden_states = torch.randn(batch_size, 1, 256)  # Single token per seq
        cos = torch.randn(batch_size, 1, 32)
        sin = torch.randn(batch_size, 1, 32)

        k_cache = torch.randn(num_blocks, block_size, 4, 32)
        v_cache = torch.randn(num_blocks, block_size, 4, 32)
        block_tables = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq))
        context_lens = torch.tensor([32, 48])

        output, (k, v) = attention(
            hidden_states,
            position_embeddings=(cos, sin),
            is_prefill=False,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
        )

        # Output should be [batch, hidden] or [batch, 1, hidden]
        assert output.shape[0] == batch_size
        assert output.shape[-1] == 256

    def test_gqa_ratio(self):
        """Test different GQA ratios."""
        for num_heads, num_kv_heads in [(8, 8), (8, 4), (8, 2), (8, 1)]:
            attention = Qwen3Attention(
                hidden_size=256,
                num_attention_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=32,
            )

            expected_ratio = num_heads // num_kv_heads
            assert attention.num_key_value_groups == expected_ratio


@pytest.mark.skipif(not HAS_FLASH_ATTN, reason="FlashAttention not available")
class TestFlashAttention:
    """Tests for FlashAttention (requires flash_attn package)."""

    def test_flash_attention_prefill(self):
        """Test FlashAttention prefill."""
        from llm_engine.model.attention import flash_attention_prefill

        total_tokens = 30
        num_heads, head_dim = 8, 64

        query = torch.randn(total_tokens, num_heads, head_dim, device="cuda", dtype=torch.float16)
        key = torch.randn(total_tokens, num_heads, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn(total_tokens, num_heads, head_dim, device="cuda", dtype=torch.float16)

        # 3 sequences of lengths 10, 8, 12
        cu_seqlens = torch.tensor([0, 10, 18, 30], device="cuda", dtype=torch.int32)
        max_seqlen = 12

        output = flash_attention_prefill(
            query, key, value, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        )

        assert output.shape == (total_tokens, num_heads, head_dim)

    def test_flash_attention_gqa(self):
        """Test FlashAttention with GQA."""
        from llm_engine.model.attention import flash_attention_prefill

        total_tokens = 20
        q_heads, kv_heads, head_dim = 8, 2, 64

        query = torch.randn(total_tokens, q_heads, head_dim, device="cuda", dtype=torch.float16)
        key = torch.randn(total_tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16)
        value = torch.randn(total_tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16)

        cu_seqlens = torch.tensor([0, 10, 20], device="cuda", dtype=torch.int32)
        max_seqlen = 10

        output = flash_attention_prefill(
            query, key, value, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        )

        assert output.shape == (total_tokens, q_heads, head_dim)


class TestAttentionCorrectness:
    """Tests comparing attention implementations for correctness."""

    def test_naive_attention_softmax_normalized(self):
        """Test that naive attention produces normalized attention weights."""
        batch, num_heads, seq_len, head_dim = 1, 1, 5, 32
        query = torch.randn(batch, num_heads, seq_len, head_dim)
        key = torch.randn(batch, num_heads, seq_len, head_dim)
        value = torch.ones(batch, num_heads, seq_len, head_dim)  # All ones

        output = naive_attention(query, key, value, causal=False)

        # With all-ones values, output should be close to 1 (weighted average of 1s = 1)
        assert torch.allclose(output, torch.ones_like(output), atol=1e-5)

    def test_identity_attention(self):
        """Test attention with identity-like patterns."""
        batch, num_heads, seq_len, head_dim = 1, 1, 5, 32

        # Create Q and K such that each query only attends to its own position
        # Use identity matrix repeated along head_dim
        identity = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)  # [1, 1, 5, 5]
        # Repeat along head_dim to get [1, 1, 5, 32]
        query = identity.repeat(1, 1, 1, head_dim // seq_len + 1)[:, :, :, :head_dim]
        key = query.clone()
        value = torch.arange(seq_len).float().view(1, 1, seq_len, 1).expand(-1, -1, -1, head_dim)

        # The attention should approximately copy values
        output = naive_attention(query, key, value, causal=False, softmax_scale=10.0)

        # Each output position should be close to its value
        for i in range(seq_len):
            expected = float(i)
            actual = output[0, 0, i, 0].item()
            assert abs(actual - expected) < 0.5  # Approximate due to softmax spread
