"""Tests for Rotary Position Embeddings (RoPE)."""

import pytest
import torch
import math

from llm_engine.model.rope import (
    compute_rope_frequencies,
    compute_rope_cos_sin,
    rotate_half,
    apply_rope,
    apply_rope_to_qk,
    RotaryEmbedding,
)


class TestComputeRopeFrequencies:
    """Tests for compute_rope_frequencies."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        head_dim = 128
        inv_freq = compute_rope_frequencies(head_dim)
        assert inv_freq.shape == (head_dim // 2,)

    def test_output_range(self):
        """Test that frequencies are in valid range."""
        head_dim = 128
        rope_theta = 1000000.0
        inv_freq = compute_rope_frequencies(head_dim, rope_theta)

        # All frequencies should be positive
        assert torch.all(inv_freq > 0)
        # Max frequency (i=0) should be 1.0
        assert torch.isclose(inv_freq[0], torch.tensor(1.0))
        # Min frequency should be theta^(-1) approximately
        assert inv_freq[-1] < 1.0

    def test_different_head_dims(self):
        """Test with various head dimensions."""
        for head_dim in [32, 64, 128, 256]:
            inv_freq = compute_rope_frequencies(head_dim)
            assert inv_freq.shape == (head_dim // 2,)

    def test_different_theta(self):
        """Test with different rope_theta values."""
        head_dim = 128

        # Higher theta = slower decay of frequencies
        freq_high_theta = compute_rope_frequencies(head_dim, rope_theta=10000.0)
        freq_low_theta = compute_rope_frequencies(head_dim, rope_theta=1000000.0)

        # With higher theta, the min frequency should be larger (slower decay)
        assert freq_high_theta[-1] > freq_low_theta[-1]


class TestComputeRopeCosSin:
    """Tests for compute_rope_cos_sin."""

    def test_output_shape(self):
        """Test output shapes."""
        batch_size, seq_len, head_dim = 2, 10, 128
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        cos, sin = compute_rope_cos_sin(position_ids, head_dim)

        assert cos.shape == (batch_size, seq_len, head_dim)
        assert sin.shape == (batch_size, seq_len, head_dim)

    def test_cos_sin_range(self):
        """Test that cos/sin are in valid range [-1, 1]."""
        position_ids = torch.arange(100).unsqueeze(0)
        cos, sin = compute_rope_cos_sin(position_ids, 128)

        assert torch.all(cos >= -1) and torch.all(cos <= 1)
        assert torch.all(sin >= -1) and torch.all(sin <= 1)

    def test_position_zero(self):
        """Test at position 0."""
        position_ids = torch.zeros(1, 1, dtype=torch.long)
        cos, sin = compute_rope_cos_sin(position_ids, 128)

        # At position 0: cos(0) = 1, sin(0) = 0
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)
        assert torch.allclose(sin, torch.zeros_like(sin), atol=1e-5)


class TestRotateHalf:
    """Tests for rotate_half."""

    def test_output_shape(self):
        """Test that output shape matches input."""
        x = torch.randn(2, 4, 10, 128)
        rotated = rotate_half(x)
        assert rotated.shape == x.shape

    def test_rotation_correctness(self):
        """Test the rotation transformation."""
        # Create simple tensor [1, 2, 3, 4]
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        rotated = rotate_half(x)

        # rotate_half should give [-3, -4, 1, 2]
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(rotated, expected)

    def test_double_rotation(self):
        """Test that rotating twice gives negative of original."""
        x = torch.randn(2, 4, 10, 128)
        rotated_once = rotate_half(x)
        rotated_twice = rotate_half(rotated_once)

        # Double rotation should give -x
        assert torch.allclose(rotated_twice, -x, atol=1e-6)


class TestApplyRope:
    """Tests for apply_rope."""

    def test_output_shape(self):
        """Test output shape matches input."""
        batch, heads, seq_len, head_dim = 2, 8, 10, 128
        x = torch.randn(batch, heads, seq_len, head_dim)
        cos = torch.randn(batch, seq_len, head_dim)
        sin = torch.randn(batch, seq_len, head_dim)

        output = apply_rope(x, cos, sin)
        assert output.shape == x.shape

    def test_identity_at_zero(self):
        """Test that at position 0 (cos=1, sin=0), output equals input."""
        batch, heads, seq_len, head_dim = 2, 8, 10, 128
        x = torch.randn(batch, heads, seq_len, head_dim)

        # At position 0: cos=1, sin=0
        cos = torch.ones(batch, seq_len, head_dim)
        sin = torch.zeros(batch, seq_len, head_dim)

        output = apply_rope(x, cos, sin)
        assert torch.allclose(output, x, atol=1e-6)


class TestApplyRopeToQK:
    """Tests for apply_rope_to_qk."""

    def test_both_rotated(self):
        """Test that both Q and K are rotated."""
        batch, q_heads, kv_heads, seq_len, head_dim = 2, 8, 4, 10, 128

        query = torch.randn(batch, q_heads, seq_len, head_dim)
        key = torch.randn(batch, kv_heads, seq_len, head_dim)
        cos = torch.randn(batch, seq_len, head_dim)
        sin = torch.randn(batch, seq_len, head_dim)

        q_rot, k_rot = apply_rope_to_qk(query, key, cos, sin)

        # Outputs should have same shapes
        assert q_rot.shape == query.shape
        assert k_rot.shape == key.shape

        # Q and K should be different from input (unless cos=1, sin=0)
        assert not torch.allclose(q_rot, query)
        assert not torch.allclose(k_rot, key)


class TestRotaryEmbedding:
    """Tests for RotaryEmbedding class."""

    def test_basic_creation(self):
        """Test creating RotaryEmbedding."""
        rope = RotaryEmbedding(head_dim=128, rope_theta=1000000.0)
        assert rope.head_dim == 128
        assert rope.rope_theta == 1000000.0

    def test_forward(self):
        """Test forward pass."""
        rope = RotaryEmbedding(head_dim=128)

        batch, q_heads, kv_heads, seq_len = 2, 8, 4, 10
        query = torch.randn(batch, q_heads, seq_len, 128)
        key = torch.randn(batch, kv_heads, seq_len, 128)

        q_rot, k_rot = rope(query, key)

        assert q_rot.shape == query.shape
        assert k_rot.shape == key.shape

    def test_with_position_ids(self):
        """Test forward with explicit position_ids."""
        rope = RotaryEmbedding(head_dim=128)

        batch, q_heads, kv_heads, seq_len = 2, 8, 4, 10
        query = torch.randn(batch, q_heads, seq_len, 128)
        key = torch.randn(batch, kv_heads, seq_len, 128)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

        q_rot, k_rot = rope(query, key, position_ids)

        assert q_rot.shape == query.shape
        assert k_rot.shape == key.shape

    def test_caching(self):
        """Test that cos/sin caching works."""
        rope = RotaryEmbedding(head_dim=128, max_position_embeddings=1000)

        query = torch.randn(1, 8, 10, 128)
        key = torch.randn(1, 4, 10, 128)

        # First call should populate cache
        rope(query, key)
        assert rope._cached_seq_len >= 10

        # Second call with same seq_len should use cache
        cached_cos = rope._cached_cos
        rope(query, key)
        assert rope._cached_cos is cached_cos  # Same object = cached

    def test_different_sequence_lengths(self):
        """Test with varying sequence lengths."""
        rope = RotaryEmbedding(head_dim=128)

        for seq_len in [1, 10, 100, 500]:
            query = torch.randn(1, 8, seq_len, 128)
            key = torch.randn(1, 4, seq_len, 128)
            q_rot, k_rot = rope(query, key)
            assert q_rot.shape == query.shape


class TestRopeRelativePosition:
    """Tests for relative position encoding property of RoPE."""

    def test_relative_position_invariance(self):
        """
        Test that RoPE preserves relative position information.

        The inner product <q_m, k_n> should depend only on m-n.
        """
        head_dim = 128
        rope = RotaryEmbedding(head_dim=head_dim)

        # Create identical Q and K at different positions
        base_vec = torch.randn(1, 1, 1, head_dim)
        query = base_vec.expand(1, 1, 10, head_dim)
        key = base_vec.expand(1, 1, 10, head_dim)

        # Apply RoPE
        q_rot, k_rot = rope(query, key)

        # Compute attention scores (inner products)
        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1))

        # The score at position (i, j) should equal score at (i+1, j+1)
        # because both have the same relative distance
        for offset in range(8):
            score_00 = scores[0, 0, 0, 0]
            score_diag = scores[0, 0, offset, offset]
            # All diagonal elements should be approximately equal
            # (same relative distance = 0)
            assert torch.isclose(score_00, score_diag, atol=1e-5)
