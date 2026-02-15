"""Tests for model building blocks (RMSNorm, MLP)."""

import pytest
import torch
import torch.nn.functional as F

from llm_engine.model.layers import RMSNorm, Qwen3MLP


class TestRMSNorm:
    """Tests for RMSNorm."""

    def test_basic_creation(self):
        """Test creating RMSNorm."""
        norm = RMSNorm(hidden_size=256)
        assert norm.hidden_size == 256
        assert norm.eps == 1e-6
        assert norm.weight.shape == (256,)

    def test_custom_eps(self):
        """Test custom epsilon value."""
        norm = RMSNorm(hidden_size=256, eps=1e-8)
        assert norm.eps == 1e-8

    def test_output_shape(self):
        """Test that output shape matches input."""
        norm = RMSNorm(hidden_size=256)
        x = torch.randn(2, 10, 256)

        output = norm(x)
        assert output.shape == x.shape

    def test_normalization_magnitude(self):
        """Test that output has approximately unit RMS per position."""
        norm = RMSNorm(hidden_size=256)
        # Initialize weight to ones for this test
        norm.weight.data.fill_(1.0)

        x = torch.randn(2, 10, 256) * 5  # Scale input
        output = norm(x)

        # Check RMS of each position is approximately 1
        rms = output.pow(2).mean(dim=-1).sqrt()
        expected_rms = torch.ones_like(rms)
        assert torch.allclose(rms, expected_rms, atol=0.1)

    def test_learnable_weight(self):
        """Test that weight parameter affects output."""
        norm = RMSNorm(hidden_size=256)
        x = torch.randn(2, 10, 256)

        output1 = norm(x)

        # Modify weights
        norm.weight.data *= 2.0
        output2 = norm(x)

        # Outputs should be different
        assert not torch.allclose(output1, output2)
        # Output should scale by 2
        assert torch.allclose(output2, output1 * 2, atol=1e-5)

    def test_dtype_preservation(self):
        """Test that input dtype is preserved."""
        norm = RMSNorm(hidden_size=256)

        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            norm_typed = norm.to(dtype)
            x = torch.randn(2, 10, 256, dtype=dtype)
            output = norm_typed(x)
            assert output.dtype == dtype

    def test_gradient_flow(self):
        """Test that gradients flow through normalization."""
        norm = RMSNorm(hidden_size=256)
        x = torch.randn(2, 10, 256, requires_grad=True)

        output = norm(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert norm.weight.grad is not None

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        norm = RMSNorm(hidden_size=256)

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 10, 256)
            output = norm(x)
            assert output.shape == (batch_size, 10, 256)

    def test_2d_input(self):
        """Test with 2D input (no batch dimension)."""
        norm = RMSNorm(hidden_size=256)
        x = torch.randn(10, 256)

        output = norm(x)
        assert output.shape == (10, 256)


class TestQwen3MLP:
    """Tests for Qwen3MLP (SwiGLU)."""

    def test_basic_creation(self):
        """Test creating Qwen3MLP."""
        mlp = Qwen3MLP(hidden_size=256, intermediate_size=688)
        assert mlp.hidden_size == 256
        assert mlp.intermediate_size == 688

    def test_output_shape(self):
        """Test that output shape matches input."""
        mlp = Qwen3MLP(hidden_size=256, intermediate_size=688)
        x = torch.randn(2, 10, 256)

        output = mlp(x)
        assert output.shape == x.shape

    def test_projection_sizes(self):
        """Test that projection layers have correct sizes."""
        mlp = Qwen3MLP(hidden_size=256, intermediate_size=688)

        # gate_proj: hidden -> intermediate
        assert mlp.gate_proj.weight.shape == (688, 256)
        # up_proj: hidden -> intermediate
        assert mlp.up_proj.weight.shape == (688, 256)
        # down_proj: intermediate -> hidden
        assert mlp.down_proj.weight.shape == (256, 688)

    def test_no_bias(self):
        """Test that projections have no bias."""
        mlp = Qwen3MLP(hidden_size=256, intermediate_size=688)

        assert mlp.gate_proj.bias is None
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None

    def test_swiglu_activation(self):
        """Test that SwiGLU activation is applied correctly."""
        mlp = Qwen3MLP(hidden_size=256, intermediate_size=688)

        # Create a simple input
        x = torch.randn(1, 1, 256)

        # Manual computation
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        hidden = F.silu(gate) * up
        expected = mlp.down_proj(hidden)

        output = mlp(x)
        assert torch.allclose(output, expected, atol=1e-6)

    def test_gradient_flow(self):
        """Test that gradients flow through MLP."""
        mlp = Qwen3MLP(hidden_size=256, intermediate_size=688)
        x = torch.randn(2, 10, 256, requires_grad=True)

        output = mlp(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert mlp.gate_proj.weight.grad is not None
        assert mlp.up_proj.weight.grad is not None
        assert mlp.down_proj.weight.grad is not None

    def test_qwen3_4b_dimensions(self):
        """Test with Qwen3-4B dimensions."""
        # Qwen3-4B: hidden=2560, intermediate=6912
        mlp = Qwen3MLP(hidden_size=2560, intermediate_size=6912)
        x = torch.randn(1, 1, 2560)

        output = mlp(x)
        assert output.shape == (1, 1, 2560)

    def test_different_dtypes(self):
        """Test with different data types."""
        mlp = Qwen3MLP(hidden_size=256, intermediate_size=688)

        for dtype in [torch.float32, torch.float16]:
            mlp_typed = mlp.to(dtype)
            x = torch.randn(2, 10, 256, dtype=dtype)
            output = mlp_typed(x)
            assert output.dtype == dtype


class TestRMSNormVsLayerNorm:
    """Tests comparing RMSNorm behavior vs LayerNorm."""

    def test_rmsnorm_no_recentering(self):
        """Test that RMSNorm doesn't recenter (subtract mean)."""
        norm = RMSNorm(hidden_size=256)
        norm.weight.data.fill_(1.0)

        # Create input with non-zero mean
        x = torch.randn(2, 10, 256) + 5.0

        output = norm(x)

        # Output should NOT have zero mean (unlike LayerNorm)
        output_mean = output.mean(dim=-1)
        assert not torch.allclose(output_mean, torch.zeros_like(output_mean), atol=0.1)

    def test_rmsnorm_computational_efficiency(self):
        """Test that RMSNorm requires fewer operations than LayerNorm."""
        # This is more of a documentation test showing RMSNorm is simpler
        hidden_size = 256

        rms_norm = RMSNorm(hidden_size)
        layer_norm = torch.nn.LayerNorm(hidden_size)

        # RMSNorm has only one learnable parameter (weight)
        rms_params = sum(p.numel() for p in rms_norm.parameters())
        # LayerNorm has two (weight and bias)
        layer_params = sum(p.numel() for p in layer_norm.parameters())

        assert rms_params == hidden_size  # Just weight
        assert layer_params == 2 * hidden_size  # weight + bias


class TestMLPMemoryEfficiency:
    """Tests for MLP memory patterns."""

    def test_intermediate_size_ratio(self):
        """Test typical intermediate to hidden size ratios."""
        # Qwen3 uses ~2.7x ratio
        hidden_size = 2560
        intermediate_size = 6912
        ratio = intermediate_size / hidden_size

        assert 2.5 < ratio < 3.0

    def test_mlp_parameter_count(self):
        """Test MLP parameter count calculation."""
        hidden_size = 2560
        intermediate_size = 6912

        mlp = Qwen3MLP(hidden_size, intermediate_size)
        total_params = sum(p.numel() for p in mlp.parameters())

        # 3 projections: gate, up, down
        expected = 3 * hidden_size * intermediate_size
        assert total_params == expected
