"""
Model building blocks for Qwen3.

This module provides the core layer components:
- RMSNorm: Root Mean Square Layer Normalization
- SwiGLU MLP: Gated Linear Unit with SiLU activation

References:
- RMSNorm: https://arxiv.org/abs/1910.07467
- SwiGLU: https://arxiv.org/abs/2002.05202
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Unlike LayerNorm, RMSNorm only normalizes by the root mean square
    and does not re-center (no mean subtraction). This is computationally
    cheaper and works well in practice.

    Formula:
        y = x / sqrt(mean(x^2) + eps) * weight

    Args:
        hidden_size: Size of the last dimension to normalize.
        eps: Small constant for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.hidden_size = hidden_size

    def _norm(self, x: Tensor) -> Tensor:
        """Compute RMS normalization."""
        # x^2
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # x / sqrt(variance + eps)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor [..., hidden_size].

        Returns:
            Normalized tensor with same shape.
        """
        # Compute in float32 for stability, then convert back
        input_dtype = x.dtype
        x = x.float()
        x = self._norm(x)
        x = x.to(input_dtype)
        return x * self.weight


class Qwen3MLP(nn.Module):
    """
    SwiGLU MLP block used in Qwen3.

    The SwiGLU activation combines a gating mechanism with SiLU (Swish)
    activation, providing better training dynamics than plain ReLU or GELU.

    Formula:
        output = down_proj(silu(gate_proj(x)) * up_proj(x))

    The gate and up projections expand to intermediate_size,
    then down projects back to hidden_size.

    Args:
        hidden_size: Input/output dimension.
        intermediate_size: Expanded dimension (typically ~2.7x hidden_size).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Gate projection: controls how much of each feature passes through
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # Up projection: transforms input to intermediate space
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # Down projection: projects back to hidden size
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply SwiGLU MLP.

        Args:
            x: Input tensor [..., hidden_size].

        Returns:
            Output tensor [..., hidden_size].
        """
        # SwiGLU: silu(gate) * up
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up

        # Project back down
        output = self.down_proj(hidden)
        return output


class Qwen3DecoderLayer(nn.Module):
    """
    Single decoder layer for Qwen3.

    Structure (pre-norm architecture):
        1. input_layernorm -> self_attention -> residual
        2. post_attention_layernorm -> mlp -> residual

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: MLP intermediate dimension.
        num_attention_heads: Number of attention heads.
        num_key_value_heads: Number of KV heads (for GQA).
        head_dim: Dimension per attention head.
        rms_norm_eps: Epsilon for RMSNorm.
        layer_idx: Index of this layer (for debugging).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        # Pre-attention normalization
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # Self-attention (will be set by Qwen3Model)
        # Keeping this as a placeholder - attention will be imported separately
        # to avoid circular imports
        self.self_attn = None

        # Post-attention normalization
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # MLP
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,
        attention_output: Tensor,
    ) -> Tensor:
        """
        Forward pass for decoder layer.

        Note: Attention is computed externally to allow for flexible
        attention implementations (FlashAttention, PagedAttention, etc.)

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_output: Pre-computed attention output [batch, seq_len, hidden_size].

        Returns:
            Output tensor [batch, seq_len, hidden_size].
        """
        # Residual connection after attention
        hidden_states = hidden_states + attention_output

        # MLP block with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward_with_attention(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple,
        **attention_kwargs,
    ):
        """
        Full forward pass including attention computation.

        This method is used when self_attn is set.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            position_embeddings: Tuple of (cos, sin) for RoPE.
            **attention_kwargs: Additional arguments for attention.

        Returns:
            Tuple of (output, (key, value)) where key/value are for KV cache.
        """
        if self.self_attn is None:
            raise ValueError("self_attn not set. Use forward() with pre-computed attention.")

        # Pre-norm for attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_output, kv = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            **attention_kwargs,
        )

        # Residual connection
        hidden_states = residual + attn_output

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, kv
