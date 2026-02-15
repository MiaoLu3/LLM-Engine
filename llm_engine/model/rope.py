"""
Rotary Position Embeddings (RoPE) for Qwen3.

RoPE encodes position information by rotating query and key vectors.
This allows the attention mechanism to be aware of relative positions
between tokens without explicit positional embeddings.

Key formula:
- x_rotated = x * cos(m * theta) + rotate_half(x) * sin(m * theta)
- where m is the position index and theta are the rotation frequencies

References:
- RoFormer: https://arxiv.org/abs/2104.09864
- Qwen3 uses rope_theta=1000000 for extended context
"""

import torch
from torch import Tensor
from typing import Tuple, Optional


def compute_rope_frequencies(
    head_dim: int,
    rope_theta: float = 1000000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Compute inverse frequencies for RoPE.

    The frequencies follow the formula:
        theta_i = rope_theta^(-2i/d) for i in [0, d/2)

    Args:
        head_dim: Dimension of each attention head (typically 128).
        rope_theta: Base for the frequency computation (1e6 for Qwen3).
        device: Device to create tensor on.
        dtype: Data type for computation.

    Returns:
        Inverse frequencies tensor [head_dim // 2].
    """
    # Create indices [0, 2, 4, ..., head_dim-2]
    freq_indices = torch.arange(0, head_dim, 2, device=device, dtype=dtype)

    # Compute inverse frequencies: 1 / (theta^(2i/d))
    inv_freq = 1.0 / (rope_theta ** (freq_indices / head_dim))

    return inv_freq


def compute_rope_cos_sin(
    position_ids: Tensor,
    head_dim: int,
    rope_theta: float = 1000000.0,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Compute cos and sin values for RoPE at given positions.

    Args:
        position_ids: Position indices [batch_size, seq_len].
        head_dim: Dimension of each attention head.
        rope_theta: RoPE base frequency.
        dtype: Output dtype (defaults to position_ids dtype or float32).

    Returns:
        Tuple of (cos, sin) tensors, each [batch_size, seq_len, head_dim].
    """
    device = position_ids.device
    if dtype is None:
        dtype = torch.float32

    # Get inverse frequencies [head_dim // 2]
    inv_freq = compute_rope_frequencies(
        head_dim, rope_theta, device=device, dtype=dtype
    )

    # Compute position * frequency for each position
    # position_ids: [batch, seq_len]
    # inv_freq: [head_dim // 2]
    # Result: [batch, seq_len, head_dim // 2]
    freqs = torch.einsum("bs, d -> bsd", position_ids.float(), inv_freq)

    # Concatenate to get full head_dim
    # Each frequency appears twice (for the rotation formula)
    emb = torch.cat([freqs, freqs], dim=-1)  # [batch, seq_len, head_dim]

    # Compute cos and sin
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    return cos, sin


def rotate_half(x: Tensor) -> Tensor:
    """
    Rotate half of the hidden dimensions.

    Transforms [x1, x2, x3, x4, ...] to [-x_{d/2+1}, -x_{d/2+2}, ..., x1, x2, ...]
    This is the rotation operation used in RoPE.

    Args:
        x: Input tensor [..., head_dim].

    Returns:
        Rotated tensor with same shape.
    """
    # Split into two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Rotate: [-x2, x1]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """
    Apply Rotary Position Embedding to input tensor.

    The transformation is:
        x_rotated = x * cos + rotate_half(x) * sin

    Args:
        x: Input tensor [batch, num_heads, seq_len, head_dim].
        cos: Cosine values [batch, seq_len, head_dim] or broadcastable.
        sin: Sine values [batch, seq_len, head_dim] or broadcastable.

    Returns:
        Rotated tensor with same shape as input.
    """
    # Ensure cos/sin have the right shape for broadcasting
    # x: [batch, num_heads, seq_len, head_dim]
    # cos/sin: [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    # Apply rotation: x * cos + rotate_half(x) * sin
    return x * cos + rotate_half(x) * sin


def apply_rope_to_qk(
    query: Tensor,
    key: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Apply RoPE to both query and key tensors.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim].
        key: Key tensor [batch, num_kv_heads, seq_len, head_dim].
        cos: Cosine values [batch, seq_len, head_dim].
        sin: Sine values [batch, seq_len, head_dim].

    Returns:
        Tuple of (rotated_query, rotated_key).
    """
    query_rotated = apply_rope(query, cos, sin)
    key_rotated = apply_rope(key, cos, sin)
    return query_rotated, key_rotated


class RotaryEmbedding:
    """
    Rotary Position Embedding module.

    This class pre-computes and caches RoPE frequencies for efficiency.
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 1000000.0,
        max_position_embeddings: int = 32768,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize RotaryEmbedding.

        Args:
            head_dim: Dimension of each attention head.
            rope_theta: Base frequency for RoPE.
            max_position_embeddings: Maximum sequence length to cache.
            device: Device for computation.
            dtype: Data type for computation.
        """
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.device = device
        self.dtype = dtype

        # Cache inverse frequencies
        self._inv_freq = compute_rope_frequencies(
            head_dim, rope_theta, device=device, dtype=dtype
        )

        # Pre-compute cos/sin for positions up to max_position_embeddings
        self._cached_cos: Optional[Tensor] = None
        self._cached_sin: Optional[Tensor] = None
        self._cached_seq_len = 0

    def _update_cache(self, seq_len: int, device: torch.device) -> None:
        """Update cos/sin cache if needed."""
        if seq_len <= self._cached_seq_len and self._cached_cos is not None:
            return

        # Compute for all positions up to seq_len
        position_ids = torch.arange(seq_len, device=device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0)  # [1, seq_len]

        cos, sin = compute_rope_cos_sin(
            position_ids,
            self.head_dim,
            self.rope_theta,
            dtype=self.dtype,
        )

        self._cached_cos = cos.squeeze(0)  # [seq_len, head_dim]
        self._cached_sin = sin.squeeze(0)  # [seq_len, head_dim]
        self._cached_seq_len = seq_len

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply RoPE to query and key.

        Args:
            query: Query tensor [batch, num_heads, seq_len, head_dim].
            key: Key tensor [batch, num_kv_heads, seq_len, head_dim].
            position_ids: Optional position indices [batch, seq_len].
                         If None, uses sequential positions [0, 1, 2, ...].

        Returns:
            Tuple of (rotated_query, rotated_key).
        """
        batch_size, _, seq_len, _ = query.shape
        device = query.device

        if position_ids is None:
            # Use cached cos/sin for sequential positions
            self._update_cache(seq_len, device)
            cos = self._cached_cos[:seq_len]  # [seq_len, head_dim]
            sin = self._cached_sin[:seq_len]

            # Expand for batch
            cos = cos.unsqueeze(0)  # [1, seq_len, head_dim]
            sin = sin.unsqueeze(0)
        else:
            # Compute cos/sin for specific positions
            cos, sin = compute_rope_cos_sin(
                position_ids,
                self.head_dim,
                self.rope_theta,
                dtype=self.dtype,
            )

        return apply_rope_to_qk(query, key, cos, sin)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
