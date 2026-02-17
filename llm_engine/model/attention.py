"""
Attention mechanisms for Qwen3.

This module implements:
- FlashAttention varlen for prefill (variable-length packed sequences)
- PagedAttention for decode (block-based KV cache with PyTorch gather)
- Qwen3Attention module combining both

References:
- FlashAttention: https://arxiv.org/abs/2205.14135
- PagedAttention: https://arxiv.org/abs/2309.06180
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from llm_engine.model.layers import RMSNorm
from llm_engine.model.rope import apply_rope_to_qk


def flash_attention_prefill(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
) -> Tensor:
    """
    FlashAttention for prefill with variable-length sequences.

    Uses flash_attn_varlen_func which efficiently handles packed sequences
    of different lengths without padding overhead.

    Args:
        query: Query tensor [total_tokens, num_heads, head_dim].
        key: Key tensor [total_tokens, num_kv_heads, head_dim].
        value: Value tensor [total_tokens, num_kv_heads, head_dim].
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1].
        cu_seqlens_k: Cumulative sequence lengths for keys [batch_size + 1].
        max_seqlen_q: Maximum query sequence length in batch.
        max_seqlen_k: Maximum key sequence length in batch.
        softmax_scale: Scale factor for attention (default: 1/sqrt(head_dim)).
        causal: Whether to apply causal masking.

    Returns:
        Output tensor [total_tokens, num_heads, head_dim].
    """
    from flash_attn import flash_attn_varlen_func

    head_dim = query.shape[-1]
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Handle GQA: expand KV heads to match query heads
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]

    if num_heads != num_kv_heads:
        # GQA: repeat KV heads
        n_rep = num_heads // num_kv_heads
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    output = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        softmax_scale=softmax_scale,
    )

    return output


def paged_attention_decode(
    query: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    softmax_scale: Optional[float] = None,
) -> Tensor:
    """
    PagedAttention for decode phase using PyTorch gather operations.

    During decode, each sequence generates one new token and needs to
    attend to all previous tokens stored in block-based KV cache.

    Args:
        query: Query tensor [batch, num_heads, head_dim].
        k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim].
        v_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim].
        block_tables: Block indices per sequence [batch, max_blocks_per_seq].
        context_lens: Number of tokens in context for each sequence [batch].
        softmax_scale: Scale factor (default: 1/sqrt(head_dim)).

    Returns:
        Output tensor [batch, num_heads, head_dim].
    """
    batch_size, num_heads, head_dim = query.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    max_blocks_per_seq = block_tables.shape[1]
    max_context_len = max_blocks_per_seq * block_size

    # Step 1: Gather KV from block tables
    # Handle invalid block indices (-1 or out of range) by clamping
    valid_blocks = block_tables.clamp(min=0)  # [batch, max_blocks]

    # Gather key/value for each sequence
    # k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    # valid_blocks: [batch, max_blocks]
    # Result: [batch, max_blocks, block_size, num_kv_heads, head_dim]
    k_gathered = k_cache[valid_blocks]
    v_gathered = v_cache[valid_blocks]

    # Step 2: Reshape to [batch, max_context_len, num_kv_heads, head_dim]
    k_flat = k_gathered.reshape(batch_size, max_context_len, num_kv_heads, head_dim)
    v_flat = v_gathered.reshape(batch_size, max_context_len, num_kv_heads, head_dim)

    # Step 3: Expand KV heads for GQA
    if num_heads != num_kv_heads:
        n_rep = num_heads // num_kv_heads
        k_flat = k_flat.repeat_interleave(n_rep, dim=2)
        v_flat = v_flat.repeat_interleave(n_rep, dim=2)

    # Step 4: Transpose for attention computation
    # k_flat: [batch, max_context_len, num_heads, head_dim]
    #      -> [batch, num_heads, max_context_len, head_dim]
    k_flat = k_flat.transpose(1, 2)
    v_flat = v_flat.transpose(1, 2)

    # Step 5: Compute attention scores
    # query: [batch, num_heads, head_dim] -> [batch, num_heads, 1, head_dim]
    query = query.unsqueeze(2)

    # [batch, num_heads, 1, head_dim] @ [batch, num_heads, head_dim, max_context_len]
    # -> [batch, num_heads, 1, max_context_len]
    scores = torch.matmul(query, k_flat.transpose(-2, -1)) * softmax_scale

    # Step 6: Apply length mask (mask out positions beyond context_lens)
    # Create mask: [batch, max_context_len]
    positions = torch.arange(max_context_len, device=query.device)
    mask = positions.unsqueeze(0) >= context_lens.unsqueeze(1)  # [batch, max_context_len]

    # Expand mask for attention: [batch, 1, 1, max_context_len]
    mask = mask.unsqueeze(1).unsqueeze(2)
    scores = scores.masked_fill(mask, float("-inf"))

    # Step 7: Softmax and weighted sum
    attn_weights = F.softmax(scores, dim=-1)

    # [batch, num_heads, 1, max_context_len] @ [batch, num_heads, max_context_len, head_dim]
    # -> [batch, num_heads, 1, head_dim]
    output = torch.matmul(attn_weights, v_flat)

    # Remove the sequence dimension: [batch, num_heads, head_dim]
    output = output.squeeze(2)

    return output


def naive_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
) -> Tensor:
    """
    Naive attention implementation for testing and fallback.

    Args:
        query: [batch, num_heads, seq_len, head_dim]
        key: [batch, num_kv_heads, seq_len, head_dim]
        value: [batch, num_kv_heads, seq_len, head_dim]
        causal: Whether to apply causal mask.
        softmax_scale: Scale factor.

    Returns:
        Output [batch, num_heads, seq_len, head_dim].
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    num_kv_heads = key.shape[1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Handle GQA
    if num_heads != num_kv_heads:
        n_rep = num_heads // num_kv_heads
        key = key.repeat_interleave(n_rep, dim=1)
        value = value.repeat_interleave(n_rep, dim=1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale

    # Apply causal mask
    if causal:
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # Softmax and output
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output


class Qwen3Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) for Qwen3.

    Supports two modes:
    - Prefill: Uses FlashAttention varlen for efficient packed attention
    - Decode: Uses PagedAttention with block-based KV cache

    Args:
        hidden_size: Model hidden dimension.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of KV heads (< num_attention_heads for GQA).
        head_dim: Dimension per head (typically hidden_size // num_attention_heads).
        layer_idx: Index of this layer (for debugging).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx

        # GQA ratio
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # Projections
        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=False
        )

        # Q/K layer norms (Qwen3 uses RMSNorm on Q/K projections)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        # For prefill mode
        is_prefill: bool = True,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        # For decode mode
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
        block_tables: Optional[Tensor] = None,
        context_lens: Optional[Tensor] = None,
        # Cache management
        cache_positions: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass with support for both prefill and decode.

        Args:
            hidden_states: Input [batch, seq_len, hidden_size] or
                          [total_tokens, hidden_size] for packed prefill.
            position_embeddings: Tuple of (cos, sin) for RoPE.
            is_prefill: Whether this is prefill (True) or decode (False).
            cu_seqlens: Cumulative sequence lengths for packed prefill.
            max_seqlen: Maximum sequence length in batch.
            k_cache: Key cache for decode [num_blocks, block_size, num_kv_heads, head_dim].
            v_cache: Value cache for decode.
            block_tables: Block indices [batch, max_blocks].
            context_lens: Context lengths [batch].
            cache_positions: Positions for caching new KV values.

        Returns:
            Tuple of:
            - output: Attention output, same shape as input.
            - (key, value): New KV tensors for caching.
        """
        cos, sin = position_embeddings

        if is_prefill:
            return self._forward_prefill(
                hidden_states, cos, sin, cu_seqlens, max_seqlen
            )
        else:
            return self._forward_decode(
                hidden_states, cos, sin, k_cache, v_cache, block_tables, context_lens
            )

    def _forward_prefill(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Optional[Tensor],
        max_seqlen: Optional[int],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Prefill forward using FlashAttention.

        Args:
            hidden_states: [total_tokens, hidden_size] (packed) or
                          [batch, seq_len, hidden_size] (padded).
            cos, sin: RoPE embeddings.
            cu_seqlens: Cumulative sequence lengths [batch + 1].
            max_seqlen: Maximum sequence length.

        Returns:
            output and (key, value) for caching.
        """
        is_packed = hidden_states.dim() == 2

        if is_packed:
            total_tokens = hidden_states.shape[0]
            # Project Q, K, V
            q = self.q_proj(hidden_states)  # [total_tokens, num_heads * head_dim]
            k = self.k_proj(hidden_states)  # [total_tokens, num_kv_heads * head_dim]
            v = self.v_proj(hidden_states)

            # Reshape to [total_tokens, num_heads, head_dim]
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

            # Apply QK-norm before RoPE
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Apply RoPE
            # For packed sequences, cos/sin should be [total_tokens, head_dim]
            # We need to apply RoPE per-token
            q = self._apply_rope_packed(q, cos, sin)
            k = self._apply_rope_packed(k, cos, sin)

            # FlashAttention
            if cu_seqlens is None:
                raise ValueError("cu_seqlens required for packed prefill")

            output = flash_attention_prefill(
                q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
            )

            # Reshape output: [total_tokens, num_heads, head_dim] -> [total_tokens, hidden_size]
            output = output.reshape(total_tokens, self.num_heads * self.head_dim)
            output = self.o_proj(output)

        else:
            # Padded batch format
            batch_size, seq_len, _ = hidden_states.shape

            # Project
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            # Reshape to [batch, seq_len, num_heads, head_dim]
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

            # Apply QK-norm before RoPE
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Transpose to [batch, num_heads, seq_len, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Apply RoPE
            q, k = apply_rope_to_qk(q, k, cos, sin)

            # Expand KV heads for GQA before attention
            # Use expand+reshape to match HF's repeat_kv memory layout
            if self.num_heads != self.num_kv_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k = k[:, :, None, :, :].expand(
                    batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim
                ).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
                v = v[:, :, None, :, :].expand(
                    batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim
                ).reshape(batch_size, self.num_heads, seq_len, self.head_dim)

            # Use PyTorch SDPA (matches HuggingFace attention backend)
            # Only set is_causal=True for seq_len > 1 (matching HF behavior)
            output = F.scaled_dot_product_attention(
                q, k, v, is_causal=(seq_len > 1),
            )

            # Reshape output: [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, hidden_size]
            output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
            output = self.o_proj(output)

            # Also reshape k, v for return
            k = k.transpose(1, 2).reshape(batch_size, seq_len, -1)
            v = v.transpose(1, 2).reshape(batch_size, seq_len, -1)

        return output, (k, v)

    def _forward_decode(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        block_tables: Tensor,
        context_lens: Tensor,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Decode forward using PagedAttention.

        Args:
            hidden_states: [batch, 1, hidden_size] (single token per sequence).
            cos, sin: RoPE embeddings for current positions.
            k_cache, v_cache: Block-based KV cache.
            block_tables: Block indices [batch, max_blocks].
            context_lens: Number of KV tokens per sequence [batch].

        Returns:
            output and (new_key, new_value) for appending to cache.
        """
        batch_size = hidden_states.shape[0]

        # Handle both [batch, 1, hidden] and [batch, hidden] inputs
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(1)  # [batch, hidden_size]

        # Project Q, K, V for current token
        q = self.q_proj(hidden_states)  # [batch, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [batch, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)

        # Reshape
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, self.num_kv_heads, self.head_dim)

        # Apply QK-norm before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to Q and K
        # cos, sin should be [batch, 1, head_dim] or [batch, head_dim]
        if cos.dim() == 3:
            cos = cos.squeeze(1)
            sin = sin.squeeze(1)
        # Apply: q, k are [batch, num_heads, head_dim]
        # We need to handle this slightly differently
        q = self._apply_rope_decode(q, cos, sin)
        k = self._apply_rope_decode(k, cos, sin)

        # PagedAttention
        output = paged_attention_decode(
            query=q,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
        )

        # Output projection
        output = output.reshape(batch_size, self.num_heads * self.head_dim)
        output = self.o_proj(output)

        # Return output and new KV for cache update
        # Reshape k, v back to cache format
        return output, (k, v)

    def _apply_rope_packed(
        self, x: Tensor, cos: Tensor, sin: Tensor
    ) -> Tensor:
        """Apply RoPE for packed sequences."""
        # x: [total_tokens, num_heads, head_dim]
        # cos, sin: [total_tokens, head_dim] or similar

        # Ensure cos/sin are broadcastable
        if cos.dim() == 2:
            cos = cos.unsqueeze(1)  # [total_tokens, 1, head_dim]
            sin = sin.unsqueeze(1)

        # Split and rotate
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)

        return x * cos + rotated * sin

    def _apply_rope_decode(
        self, x: Tensor, cos: Tensor, sin: Tensor
    ) -> Tensor:
        """Apply RoPE for decode (single position)."""
        # x: [batch, num_heads, head_dim]
        # cos, sin: [batch, head_dim]

        # Expand for broadcasting
        if cos.dim() == 2:
            cos = cos.unsqueeze(1)  # [batch, 1, head_dim]
            sin = sin.unsqueeze(1)

        # Split and rotate
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)

        return x * cos + rotated * sin
