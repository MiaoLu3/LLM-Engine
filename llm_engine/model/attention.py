"""
Attention mechanisms for Qwen3.

This module implements:
- FlashAttention varlen for prefill (variable-length packed sequences)
- PagedAttention for decode (block-based KV cache with PyTorch gather)
- Qwen3Attention module with unified forward (Option C architecture)

The unified forward handles:
- Pure prefill (no KV cache)
- Chunked prefill (partial KV cache + new tokens)
- Decode (single token, full KV cache)
- Mixed batches (prefill + decode in same forward pass)

References:
- FlashAttention: https://arxiv.org/abs/2205.14135
- PagedAttention: https://arxiv.org/abs/2309.06180
- vLLM FlashInfer backend
"""

import math
from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from llm_engine.model.layers import RMSNorm
from llm_engine.model.rope import apply_rope_to_qk

if TYPE_CHECKING:
    from llm_engine.model.attention_metadata import AttentionMetadata


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


def write_to_kv_cache(
    key: Tensor,
    value: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    slot_mapping: Tensor,
) -> None:
    """
    Write K/V to cache at specified slots.

    Args:
        key: New keys [num_tokens, num_kv_heads, head_dim]
        value: New values [num_tokens, num_kv_heads, head_dim]
        k_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] - linear index into flattened cache
    """
    # Flatten cache for indexing: [num_blocks * block_size, num_kv_heads, head_dim]
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    k_cache_flat = k_cache.view(-1, num_kv_heads, head_dim)
    v_cache_flat = v_cache.view(-1, num_kv_heads, head_dim)

    # Write to cache
    k_cache_flat[slot_mapping] = key
    v_cache_flat[slot_mapping] = value


def gather_from_kv_cache(
    k_cache: Tensor,
    v_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Gather K/V from paged cache for chunked prefill.

    Args:
        k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        block_tables: [num_seqs, max_blocks] - block indices per sequence
        context_lens: [num_seqs] - number of cached tokens per sequence

    Returns:
        Tuple of (k_gathered, v_gathered) with shapes:
        [total_cached_tokens, num_kv_heads, head_dim]
    """
    num_seqs = block_tables.shape[0]
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape

    gathered_k = []
    gathered_v = []

    for i in range(num_seqs):
        ctx_len = context_lens[i].item()
        if ctx_len == 0:
            continue

        num_full_blocks = ctx_len // block_size
        remaining = ctx_len % block_size

        # Gather full blocks
        for b in range(num_full_blocks):
            block_idx = block_tables[i, b].item()
            gathered_k.append(k_cache[block_idx])  # [block_size, num_kv_heads, head_dim]
            gathered_v.append(v_cache[block_idx])

        # Gather partial block
        if remaining > 0:
            block_idx = block_tables[i, num_full_blocks].item()
            gathered_k.append(k_cache[block_idx, :remaining])
            gathered_v.append(v_cache[block_idx, :remaining])

    if gathered_k:
        return torch.cat(gathered_k, dim=0), torch.cat(gathered_v, dim=0)
    else:
        return (
            torch.empty(0, num_kv_heads, head_dim, device=k_cache.device, dtype=k_cache.dtype),
            torch.empty(0, num_kv_heads, head_dim, device=v_cache.device, dtype=v_cache.dtype),
        )


class Qwen3Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA) for Qwen3.

    Unified forward supporting:
    - Pure prefill (FlashAttention varlen, no cache)
    - Chunked prefill (SDPA with cached + new KV)
    - Decode (PagedAttention with full cache)
    - Mixed batches (split processing)

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
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> Tensor:
        """
        Unified forward pass handling all attention modes.

        Token layout in hidden_states (when using metadata):
            [<-- prefill tokens -->|<-- decode tokens -->]

        Args:
            hidden_states: [total_tokens, hidden_size] for packed input, or
                          [batch, seq_len, hidden_size] for padded input.
            position_embeddings: Tuple of (cos, sin) for RoPE.
            kv_cache: Optional tuple of (k_cache, v_cache) for this layer.
                     Shape: [num_blocks, block_size, num_kv_heads, head_dim]
            attn_metadata: AttentionMetadata describing batch composition.
                          If None, falls back to legacy pure prefill mode.

        Returns:
            Output tensor, same shape as hidden_states.
        """
        cos, sin = position_embeddings

        # Legacy mode: no metadata, assume pure prefill
        if attn_metadata is None:
            return self._forward_legacy_prefill(hidden_states, cos, sin)

        # Unified mode with metadata
        return self._forward_unified(hidden_states, cos, sin, kv_cache, attn_metadata)

    def _forward_unified(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: Optional[Tuple[Tensor, Tensor]],
        attn_metadata: "AttentionMetadata",
    ) -> Tensor:
        """
        Unified forward with metadata-driven routing.

        Handles mixed prefill + decode batches by:
        1. Projecting all tokens together
        2. Writing new KV to cache
        3. Split-processing prefill and decode tokens
        4. Concatenating outputs
        """
        num_tokens = hidden_states.shape[0]
        num_prefill = attn_metadata.num_prefill_tokens
        num_decode = attn_metadata.num_decode_tokens

        # 1. Project all tokens
        q = self.q_proj(hidden_states)  # [num_tokens, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [num_tokens, num_kv_heads * head_dim]
        v = self.v_proj(hidden_states)

        # Reshape to [num_tokens, num_heads, head_dim]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # 2. Apply QK-norm before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. Apply RoPE
        q, k = self._apply_rope_packed(q, k, cos, sin)

        # 4. Write new KV to cache (if cache provided)
        if kv_cache is not None and attn_metadata.slot_mapping is not None:
            k_cache, v_cache = kv_cache
            write_to_kv_cache(k, v, k_cache, v_cache, attn_metadata.slot_mapping)

        # 5. Compute attention (split prefill and decode)
        output = torch.empty(
            num_tokens, self.num_heads * self.head_dim,
            device=hidden_states.device, dtype=hidden_states.dtype
        )

        if num_prefill > 0:
            q_prefill = q[:num_prefill]
            k_prefill = k[:num_prefill]
            v_prefill = v[:num_prefill]

            if attn_metadata.has_chunked_prefill and kv_cache is not None:
                # Chunked prefill: gather cached KV + use current KV
                output_prefill = self._attention_chunked_prefill(
                    q_prefill, k_prefill, v_prefill, kv_cache, attn_metadata
                )
            else:
                # Pure prefill: FlashAttention varlen
                output_prefill = self._attention_pure_prefill(
                    q_prefill, k_prefill, v_prefill, attn_metadata
                )

            output[:num_prefill] = output_prefill.view(num_prefill, -1)

        if num_decode > 0:
            q_decode = q[num_prefill:]

            if kv_cache is not None:
                k_cache, v_cache = kv_cache
                output_decode = self._attention_decode(
                    q_decode, k_cache, v_cache, attn_metadata
                )
                output[num_prefill:] = output_decode.view(num_decode, -1)

        # 6. Output projection
        return self.o_proj(output)

    def _attention_pure_prefill(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> Tensor:
        """Pure prefill using FlashAttention varlen."""
        try:
            output = flash_attention_prefill(
                q, k, v,
                attn_metadata.prefill_cu_seqlens_q,
                attn_metadata.prefill_cu_seqlens_kv,
                attn_metadata.max_prefill_seq_len,
                attn_metadata.max_prefill_seq_len,
            )
        except Exception:
            # Fallback to SDPA if FlashAttention not available
            output = self._attention_prefill_sdpa(q, k, v, attn_metadata)

        return output

    def _attention_prefill_sdpa(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> Tensor:
        """Prefill using PyTorch SDPA (fallback when FlashAttention unavailable)."""
        # Process each sequence separately
        outputs = []
        cu_seqlens = attn_metadata.prefill_cu_seqlens_q

        for i in range(attn_metadata.num_prefill_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seq_len = end - start

            q_seq = q[start:end].unsqueeze(0).transpose(1, 2)  # [1, heads, seq, dim]
            k_seq = k[start:end].unsqueeze(0).transpose(1, 2)
            v_seq = v[start:end].unsqueeze(0).transpose(1, 2)

            # GQA expansion
            if self.num_heads != self.num_kv_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k_seq = k_seq[:, :, None, :, :].expand(
                    1, self.num_kv_heads, n_rep, seq_len, self.head_dim
                ).reshape(1, self.num_heads, seq_len, self.head_dim)
                v_seq = v_seq[:, :, None, :, :].expand(
                    1, self.num_kv_heads, n_rep, seq_len, self.head_dim
                ).reshape(1, self.num_heads, seq_len, self.head_dim)

            out_seq = F.scaled_dot_product_attention(
                q_seq, k_seq, v_seq, is_causal=(seq_len > 1)
            )
            out_seq = out_seq.transpose(1, 2).squeeze(0)  # [seq, heads, dim]
            outputs.append(out_seq)

        return torch.cat(outputs, dim=0)

    def _attention_chunked_prefill(
        self,
        q: Tensor,
        k_new: Tensor,
        v_new: Tensor,
        kv_cache: Tuple[Tensor, Tensor],
        attn_metadata: "AttentionMetadata",
    ) -> Tensor:
        """
        Chunked prefill: Q attends to cached KV + new KV.

        For each sequence:
        1. Gather cached KV from paged cache
        2. Concatenate with new KV
        3. Run attention (SDPA)
        """
        k_cache, v_cache = kv_cache
        outputs = []

        cu_seqlens_q = attn_metadata.prefill_cu_seqlens_q
        context_lens = attn_metadata.prefill_context_lens
        block_tables = attn_metadata.prefill_block_tables

        for i in range(attn_metadata.num_prefill_seqs):
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i + 1].item()
            query_len = q_end - q_start
            ctx_len = context_lens[i].item()

            q_seq = q[q_start:q_end]  # [query_len, heads, dim]
            k_seq_new = k_new[q_start:q_end]
            v_seq_new = v_new[q_start:q_end]

            if ctx_len > 0 and block_tables is not None:
                # Gather cached KV
                k_cached, v_cached = self._gather_seq_kv(
                    k_cache, v_cache, block_tables[i], ctx_len
                )
                # Concat: [cached_len + query_len, kv_heads, dim]
                k_seq = torch.cat([k_cached, k_seq_new], dim=0)
                v_seq = torch.cat([v_cached, v_seq_new], dim=0)
            else:
                k_seq = k_seq_new
                v_seq = v_seq_new

            total_len = k_seq.shape[0]

            # Reshape for SDPA: [1, heads, seq, dim]
            q_seq = q_seq.unsqueeze(0).transpose(1, 2)
            k_seq = k_seq.unsqueeze(0).transpose(1, 2)
            v_seq = v_seq.unsqueeze(0).transpose(1, 2)

            # GQA expansion
            if self.num_heads != self.num_kv_heads:
                n_rep = self.num_heads // self.num_kv_heads
                k_seq = k_seq[:, :, None, :, :].expand(
                    1, self.num_kv_heads, n_rep, total_len, self.head_dim
                ).reshape(1, self.num_heads, total_len, self.head_dim)
                v_seq = v_seq[:, :, None, :, :].expand(
                    1, self.num_kv_heads, n_rep, total_len, self.head_dim
                ).reshape(1, self.num_heads, total_len, self.head_dim)

            # Build causal mask for chunked attention
            # Q can only attend to positions <= its position in the full sequence
            # Q positions: [ctx_len, ctx_len+1, ..., ctx_len+query_len-1]
            # K positions: [0, 1, ..., total_len-1]
            attn_mask = self._build_chunked_causal_mask(query_len, total_len, ctx_len, q.device)

            out_seq = F.scaled_dot_product_attention(
                q_seq, k_seq, v_seq, attn_mask=attn_mask
            )
            out_seq = out_seq.transpose(1, 2).squeeze(0)  # [query_len, heads, dim]
            outputs.append(out_seq)

        return torch.cat(outputs, dim=0)

    def _attention_decode(
        self,
        q: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> Tensor:
        """Decode using PagedAttention."""
        # q: [num_decode, num_heads, head_dim]
        return paged_attention_decode(
            q,
            k_cache,
            v_cache,
            attn_metadata.decode_block_tables,
            attn_metadata.decode_seq_lens,
        )

    def _gather_seq_kv(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        block_table: Tensor,
        context_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """Gather KV for a single sequence from paged cache."""
        block_size = k_cache.shape[1]
        num_full_blocks = context_len // block_size
        remaining = context_len % block_size

        k_parts = []
        v_parts = []

        for b in range(num_full_blocks):
            block_idx = block_table[b].item()
            k_parts.append(k_cache[block_idx])
            v_parts.append(v_cache[block_idx])

        if remaining > 0:
            block_idx = block_table[num_full_blocks].item()
            k_parts.append(k_cache[block_idx, :remaining])
            v_parts.append(v_cache[block_idx, :remaining])

        return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)

    def _build_chunked_causal_mask(
        self,
        query_len: int,
        kv_len: int,
        context_len: int,
        device: torch.device,
    ) -> Tensor:
        """
        Build causal mask for chunked prefill.

        Q positions: [context_len, context_len+1, ..., context_len+query_len-1]
        K positions: [0, 1, ..., kv_len-1]

        Q[i] can attend to K[j] iff j <= context_len + i
        """
        # Q absolute positions
        q_pos = torch.arange(context_len, context_len + query_len, device=device)
        # K positions
        k_pos = torch.arange(kv_len, device=device)

        # Causal: q_pos[i] >= k_pos[j]
        mask = q_pos.unsqueeze(1) >= k_pos.unsqueeze(0)  # [query_len, kv_len]

        # Convert to attention mask (0 = attend, -inf = mask)
        attn_mask = torch.zeros(query_len, kv_len, device=device, dtype=torch.float32)
        attn_mask.masked_fill_(~mask, float("-inf"))

        return attn_mask

    def _apply_rope_packed(
        self,
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Apply RoPE for packed sequences."""
        # q, k: [total_tokens, num_heads, head_dim]
        # cos, sin: [total_tokens, head_dim] or [total_tokens, 1, head_dim]

        if cos.dim() == 2:
            cos = cos.unsqueeze(1)  # [total_tokens, 1, head_dim]
            sin = sin.unsqueeze(1)

        # Split and rotate
        q1 = q[..., : self.head_dim // 2]
        q2 = q[..., self.head_dim // 2:]
        q_rotated = torch.cat([-q2, q1], dim=-1)
        q_out = q * cos + q_rotated * sin

        k1 = k[..., : self.head_dim // 2]
        k2 = k[..., self.head_dim // 2:]
        k_rotated = torch.cat([-k2, k1], dim=-1)
        k_out = k * cos[:, :self.num_kv_heads, :] if cos.shape[1] > 1 else k * cos + k_rotated * sin

        # Actually apply to k properly
        k_out = k * cos + k_rotated * sin

        return q_out, k_out

    def _forward_legacy_prefill(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """
        Legacy pure prefill mode (for backward compatibility and HF comparison).

        Handles both packed [total_tokens, hidden] and batched [batch, seq, hidden].
        """
        is_packed = hidden_states.dim() == 2

        if is_packed:
            return self._forward_legacy_packed(hidden_states, cos, sin)
        else:
            return self._forward_legacy_batched(hidden_states, cos, sin)

    def _forward_legacy_packed(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """Legacy packed prefill."""
        total_tokens = hidden_states.shape[0]

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(total_tokens, self.num_heads, self.head_dim)
        k = k.view(total_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(total_tokens, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self._apply_rope_packed(q, k, cos, sin)

        # Use SDPA for single sequence
        q = q.unsqueeze(0).transpose(1, 2)  # [1, heads, seq, dim]
        k = k.unsqueeze(0).transpose(1, 2)
        v = v.unsqueeze(0).transpose(1, 2)

        # GQA expansion
        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k[:, :, None, :, :].expand(
                1, self.num_kv_heads, n_rep, total_tokens, self.head_dim
            ).reshape(1, self.num_heads, total_tokens, self.head_dim)
            v = v[:, :, None, :, :].expand(
                1, self.num_kv_heads, n_rep, total_tokens, self.head_dim
            ).reshape(1, self.num_heads, total_tokens, self.head_dim)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=(total_tokens > 1))
        output = output.transpose(1, 2).squeeze(0)  # [seq, heads, dim]
        output = output.reshape(total_tokens, -1)

        return self.o_proj(output)

    def _forward_legacy_batched(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> Tensor:
        """Legacy batched prefill (for HF comparison)."""
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope_to_qk(q, k, cos, sin)

        # GQA expansion
        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k[:, :, None, :, :].expand(
                batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim
            ).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
            v = v[:, :, None, :, :].expand(
                batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim
            ).reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=(seq_len > 1))
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)

        return self.o_proj(output)
