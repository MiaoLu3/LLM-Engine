"""
Attention metadata for unified prefill/decode forward pass.

This module provides:
- AttentionMetadata: Dataclass describing batch composition
- AttentionMetadataBuilder: Helper to construct metadata from sequences

The metadata enables a single forward() call to handle:
- Pure prefill (no KV cache)
- Chunked prefill (partial KV cache)
- Decode (full KV cache, single token)
- Mixed batches (prefill + decode tokens together)

Reference: vLLM's FlashInfer attention backend
"""

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from llm_engine.data_structures.sequence import Sequence


@dataclass
class AttentionMetadata:
    """
    Metadata for unified attention computation.

    Describes how tokens in a batch should be processed:
    - Which are prefill vs decode tokens
    - Where to read cached KV from
    - Where to write new KV to

    Token layout in batch (after reordering):
        [<-- prefill tokens -->|<-- decode tokens -->]

    Attributes:
        num_prefill_tokens: Total prefill tokens in batch
        num_decode_tokens: Total decode tokens in batch (usually = num_decode_seqs)
        num_prefill_seqs: Number of sequences in prefill phase
        num_decode_seqs: Number of sequences in decode phase

        # For prefill (FlashAttention varlen)
        prefill_seq_lens: [num_prefill_seqs] - query length per prefill seq
        prefill_context_lens: [num_prefill_seqs] - cached KV length (0 for pure prefill)
        prefill_cu_seqlens_q: [num_prefill_seqs + 1] - cumulative query lengths
        prefill_cu_seqlens_kv: [num_prefill_seqs + 1] - cumulative KV lengths
        max_prefill_seq_len: Maximum query length in prefill batch
        prefill_block_tables: [num_prefill_seqs, max_blocks] - for chunked prefill

        # For decode (PagedAttention)
        decode_seq_lens: [num_decode_seqs] - total context length per decode seq
        decode_block_tables: [num_decode_seqs, max_blocks] - block indices

        # KV cache write positions
        slot_mapping: [num_tokens] - where to write new KV in cache

        # Device
        device: torch.device
    """

    # Batch composition
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefill_seqs: int = 0
    num_decode_seqs: int = 0

    # Prefill metadata
    prefill_seq_lens: Optional[Tensor] = None
    prefill_context_lens: Optional[Tensor] = None
    prefill_cu_seqlens_q: Optional[Tensor] = None
    prefill_cu_seqlens_kv: Optional[Tensor] = None
    max_prefill_seq_len: int = 0
    prefill_block_tables: Optional[Tensor] = None

    # Decode metadata
    decode_seq_lens: Optional[Tensor] = None
    decode_block_tables: Optional[Tensor] = None

    # KV cache positions
    slot_mapping: Optional[Tensor] = None

    # Device
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    @property
    def num_tokens(self) -> int:
        """Total tokens in batch."""
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def has_prefill(self) -> bool:
        """Whether batch contains prefill tokens."""
        return self.num_prefill_tokens > 0

    @property
    def has_decode(self) -> bool:
        """Whether batch contains decode tokens."""
        return self.num_decode_tokens > 0

    @property
    def is_pure_prefill(self) -> bool:
        """Whether batch is prefill-only with no cached KV."""
        if not self.has_prefill or self.has_decode:
            return False
        if self.prefill_context_lens is None:
            return True
        return self.prefill_context_lens.sum().item() == 0

    @property
    def is_pure_decode(self) -> bool:
        """Whether batch is decode-only."""
        return self.has_decode and not self.has_prefill

    @property
    def has_chunked_prefill(self) -> bool:
        """Whether any prefill sequence has cached KV (chunked prefill)."""
        if not self.has_prefill:
            return False
        if self.prefill_context_lens is None:
            return False
        return self.prefill_context_lens.sum().item() > 0


class AttentionMetadataBuilder:
    """
    Builds AttentionMetadata from scheduled sequences.

    Used by the scheduler to construct metadata for each forward pass.

    Args:
        block_size: Tokens per KV cache block
        max_blocks_per_seq: Maximum blocks per sequence
        device: Device for tensors
    """

    def __init__(
        self,
        block_size: int,
        max_blocks_per_seq: int,
        device: torch.device = torch.device("cuda"),
    ):
        self.block_size = block_size
        self.max_blocks_per_seq = max_blocks_per_seq
        self.device = device

    def build(
        self,
        prefill_seqs: List["Sequence"],
        decode_seqs: List["Sequence"],
        prefill_query_lens: List[int],
        prefill_context_lens: List[int],
        decode_context_lens: List[int],
        prefill_block_tables: List[List[int]],
        decode_block_tables: List[List[int]],
        slot_mapping: List[int],
    ) -> AttentionMetadata:
        """
        Build metadata from sequence lists.

        Args:
            prefill_seqs: Sequences in prefill phase
            decode_seqs: Sequences in decode phase
            prefill_query_lens: Query length per prefill seq (chunk size)
            prefill_context_lens: Cached KV length per prefill seq
            decode_context_lens: Total context per decode seq
            prefill_block_tables: Block indices per prefill seq
            decode_block_tables: Block indices per decode seq
            slot_mapping: Cache write position per token

        Returns:
            AttentionMetadata for forward pass
        """
        num_prefill_seqs = len(prefill_seqs)
        num_decode_seqs = len(decode_seqs)
        num_prefill_tokens = sum(prefill_query_lens)
        num_decode_tokens = num_decode_seqs  # 1 token per decode seq

        # Build prefill metadata
        prefill_seq_lens_t = None
        prefill_context_lens_t = None
        prefill_cu_seqlens_q = None
        prefill_cu_seqlens_kv = None
        max_prefill_seq_len = 0
        prefill_block_tables_t = None

        if num_prefill_seqs > 0:
            prefill_seq_lens_t = torch.tensor(
                prefill_query_lens, dtype=torch.int32, device=self.device
            )
            prefill_context_lens_t = torch.tensor(
                prefill_context_lens, dtype=torch.int32, device=self.device
            )

            # Cumulative sequence lengths for FlashAttention varlen
            cu_q = [0]
            cu_kv = [0]
            for q_len, ctx_len in zip(prefill_query_lens, prefill_context_lens):
                cu_q.append(cu_q[-1] + q_len)
                cu_kv.append(cu_kv[-1] + ctx_len + q_len)

            prefill_cu_seqlens_q = torch.tensor(
                cu_q, dtype=torch.int32, device=self.device
            )
            prefill_cu_seqlens_kv = torch.tensor(
                cu_kv, dtype=torch.int32, device=self.device
            )

            max_prefill_seq_len = max(prefill_query_lens)

            # Block tables for chunked prefill
            if any(prefill_context_lens):
                prefill_block_tables_t = self._pad_block_tables(prefill_block_tables)

        # Build decode metadata
        decode_seq_lens_t = None
        decode_block_tables_t = None

        if num_decode_seqs > 0:
            decode_seq_lens_t = torch.tensor(
                decode_context_lens, dtype=torch.int32, device=self.device
            )
            decode_block_tables_t = self._pad_block_tables(decode_block_tables)

        # Slot mapping
        slot_mapping_t = torch.tensor(
            slot_mapping, dtype=torch.int64, device=self.device
        )

        return AttentionMetadata(
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefill_seqs=num_prefill_seqs,
            num_decode_seqs=num_decode_seqs,
            prefill_seq_lens=prefill_seq_lens_t,
            prefill_context_lens=prefill_context_lens_t,
            prefill_cu_seqlens_q=prefill_cu_seqlens_q,
            prefill_cu_seqlens_kv=prefill_cu_seqlens_kv,
            max_prefill_seq_len=max_prefill_seq_len,
            prefill_block_tables=prefill_block_tables_t,
            decode_seq_lens=decode_seq_lens_t,
            decode_block_tables=decode_block_tables_t,
            slot_mapping=slot_mapping_t,
            device=self.device,
        )

    def _pad_block_tables(self, block_tables: List[List[int]]) -> Tensor:
        """Pad block tables to uniform length."""
        if not block_tables:
            return None

        max_len = max(len(bt) for bt in block_tables)
        max_len = max(max_len, 1)  # At least 1

        padded = []
        for bt in block_tables:
            padded_bt = bt + [0] * (max_len - len(bt))
            padded.append(padded_bt)

        return torch.tensor(padded, dtype=torch.int32, device=self.device)


def create_empty_metadata(device: torch.device = torch.device("cuda")) -> AttentionMetadata:
    """Create empty metadata (for testing)."""
    return AttentionMetadata(device=device)


def create_prefill_metadata(
    seq_lens: List[int],
    device: torch.device = torch.device("cuda"),
) -> AttentionMetadata:
    """
    Create metadata for pure prefill (no KV cache).

    Args:
        seq_lens: Length of each sequence
        device: Device for tensors

    Returns:
        AttentionMetadata for prefill-only batch
    """
    num_seqs = len(seq_lens)
    total_tokens = sum(seq_lens)

    # Cumulative lengths
    cu_seqlens = [0]
    for length in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + length)

    return AttentionMetadata(
        num_prefill_tokens=total_tokens,
        num_decode_tokens=0,
        num_prefill_seqs=num_seqs,
        num_decode_seqs=0,
        prefill_seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        prefill_context_lens=torch.zeros(num_seqs, dtype=torch.int32, device=device),
        prefill_cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        prefill_cu_seqlens_kv=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        max_prefill_seq_len=max(seq_lens) if seq_lens else 0,
        prefill_block_tables=None,
        decode_seq_lens=None,
        decode_block_tables=None,
        slot_mapping=None,
        device=device,
    )


def create_decode_metadata(
    context_lens: List[int],
    block_tables: List[List[int]],
    slot_mapping: List[int],
    device: torch.device = torch.device("cuda"),
) -> AttentionMetadata:
    """
    Create metadata for decode-only batch.

    Args:
        context_lens: Total context length per sequence
        block_tables: Block indices per sequence
        slot_mapping: Cache write position per token
        device: Device for tensors

    Returns:
        AttentionMetadata for decode-only batch
    """
    num_seqs = len(context_lens)

    # Pad block tables
    max_blocks = max(len(bt) for bt in block_tables) if block_tables else 1
    padded_bt = []
    for bt in block_tables:
        padded_bt.append(bt + [0] * (max_blocks - len(bt)))

    return AttentionMetadata(
        num_prefill_tokens=0,
        num_decode_tokens=num_seqs,
        num_prefill_seqs=0,
        num_decode_seqs=num_seqs,
        prefill_seq_lens=None,
        prefill_context_lens=None,
        prefill_cu_seqlens_q=None,
        prefill_cu_seqlens_kv=None,
        max_prefill_seq_len=0,
        prefill_block_tables=None,
        decode_seq_lens=torch.tensor(context_lens, dtype=torch.int32, device=device),
        decode_block_tables=torch.tensor(padded_bt, dtype=torch.int32, device=device),
        slot_mapping=torch.tensor(slot_mapping, dtype=torch.int64, device=device),
        device=device,
    )
