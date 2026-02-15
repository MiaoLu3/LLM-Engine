"""
Block and BlockTable data structures for KV cache management.

A Block represents a fixed-size unit of KV cache storage (e.g., 16 tokens).
BlockTable maps logical blocks to physical blocks for each sequence.

This implements the PagedAttention concept from vLLM:
- KV cache is divided into fixed-size blocks
- Sequences can use non-contiguous blocks
- Blocks can be shared (for prefix caching)
- Reference counting for block lifecycle
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class BlockStatus(Enum):
    """Status of a physical block in the block pool."""

    FREE = "free"
    """Block is available for allocation."""

    ALLOCATED = "allocated"
    """Block is in use by one or more sequences."""

    CACHED = "cached"
    """Block is cached (ref_count=0) but holds valid KV data for prefix caching.
    Can be evicted if memory is needed."""


@dataclass
class PhysicalBlock:
    """
    A physical block in GPU memory holding KV cache data.

    Each block stores KV cache for `block_size` tokens.
    Multiple sequences can reference the same block (prefix caching).
    """

    block_id: int
    """Unique identifier for this physical block."""

    block_size: int
    """Number of tokens this block can hold."""

    ref_count: int = 0
    """Number of sequences currently referencing this block.
    Block is FREE when ref_count == 0 and no prefix caching.
    Block is CACHED when ref_count == 0 but has cached content."""

    status: BlockStatus = BlockStatus.FREE
    """Current status of the block."""

    # Prefix caching metadata
    content_hash: Optional[int] = None
    """Hash of the token content in this block.
    Used for prefix cache lookup. None if not cacheable."""

    last_accessed: float = 0.0
    """Timestamp of last access. Used for LRU eviction."""

    num_tokens: int = 0
    """Number of valid tokens in this block (may be < block_size for last block)."""

    def allocate(self) -> None:
        """Mark block as allocated and increment ref count."""
        self.ref_count += 1
        self.status = BlockStatus.ALLOCATED

    def release(self, keep_cached: bool = False) -> None:
        """
        Decrement reference count and potentially free the block.

        Args:
            keep_cached: If True and ref_count becomes 0, mark as CACHED
                        instead of FREE (for prefix caching).
        """
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.ref_count = 0
            if keep_cached and self.content_hash is not None:
                self.status = BlockStatus.CACHED
            else:
                self.status = BlockStatus.FREE
                self.content_hash = None

    def free(self) -> None:
        """Completely free the block, clearing all metadata."""
        self.ref_count = 0
        self.status = BlockStatus.FREE
        self.content_hash = None
        self.num_tokens = 0

    @property
    def is_free(self) -> bool:
        """Check if block can be allocated."""
        return self.status == BlockStatus.FREE

    @property
    def is_cached(self) -> bool:
        """Check if block is in cached state (evictable)."""
        return self.status == BlockStatus.CACHED

    @property
    def is_allocated(self) -> bool:
        """Check if block is currently in use."""
        return self.status == BlockStatus.ALLOCATED

    @property
    def is_evictable(self) -> bool:
        """Check if block can be evicted (cached with ref_count=0)."""
        return self.status == BlockStatus.CACHED and self.ref_count == 0

    def __repr__(self) -> str:
        return (
            f"PhysicalBlock(id={self.block_id}, ref={self.ref_count}, "
            f"status={self.status.value}, tokens={self.num_tokens}/{self.block_size})"
        )


@dataclass
class LogicalBlock:
    """
    A logical block in a sequence's block table.

    Each sequence has a list of logical blocks that map to physical blocks.
    This indirection enables:
    - Non-contiguous physical allocation
    - Copy-on-write for shared prefixes
    - Block sharing across sequences
    """

    logical_idx: int
    """Index of this block in the sequence's block table."""

    physical_block_id: Optional[int] = None
    """ID of the physical block this maps to. None if not yet allocated."""

    num_tokens: int = 0
    """Number of valid tokens in this logical block."""

    @property
    def is_allocated(self) -> bool:
        """Check if this logical block has a physical block assigned."""
        return self.physical_block_id is not None

    def __repr__(self) -> str:
        return f"LogicalBlock(idx={self.logical_idx}, physical={self.physical_block_id})"


@dataclass
class BlockTable:
    """
    Maps logical blocks to physical blocks for a sequence.

    This is the sequence's view of its KV cache:
    - Each entry maps a logical block index to a physical block ID
    - The scheduler uses this to build attention inputs
    - Supports non-contiguous physical allocation
    """

    seq_id: int
    """ID of the sequence this block table belongs to."""

    block_size: int
    """Number of tokens per block."""

    blocks: list[LogicalBlock] = field(default_factory=list)
    """List of logical blocks in order."""

    def __len__(self) -> int:
        """Number of logical blocks."""
        return len(self.blocks)

    def get_physical_blocks(self) -> list[int]:
        """Get list of physical block IDs for all allocated logical blocks."""
        return [
            block.physical_block_id
            for block in self.blocks
            if block.physical_block_id is not None
        ]

    def get_block_for_token(self, token_idx: int) -> Optional[LogicalBlock]:
        """Get the logical block containing a specific token index."""
        block_idx = token_idx // self.block_size
        if block_idx < len(self.blocks):
            return self.blocks[block_idx]
        return None

    def num_blocks_needed(self, num_tokens: int) -> int:
        """Calculate number of blocks needed for a given number of tokens."""
        return (num_tokens + self.block_size - 1) // self.block_size

    def num_allocated_blocks(self) -> int:
        """Count blocks that have physical allocation."""
        return sum(1 for block in self.blocks if block.is_allocated)

    def allocate_logical_block(self) -> LogicalBlock:
        """Add a new logical block (physical allocation done separately)."""
        new_block = LogicalBlock(logical_idx=len(self.blocks))
        self.blocks.append(new_block)
        return new_block

    def assign_physical_block(self, logical_idx: int, physical_block_id: int) -> None:
        """Assign a physical block to a logical block."""
        if logical_idx < len(self.blocks):
            self.blocks[logical_idx].physical_block_id = physical_block_id

    def clear(self) -> None:
        """Clear all block mappings (for sequence reset)."""
        self.blocks.clear()

    def to_tensor_format(self, max_blocks: int) -> list[int]:
        """
        Convert to tensor format for model input.

        Args:
            max_blocks: Pad to this length for batching.

        Returns:
            List of physical block IDs, padded with -1.
        """
        result = [
            block.physical_block_id if block.physical_block_id is not None else -1
            for block in self.blocks
        ]
        # Pad to max_blocks
        while len(result) < max_blocks:
            result.append(-1)
        return result[:max_blocks]

    def __repr__(self) -> str:
        physical = self.get_physical_blocks()
        return f"BlockTable(seq={self.seq_id}, blocks={physical})"


def compute_num_blocks(num_tokens: int, block_size: int) -> int:
    """
    Compute number of blocks needed for a given number of tokens.

    Args:
        num_tokens: Total tokens to store.
        block_size: Tokens per block.

    Returns:
        Number of blocks needed (ceiling division).
    """
    return (num_tokens + block_size - 1) // block_size


def compute_block_hash(token_ids: list[int], block_idx: int, block_size: int) -> int:
    """
    Compute content hash for a block (for prefix caching).

    The hash includes:
    - Token IDs in the block
    - Block index (position matters)

    This ensures blocks with same content at same position hash identically.

    Args:
        token_ids: All token IDs in the sequence.
        block_idx: Index of this block (0-indexed).
        block_size: Number of tokens per block.

    Returns:
        Hash value for this block.
    """
    start = block_idx * block_size
    end = min(start + block_size, len(token_ids))
    block_tokens = tuple(token_ids[start:end])

    # Include block index in hash to distinguish position
    return hash((block_idx, block_tokens))
