"""
Block Manager for KV cache allocation.

This module manages the allocation and deallocation of physical blocks
in the KV cache. It implements:

1. Block allocation: Assign free blocks to sequences
2. Block deallocation: Return blocks when sequences complete
3. Copy-on-write: Duplicate blocks when sequences diverge
4. Slot mapping: Compute cache write positions for new tokens

The BlockManager is used by the Scheduler to:
- Check if memory is available for new sequences
- Allocate blocks for prefill and decode
- Handle memory pressure through preemption

Reference: vLLM BlockSpaceManager
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import time

from llm_engine.data_structures.block import (
    PhysicalBlock,
    BlockTable,
    LogicalBlock,
    BlockStatus,
    compute_num_blocks,
)


@dataclass
class BlockManagerConfig:
    """Configuration for BlockManager."""

    num_blocks: int
    """Total number of physical blocks available."""

    block_size: int = 16
    """Number of tokens per block."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching (cached blocks not immediately freed)."""


class BlockManager:
    """
    Manages physical block allocation for sequences.

    Responsibilities:
    - Track free/allocated/cached blocks
    - Allocate blocks for new tokens
    - Free blocks when sequences complete
    - Handle copy-on-write for beam search
    - Compute slot mapping for KV cache writes

    Block States:
    - FREE: Available for allocation
    - ALLOCATED: In use by one or more sequences
    - CACHED: ref_count=0 but holds valid KV (for prefix caching)

    Example:
        >>> manager = BlockManager(BlockManagerConfig(num_blocks=1000))
        >>> seq_id = 1
        >>> blocks = manager.allocate_blocks(seq_id, num_blocks=3)
        >>> slots = manager.get_slot_mapping(seq_id, context_len=0, num_tokens=48)
        >>> manager.free_sequence(seq_id)
    """

    def __init__(self, config: BlockManagerConfig):
        """
        Initialize block manager with a pool of physical blocks.

        Args:
            config: Block manager configuration
        """
        self.config = config
        self.num_blocks = config.num_blocks
        self.block_size = config.block_size
        self.enable_prefix_caching = config.enable_prefix_caching

        # Physical block pool
        self.blocks: List[PhysicalBlock] = [
            PhysicalBlock(block_id=i, block_size=self.block_size)
            for i in range(self.num_blocks)
        ]

        # Track free block indices
        self.free_blocks: Set[int] = set(range(self.num_blocks))

        # Sequence ID -> BlockTable mapping
        self.block_tables: Dict[int, BlockTable] = {}

        # For prefix caching: hash -> block_id
        self.prefix_cache: Dict[int, int] = {}

        # Statistics
        self.num_allocated = 0
        self.num_cached = 0

    def get_num_free_blocks(self) -> int:
        """Get number of free blocks available for allocation."""
        return len(self.free_blocks)

    def get_num_allocated_blocks(self) -> int:
        """Get number of currently allocated blocks."""
        return self.num_allocated

    def can_allocate(self, num_blocks: int) -> bool:
        """
        Check if we can allocate the requested number of blocks.

        Args:
            num_blocks: Number of blocks needed

        Returns:
            True if allocation is possible
        """
        return len(self.free_blocks) >= num_blocks

    def can_allocate_tokens(self, seq_id: int, num_new_tokens: int) -> bool:
        """
        Check if we can allocate blocks for new tokens.

        Args:
            seq_id: Sequence ID
            num_new_tokens: Number of new tokens to be added

        Returns:
            True if allocation is possible
        """
        if seq_id not in self.block_tables:
            # New sequence: need full allocation
            num_blocks = compute_num_blocks(num_new_tokens, self.block_size)
            return self.can_allocate(num_blocks)

        # Existing sequence: check if current block has space
        block_table = self.block_tables[seq_id]
        current_tokens = self._get_context_len(seq_id)
        new_total = current_tokens + num_new_tokens

        current_blocks = len(block_table)
        needed_blocks = compute_num_blocks(new_total, self.block_size)

        new_blocks_needed = max(0, needed_blocks - current_blocks)
        return self.can_allocate(new_blocks_needed)

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> List[int]:
        """
        Allocate blocks for a new sequence (prefill).

        Args:
            seq_id: Unique sequence identifier
            num_tokens: Number of tokens in the prompt

        Returns:
            List of allocated physical block IDs

        Raises:
            RuntimeError: If not enough free blocks
        """
        num_blocks = compute_num_blocks(num_tokens, self.block_size)

        if not self.can_allocate(num_blocks):
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks. "
                f"Only {len(self.free_blocks)} free blocks available."
            )

        # Create block table for sequence
        block_table = BlockTable(seq_id=seq_id, block_size=self.block_size)

        # Allocate physical blocks
        allocated_ids = []
        for i in range(num_blocks):
            block_id = self._allocate_block()
            allocated_ids.append(block_id)

            # Create logical block and assign physical
            logical = block_table.allocate_logical_block()
            logical.physical_block_id = block_id

            # Track tokens in each block
            start_token = i * self.block_size
            end_token = min((i + 1) * self.block_size, num_tokens)
            logical.num_tokens = end_token - start_token
            self.blocks[block_id].num_tokens = logical.num_tokens

        self.block_tables[seq_id] = block_table
        return allocated_ids

    def allocate_token(self, seq_id: int) -> Optional[int]:
        """
        Allocate space for one new token (decode step).

        If the current block has space, no new allocation is needed.
        Otherwise, allocate a new block.

        Args:
            seq_id: Sequence ID

        Returns:
            Block ID if new block was allocated, None if using existing block

        Raises:
            RuntimeError: If sequence not found or allocation fails
        """
        if seq_id not in self.block_tables:
            raise RuntimeError(f"Sequence {seq_id} not found")

        block_table = self.block_tables[seq_id]
        current_tokens = self._get_context_len(seq_id)

        # Check if current block has space
        if current_tokens % self.block_size != 0:
            # Current block has space, just update token count
            last_logical = block_table.blocks[-1]
            last_logical.num_tokens += 1
            self.blocks[last_logical.physical_block_id].num_tokens += 1
            return None

        # Need a new block
        if not self.can_allocate(1):
            raise RuntimeError("No free blocks available for decode")

        block_id = self._allocate_block()

        logical = block_table.allocate_logical_block()
        logical.physical_block_id = block_id
        logical.num_tokens = 1
        self.blocks[block_id].num_tokens = 1

        return block_id

    def free_sequence(self, seq_id: int) -> None:
        """
        Free all blocks allocated to a sequence.

        Args:
            seq_id: Sequence ID to free
        """
        if seq_id not in self.block_tables:
            return

        block_table = self.block_tables[seq_id]

        for logical in block_table.blocks:
            if logical.physical_block_id is not None:
                self._free_block(logical.physical_block_id)

        del self.block_tables[seq_id]

    def fork_sequence(self, src_seq_id: int, dst_seq_id: int) -> None:
        """
        Fork a sequence (copy-on-write for beam search).

        Creates a new sequence that shares blocks with the source.
        Actual copying happens only when one sequence modifies a shared block.

        Args:
            src_seq_id: Source sequence ID
            dst_seq_id: Destination sequence ID

        Raises:
            RuntimeError: If source sequence not found
        """
        if src_seq_id not in self.block_tables:
            raise RuntimeError(f"Source sequence {src_seq_id} not found")

        src_table = self.block_tables[src_seq_id]

        # Create new block table referencing same physical blocks
        dst_table = BlockTable(seq_id=dst_seq_id, block_size=self.block_size)

        for src_logical in src_table.blocks:
            if src_logical.physical_block_id is not None:
                # Increment ref count on shared block
                self.blocks[src_logical.physical_block_id].ref_count += 1

                dst_logical = dst_table.allocate_logical_block()
                dst_logical.physical_block_id = src_logical.physical_block_id
                dst_logical.num_tokens = src_logical.num_tokens

        self.block_tables[dst_seq_id] = dst_table

    def copy_on_write(self, seq_id: int, block_idx: int) -> Tuple[int, int]:
        """
        Copy a shared block for modification (copy-on-write).

        Called when a sequence needs to modify a block that's shared
        with other sequences.

        Args:
            seq_id: Sequence that needs to modify
            block_idx: Logical block index to copy

        Returns:
            Tuple of (src_block_id, dst_block_id) for the copy operation

        Raises:
            RuntimeError: If allocation fails
        """
        if seq_id not in self.block_tables:
            raise RuntimeError(f"Sequence {seq_id} not found")

        block_table = self.block_tables[seq_id]
        logical = block_table.blocks[block_idx]
        src_block_id = logical.physical_block_id

        if src_block_id is None:
            raise RuntimeError(f"Block {block_idx} not allocated")

        src_block = self.blocks[src_block_id]

        # If only one reference, no copy needed
        if src_block.ref_count == 1:
            return (src_block_id, src_block_id)

        # Allocate new block
        if not self.can_allocate(1):
            raise RuntimeError("No free blocks for copy-on-write")

        dst_block_id = self._allocate_block()

        # Update ref counts
        src_block.ref_count -= 1

        # Update block table to point to new block
        logical.physical_block_id = dst_block_id
        self.blocks[dst_block_id].num_tokens = src_block.num_tokens

        return (src_block_id, dst_block_id)

    def get_block_table(self, seq_id: int) -> List[int]:
        """
        Get physical block IDs for a sequence.

        Args:
            seq_id: Sequence ID

        Returns:
            List of physical block IDs
        """
        if seq_id not in self.block_tables:
            return []

        return self.block_tables[seq_id].get_physical_blocks()

    def get_block_table_tensor(self, seq_id: int, max_blocks: int) -> List[int]:
        """
        Get block table in tensor format (padded).

        Args:
            seq_id: Sequence ID
            max_blocks: Pad to this length

        Returns:
            List of physical block IDs, padded with 0
        """
        if seq_id not in self.block_tables:
            return [0] * max_blocks

        return self.block_tables[seq_id].to_tensor_format(max_blocks)

    def get_slot_mapping(
        self,
        seq_id: int,
        context_len: int,
        num_new_tokens: int,
    ) -> List[int]:
        """
        Compute slot mapping for writing new KV to cache.

        The slot is a linear index into the flattened cache:
            slot = block_id * block_size + offset_within_block

        Args:
            seq_id: Sequence ID
            context_len: Number of tokens already in cache
            num_new_tokens: Number of new tokens to write

        Returns:
            List of slot indices, one per new token
        """
        if seq_id not in self.block_tables:
            return []

        block_table = self.get_block_table(seq_id)
        slots = []

        for i in range(num_new_tokens):
            token_pos = context_len + i
            block_idx = token_pos // self.block_size
            offset = token_pos % self.block_size

            if block_idx < len(block_table):
                block_id = block_table[block_idx]
                slot = block_id * self.block_size + offset
                slots.append(slot)

        return slots

    def _allocate_block(self) -> int:
        """Allocate a single physical block."""
        if not self.free_blocks:
            # Try to evict cached blocks if prefix caching enabled
            if self.enable_prefix_caching:
                evicted = self._evict_cached_block()
                if evicted is not None:
                    return evicted
            raise RuntimeError("No free blocks available")

        block_id = self.free_blocks.pop()
        self.blocks[block_id].allocate()
        self.num_allocated += 1
        return block_id

    def _free_block(self, block_id: int) -> None:
        """Free a single physical block."""
        block = self.blocks[block_id]
        keep_cached = self.enable_prefix_caching and block.content_hash is not None

        block.release(keep_cached=keep_cached)

        if block.is_free:
            self.free_blocks.add(block_id)
            self.num_allocated -= 1
            # Remove from prefix cache if present
            if block.content_hash in self.prefix_cache:
                del self.prefix_cache[block.content_hash]
        elif block.is_cached:
            self.num_cached += 1

    def _evict_cached_block(self) -> Optional[int]:
        """Evict a cached block (LRU) and return its ID."""
        # Find oldest cached block
        oldest_time = float('inf')
        oldest_block = None

        for block in self.blocks:
            if block.is_evictable and block.last_accessed < oldest_time:
                oldest_time = block.last_accessed
                oldest_block = block

        if oldest_block is not None:
            # Remove from prefix cache
            if oldest_block.content_hash in self.prefix_cache:
                del self.prefix_cache[oldest_block.content_hash]

            oldest_block.free()
            self.num_cached -= 1
            self.num_allocated += 1
            oldest_block.allocate()
            return oldest_block.block_id

        return None

    def _get_context_len(self, seq_id: int) -> int:
        """Get total number of tokens for a sequence."""
        if seq_id not in self.block_tables:
            return 0

        block_table = self.block_tables[seq_id]
        return sum(logical.num_tokens for logical in block_table.blocks)

    def get_sequence_info(self, seq_id: int) -> dict:
        """Get debug info for a sequence."""
        if seq_id not in self.block_tables:
            return {"error": "Sequence not found"}

        block_table = self.block_tables[seq_id]
        return {
            "seq_id": seq_id,
            "num_blocks": len(block_table),
            "context_len": self._get_context_len(seq_id),
            "physical_blocks": block_table.get_physical_blocks(),
        }

    def __repr__(self) -> str:
        return (
            f"BlockManager(total={self.num_blocks}, "
            f"free={len(self.free_blocks)}, "
            f"allocated={self.num_allocated}, "
            f"cached={self.num_cached})"
        )


def create_block_manager(
    num_blocks: int,
    block_size: int = 16,
    enable_prefix_caching: bool = False,
) -> BlockManager:
    """
    Convenience function to create a BlockManager.

    Args:
        num_blocks: Total number of blocks
        block_size: Tokens per block
        enable_prefix_caching: Whether to enable prefix caching

    Returns:
        Initialized BlockManager
    """
    config = BlockManagerConfig(
        num_blocks=num_blocks,
        block_size=block_size,
        enable_prefix_caching=enable_prefix_caching,
    )
    return BlockManager(config)
