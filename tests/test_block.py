"""Tests for Block and BlockTable data structures."""

import pytest
from llm_engine.data_structures.block import (
    PhysicalBlock,
    LogicalBlock,
    BlockTable,
    BlockStatus,
    compute_num_blocks,
    compute_block_hash,
)


class TestBlockStatus:
    """Tests for BlockStatus enum."""

    def test_all_statuses_exist(self):
        """Verify all expected block statuses exist."""
        assert BlockStatus.FREE is not None
        assert BlockStatus.ALLOCATED is not None
        assert BlockStatus.CACHED is not None


class TestPhysicalBlock:
    """Tests for PhysicalBlock dataclass."""

    def test_basic_creation(self):
        """Test creating a basic physical block."""
        block = PhysicalBlock(block_id=0, block_size=16)
        assert block.block_id == 0
        assert block.block_size == 16
        assert block.ref_count == 0
        assert block.status == BlockStatus.FREE
        assert block.content_hash is None
        assert block.num_tokens == 0

    def test_is_free(self):
        """Test is_free property."""
        block = PhysicalBlock(block_id=0, block_size=16)
        assert block.is_free is True

        block.status = BlockStatus.ALLOCATED
        assert block.is_free is False

    def test_is_allocated(self):
        """Test is_allocated property."""
        block = PhysicalBlock(block_id=0, block_size=16)
        assert block.is_allocated is False

        block.status = BlockStatus.ALLOCATED
        assert block.is_allocated is True

    def test_is_cached(self):
        """Test is_cached property."""
        block = PhysicalBlock(block_id=0, block_size=16)
        assert block.is_cached is False

        block.status = BlockStatus.CACHED
        assert block.is_cached is True

    def test_allocate(self):
        """Test allocating a block."""
        block = PhysicalBlock(block_id=0, block_size=16)
        block.allocate()

        assert block.ref_count == 1
        assert block.status == BlockStatus.ALLOCATED

        # Allocate again (shared block)
        block.allocate()
        assert block.ref_count == 2

    def test_release_to_free(self):
        """Test releasing a block back to free state."""
        block = PhysicalBlock(block_id=0, block_size=16)
        block.allocate()
        block.release(keep_cached=False)

        assert block.ref_count == 0
        assert block.status == BlockStatus.FREE

    def test_release_to_cached(self):
        """Test releasing a block to cached state (for prefix caching)."""
        block = PhysicalBlock(block_id=0, block_size=16)
        block.allocate()
        block.content_hash = 12345  # Has cacheable content
        block.release(keep_cached=True)

        assert block.ref_count == 0
        assert block.status == BlockStatus.CACHED
        assert block.content_hash == 12345  # Hash preserved

    def test_release_shared_block(self):
        """Test releasing a shared block decrements ref count."""
        block = PhysicalBlock(block_id=0, block_size=16)
        block.allocate()
        block.allocate()  # ref_count = 2
        block.release()

        assert block.ref_count == 1
        assert block.status == BlockStatus.ALLOCATED  # Still in use

    def test_free(self):
        """Test completely freeing a block."""
        block = PhysicalBlock(block_id=0, block_size=16)
        block.allocate()
        block.content_hash = 12345
        block.num_tokens = 10
        block.free()

        assert block.ref_count == 0
        assert block.status == BlockStatus.FREE
        assert block.content_hash is None
        assert block.num_tokens == 0

    def test_is_evictable(self):
        """Test is_evictable property."""
        block = PhysicalBlock(block_id=0, block_size=16)

        # Free block is not evictable (nothing to evict)
        assert block.is_evictable is False

        # Cached block with ref_count=0 is evictable
        block.status = BlockStatus.CACHED
        block.ref_count = 0
        assert block.is_evictable is True

        # Cached block with ref_count>0 is not evictable
        block.ref_count = 1
        assert block.is_evictable is False


class TestLogicalBlock:
    """Tests for LogicalBlock dataclass."""

    def test_basic_creation(self):
        """Test creating a logical block."""
        block = LogicalBlock(logical_idx=0)
        assert block.logical_idx == 0
        assert block.physical_block_id is None
        assert block.num_tokens == 0

    def test_is_allocated(self):
        """Test is_allocated property."""
        block = LogicalBlock(logical_idx=0)
        assert block.is_allocated is False

        block.physical_block_id = 5
        assert block.is_allocated is True


class TestBlockTable:
    """Tests for BlockTable dataclass."""

    def test_basic_creation(self):
        """Test creating a block table."""
        table = BlockTable(seq_id=0, block_size=16)
        assert table.seq_id == 0
        assert table.block_size == 16
        assert len(table) == 0

    def test_allocate_logical_block(self):
        """Test allocating logical blocks."""
        table = BlockTable(seq_id=0, block_size=16)

        block1 = table.allocate_logical_block()
        assert block1.logical_idx == 0
        assert len(table) == 1

        block2 = table.allocate_logical_block()
        assert block2.logical_idx == 1
        assert len(table) == 2

    def test_assign_physical_block(self):
        """Test assigning physical blocks to logical blocks."""
        table = BlockTable(seq_id=0, block_size=16)
        table.allocate_logical_block()
        table.allocate_logical_block()

        table.assign_physical_block(logical_idx=0, physical_block_id=10)
        table.assign_physical_block(logical_idx=1, physical_block_id=25)

        assert table.blocks[0].physical_block_id == 10
        assert table.blocks[1].physical_block_id == 25

    def test_get_physical_blocks(self):
        """Test getting list of physical block IDs."""
        table = BlockTable(seq_id=0, block_size=16)
        table.allocate_logical_block()
        table.allocate_logical_block()
        table.allocate_logical_block()

        table.assign_physical_block(0, 10)
        table.assign_physical_block(2, 25)
        # Block 1 is not assigned

        physical = table.get_physical_blocks()
        assert physical == [10, 25]

    def test_num_allocated_blocks(self):
        """Test counting allocated blocks."""
        table = BlockTable(seq_id=0, block_size=16)
        table.allocate_logical_block()
        table.allocate_logical_block()
        table.allocate_logical_block()

        assert table.num_allocated_blocks() == 0

        table.assign_physical_block(0, 10)
        assert table.num_allocated_blocks() == 1

        table.assign_physical_block(2, 25)
        assert table.num_allocated_blocks() == 2

    def test_num_blocks_needed(self):
        """Test calculating blocks needed for tokens."""
        table = BlockTable(seq_id=0, block_size=16)

        assert table.num_blocks_needed(0) == 0
        assert table.num_blocks_needed(1) == 1
        assert table.num_blocks_needed(16) == 1
        assert table.num_blocks_needed(17) == 2
        assert table.num_blocks_needed(32) == 2
        assert table.num_blocks_needed(33) == 3

    def test_get_block_for_token(self):
        """Test getting block for a token index."""
        table = BlockTable(seq_id=0, block_size=16)
        table.allocate_logical_block()
        table.allocate_logical_block()

        # Tokens 0-15 are in block 0
        assert table.get_block_for_token(0).logical_idx == 0
        assert table.get_block_for_token(15).logical_idx == 0

        # Tokens 16-31 are in block 1
        assert table.get_block_for_token(16).logical_idx == 1
        assert table.get_block_for_token(31).logical_idx == 1

        # Token 32 would be in block 2, which doesn't exist
        assert table.get_block_for_token(32) is None

    def test_to_tensor_format(self):
        """Test converting to tensor format."""
        table = BlockTable(seq_id=0, block_size=16)
        table.allocate_logical_block()
        table.allocate_logical_block()
        table.allocate_logical_block()

        table.assign_physical_block(0, 10)
        table.assign_physical_block(1, 25)
        # Block 2 not assigned

        tensor = table.to_tensor_format(max_blocks=5)
        assert tensor == [10, 25, -1, -1, -1]

    def test_clear(self):
        """Test clearing block table."""
        table = BlockTable(seq_id=0, block_size=16)
        table.allocate_logical_block()
        table.allocate_logical_block()
        table.assign_physical_block(0, 10)

        table.clear()
        assert len(table) == 0


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_num_blocks(self):
        """Test compute_num_blocks function."""
        assert compute_num_blocks(0, 16) == 0
        assert compute_num_blocks(1, 16) == 1
        assert compute_num_blocks(16, 16) == 1
        assert compute_num_blocks(17, 16) == 2
        assert compute_num_blocks(100, 16) == 7  # ceil(100/16) = 7

    def test_compute_block_hash_same_content_same_position(self):
        """Test that same content at same position gives same hash."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        hash1 = compute_block_hash(tokens, block_idx=0, block_size=16)
        hash2 = compute_block_hash(tokens, block_idx=0, block_size=16)
        assert hash1 == hash2

    def test_compute_block_hash_different_position(self):
        """Test that same content at different positions gives different hash."""
        tokens = list(range(32))  # 32 tokens = 2 blocks

        hash_block0 = compute_block_hash(tokens, block_idx=0, block_size=16)
        hash_block1 = compute_block_hash(tokens, block_idx=1, block_size=16)

        # Different positions should give different hashes
        # (even if token values happened to be same)
        assert hash_block0 != hash_block1

    def test_compute_block_hash_different_content(self):
        """Test that different content gives different hash."""
        tokens1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        tokens2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 99]

        hash1 = compute_block_hash(tokens1, block_idx=0, block_size=16)
        hash2 = compute_block_hash(tokens2, block_idx=0, block_size=16)
        assert hash1 != hash2

    def test_compute_block_hash_partial_block(self):
        """Test hash computation for partial (last) block."""
        tokens = [1, 2, 3, 4, 5]  # Only 5 tokens

        # Should still compute hash correctly
        hash_val = compute_block_hash(tokens, block_idx=0, block_size=16)
        assert isinstance(hash_val, int)
