#!/usr/bin/env python3
"""
Test KV Cache and BlockManager implementation.

Tests:
1. KVCache allocation and memory layout
2. BlockManager allocation/deallocation
3. Slot mapping computation
4. Copy-on-write for forked sequences
5. Integration with attention (write_to_kv_cache, decode)

Usage:
    python scripts/test_kv_cache.py
"""

import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_engine.memory import (
    KVCache,
    KVCacheConfig,
    create_kv_cache,
    compute_slot_mapping,
    BlockManager,
    BlockManagerConfig,
    create_block_manager,
)
from llm_engine.model.attention import write_to_kv_cache, paged_attention_decode


def test_kv_cache_allocation():
    """Test KVCache tensor allocation."""
    print("\n" + "=" * 60)
    print("Test 1: KVCache Allocation")
    print("=" * 60)

    # Qwen3-4B config
    config = KVCacheConfig(
        num_layers=36,
        num_kv_heads=4,
        head_dim=128,
        block_size=16,
        dtype=torch.bfloat16,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    num_blocks = 100
    cache = KVCache(config, num_blocks)

    # Verify shapes
    k_cache, v_cache = cache.get_layer_cache(0)
    expected_shape = (num_blocks, config.block_size, config.num_kv_heads, config.head_dim)

    print(f"Config: {config.num_layers} layers, {config.num_kv_heads} KV heads, head_dim={config.head_dim}")
    print(f"Allocated {num_blocks} blocks with block_size={config.block_size}")
    print(f"K cache shape: {k_cache.shape} (expected {expected_shape})")
    print(f"V cache shape: {v_cache.shape}")
    print(f"Memory usage: {cache.get_memory_usage_mb():.2f} MB")

    # Verify all layers have same shape
    all_caches = cache.get_all_layer_caches()
    shapes_correct = all(
        k.shape == expected_shape and v.shape == expected_shape
        for k, v in all_caches
    )

    checks = [
        ("k_cache shape", k_cache.shape, expected_shape),
        ("v_cache shape", v_cache.shape, expected_shape),
        ("num layers", len(all_caches), config.num_layers),
        ("all shapes correct", shapes_correct, True),
    ]

    all_pass = True
    for name, actual, expected in checks:
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False

    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_block_manager_allocation():
    """Test BlockManager sequence allocation."""
    print("\n" + "=" * 60)
    print("Test 2: BlockManager Allocation")
    print("=" * 60)

    manager = create_block_manager(num_blocks=100, block_size=16)
    print(f"Created: {manager}")

    # Allocate sequence with 48 tokens (3 blocks)
    seq_id = 1
    num_tokens = 48
    blocks = manager.allocate_sequence(seq_id, num_tokens)

    print(f"\nAllocated sequence {seq_id} with {num_tokens} tokens")
    print(f"  Blocks: {blocks}")
    print(f"  Block table: {manager.get_block_table(seq_id)}")
    print(f"  Sequence info: {manager.get_sequence_info(seq_id)}")

    # Verify allocation
    expected_blocks = 3  # 48 tokens / 16 per block = 3 blocks
    checks = [
        ("num blocks allocated", len(blocks), expected_blocks),
        ("free blocks", manager.get_num_free_blocks(), 100 - expected_blocks),
        ("allocated blocks", manager.get_num_allocated_blocks(), expected_blocks),
    ]

    all_pass = True
    for name, actual, expected in checks:
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False

    # Test decode allocation (add 1 token)
    print("\n--- Decode: Add 1 token ---")
    new_block = manager.allocate_token(seq_id)
    context_len = manager._get_context_len(seq_id)
    print(f"  New block allocated: {new_block}")
    print(f"  Context length: {context_len}")

    # 49 tokens still fits in 4 blocks (needs 4th block at token 49)
    # Actually 48 tokens = 3 blocks exactly, token 49 needs block 4
    # Wait, 48/16 = 3 exactly, so token 49 (index 48) needs block index 3
    # So we should get a new block
    expected_new_block = new_block is not None  # Should allocate 4th block

    checks.append(("decode allocates new block", new_block is not None, True))
    checks.append(("context_len after decode", context_len, 49))

    # Free sequence
    print("\n--- Free sequence ---")
    manager.free_sequence(seq_id)
    print(f"  Free blocks after: {manager.get_num_free_blocks()}")

    checks.append(("free blocks after free", manager.get_num_free_blocks(), 100))

    print("\n--- Verification ---")
    for name, actual, expected in checks:
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False

    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_slot_mapping():
    """Test slot mapping computation."""
    print("\n" + "=" * 60)
    print("Test 3: Slot Mapping")
    print("=" * 60)

    manager = create_block_manager(num_blocks=100, block_size=16)

    # Allocate sequence
    seq_id = 1
    manager.allocate_sequence(seq_id, num_tokens=32)  # 2 blocks

    # Get slot mapping for prefill (all 32 tokens)
    slots = manager.get_slot_mapping(seq_id, context_len=0, num_new_tokens=32)
    print(f"Slot mapping for 32 prefill tokens: {slots[:5]}...{slots[-5:]}")

    # Verify slot mapping
    block_table = manager.get_block_table(seq_id)
    print(f"Block table: {block_table}")

    # First token should be at block_table[0] * 16 + 0
    expected_first = block_table[0] * 16 + 0
    # Token 16 should be at block_table[1] * 16 + 0
    expected_token16 = block_table[1] * 16 + 0

    checks = [
        ("slot[0]", slots[0], expected_first),
        ("slot[16]", slots[16], expected_token16),
        ("num slots", len(slots), 32),
    ]

    all_pass = True
    for name, actual, expected in checks:
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False

    # Test decode slot mapping
    print("\n--- Decode slot mapping ---")
    manager.allocate_token(seq_id)  # Token 33
    decode_slot = manager.get_slot_mapping(seq_id, context_len=32, num_new_tokens=1)
    print(f"Decode slot for token 33: {decode_slot}")

    # Token 32 (0-indexed) should be at block_table[2] * 16 + 0
    expected_decode = manager.get_block_table(seq_id)[2] * 16 + 0
    checks.append(("decode slot", decode_slot[0], expected_decode))

    print("\n--- Verification ---")
    for name, actual, expected in checks:
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False

    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_copy_on_write():
    """Test copy-on-write for forked sequences."""
    print("\n" + "=" * 60)
    print("Test 4: Copy-on-Write (Beam Search)")
    print("=" * 60)

    manager = create_block_manager(num_blocks=100, block_size=16)

    # Allocate parent sequence
    parent_id = 1
    manager.allocate_sequence(parent_id, num_tokens=32)
    parent_blocks = manager.get_block_table(parent_id)
    print(f"Parent sequence {parent_id}: blocks = {parent_blocks}")

    # Fork to create child
    child_id = 2
    manager.fork_sequence(parent_id, child_id)
    child_blocks = manager.get_block_table(child_id)
    print(f"Child sequence {child_id}: blocks = {child_blocks}")

    # Verify blocks are shared
    blocks_shared = parent_blocks == child_blocks
    print(f"Blocks shared: {blocks_shared}")

    # Check ref counts
    ref_counts = [manager.blocks[b].ref_count for b in parent_blocks]
    print(f"Ref counts: {ref_counts}")

    checks = [
        ("blocks shared", blocks_shared, True),
        ("ref counts", ref_counts, [2, 2]),  # Both seqs reference same blocks
    ]

    # Trigger copy-on-write
    print("\n--- Copy-on-write ---")
    src_block, dst_block = manager.copy_on_write(child_id, block_idx=1)
    print(f"CoW: copied block {src_block} to {dst_block}")

    child_blocks_after = manager.get_block_table(child_id)
    print(f"Child blocks after CoW: {child_blocks_after}")

    # Child's block 1 should now be different from parent's
    checks.append(("CoW creates new block", dst_block != src_block, True))
    checks.append(("child block 1 changed", child_blocks_after[1] != parent_blocks[1], True))

    all_pass = True
    print("\n--- Verification ---")
    for name, actual, expected in checks:
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False

    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_kv_cache_integration():
    """Test KVCache with write_to_kv_cache and paged_attention_decode."""
    print("\n" + "=" * 60)
    print("Test 5: KVCache Integration with Attention")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Create small cache for testing
    num_blocks = 10
    block_size = 16
    num_kv_heads = 4
    head_dim = 128

    cache = create_kv_cache(
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )

    manager = create_block_manager(num_blocks=num_blocks, block_size=block_size)

    # Allocate sequence with 20 tokens (2 blocks)
    seq_id = 1
    num_tokens = 20
    manager.allocate_sequence(seq_id, num_tokens)

    print(f"Allocated sequence with {num_tokens} tokens")
    print(f"Block table: {manager.get_block_table(seq_id)}")

    # Create fake KV tensors
    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Compute slot mapping
    slot_mapping = manager.get_slot_mapping(seq_id, context_len=0, num_new_tokens=num_tokens)
    slot_tensor = torch.tensor(slot_mapping, dtype=torch.int64, device=device)

    print(f"Slot mapping: {slot_mapping[:5]}...")

    # Write to cache
    k_cache, v_cache = cache.get_layer_cache(0)
    write_to_kv_cache(key, value, k_cache, v_cache, slot_tensor)

    print("Wrote KV to cache")

    # Verify write by reading back
    block_table = manager.get_block_table(seq_id)

    # Check first token of first block
    first_block = block_table[0]
    k_read = k_cache[first_block, 0]  # [num_kv_heads, head_dim]
    k_expected = key[0]

    write_correct = torch.allclose(k_read, k_expected, atol=1e-5)
    print(f"Write verified: {write_correct}")

    # Test decode attention
    print("\n--- Decode Attention ---")

    # Allocate one more token
    manager.allocate_token(seq_id)

    # Decode query (batch=1)
    query = torch.randn(1, num_kv_heads * 5, head_dim, dtype=dtype, device=device)  # num_heads = 20

    # Build block table tensor
    block_table_tensor = torch.tensor(
        [manager.get_block_table_tensor(seq_id, max_blocks=2)],
        dtype=torch.int32,
        device=device,
    )
    context_lens = torch.tensor([num_tokens], dtype=torch.int32, device=device)

    print(f"Block table tensor: {block_table_tensor}")
    print(f"Context lens: {context_lens}")

    # Run paged attention
    output = paged_attention_decode(
        query,
        k_cache,
        v_cache,
        block_table_tensor,
        context_lens,
    )

    print(f"Decode output shape: {output.shape}")

    checks = [
        ("write correct", write_correct, True),
        ("output shape", output.shape, (1, num_kv_heads * 5, head_dim)),
    ]

    all_pass = True
    print("\n--- Verification ---")
    for name, actual, expected in checks:
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False

    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main():
    print("=" * 60)
    print("Testing KV Cache and BlockManager (Step 4)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = []

    # Test 1: KVCache allocation
    results.append(("KVCache Allocation", test_kv_cache_allocation()))

    # Test 2: BlockManager allocation
    results.append(("BlockManager Allocation", test_block_manager_allocation()))

    # Test 3: Slot mapping
    results.append(("Slot Mapping", test_slot_mapping()))

    # Test 4: Copy-on-write
    results.append(("Copy-on-Write", test_copy_on_write()))

    # Test 5: Integration
    results.append(("KVCache Integration", test_kv_cache_integration()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests passed! ✓")
        print("KV Cache and BlockManager are working correctly.")
        print("\nNext step: Implement Scheduler (Step 5)")
        return 0
    else:
        print("\nSome tests failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
