#!/usr/bin/env python3
"""
Test unified forward (Option C) for pure prefill mode.

This script verifies that the unified forward path with AttentionMetadata
produces the same output as the legacy forward path for pure prefill.

What we can test NOW (no KV cache needed):
1. Pure prefill via unified forward
2. Compare with forward_legacy (should match)
3. Compare with HuggingFace (should match within tolerance)

What requires Step 4 (KV Cache):
- Decode mode
- Chunked prefill mode

Usage:
    python scripts/test_unified_prefill.py --model-path /path/to/Qwen3-4B
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_engine.model.qwen3 import Qwen3ForCausalLM
from llm_engine.model.attention_metadata import (
    AttentionMetadata,
    create_prefill_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Test unified prefill")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Qwen3 model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type",
    )
    return parser.parse_args()


def test_single_sequence_prefill(model, tokenizer, device, dtype):
    """Test unified prefill with a single sequence."""
    print("\n" + "=" * 60)
    print("Test 1: Single Sequence Pure Prefill")
    print("=" * 60)

    prompt = "Hello, how are you today?"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    batch_size, seq_len = input_ids.shape

    print(f"Prompt: {prompt}")
    print(f"Input shape: {input_ids.shape}")
    print(f"Token IDs: {input_ids[0].tolist()}")

    # Test 1a: Legacy forward (batched format)
    print("\n--- Legacy Forward (batched) ---")
    with torch.no_grad():
        logits_legacy = model.forward_legacy(input_ids)
    print(f"Logits shape: {logits_legacy.shape}")

    # Test 1b: Unified forward (packed format)
    print("\n--- Unified Forward (packed) ---")

    # Pack input: [total_tokens]
    packed_ids = input_ids.view(-1)  # [seq_len]
    positions = torch.arange(seq_len, device=device)  # [seq_len]

    # Create prefill metadata
    attn_metadata = create_prefill_metadata(
        seq_lens=[seq_len],
        device=device,
    )
    print(f"AttentionMetadata:")
    print(f"  num_prefill_tokens: {attn_metadata.num_prefill_tokens}")
    print(f"  num_prefill_seqs: {attn_metadata.num_prefill_seqs}")
    print(f"  prefill_cu_seqlens_q: {attn_metadata.prefill_cu_seqlens_q}")

    with torch.no_grad():
        logits_unified = model(
            packed_ids,
            positions,
            kv_caches=None,
            attn_metadata=attn_metadata,
        )

    # Reshape to match legacy format
    logits_unified = logits_unified.view(1, seq_len, -1)
    print(f"Logits shape: {logits_unified.shape}")

    # Compare
    diff = torch.abs(logits_legacy - logits_unified)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n--- Comparison ---")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    # Check top-1 predictions match
    legacy_top1 = logits_legacy[0, -1].argmax().item()
    unified_top1 = logits_unified[0, -1].argmax().item()
    top1_match = legacy_top1 == unified_top1

    print(f"Legacy top-1: {legacy_top1} ({tokenizer.decode([legacy_top1])})")
    print(f"Unified top-1: {unified_top1} ({tokenizer.decode([unified_top1])})")
    print(f"Top-1 match: {'✓' if top1_match else '✗'}")

    success = max_diff < 1e-4 and top1_match
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    return success


def test_multi_sequence_prefill(model, tokenizer, device, dtype):
    """Test unified prefill with multiple packed sequences."""
    print("\n" + "=" * 60)
    print("Test 2: Multi-Sequence Packed Prefill")
    print("=" * 60)

    prompts = [
        "Hello world",
        "How are you doing today?",
        "The quick brown fox",
    ]

    # Tokenize each prompt
    all_input_ids = []
    seq_lens = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)[0]
        all_input_ids.append(ids)
        seq_lens.append(len(ids))

    print(f"Prompts: {prompts}")
    print(f"Sequence lengths: {seq_lens}")

    # Test 2a: Legacy forward on each sequence separately
    print("\n--- Legacy Forward (per-sequence) ---")
    legacy_logits = []
    for ids in all_input_ids:
        with torch.no_grad():
            logits = model.forward_legacy(ids.unsqueeze(0))  # [1, seq_len, vocab]
        legacy_logits.append(logits.squeeze(0))  # [seq_len, vocab]

    # Test 2b: Unified forward (packed)
    print("\n--- Unified Forward (packed) ---")

    # Pack all sequences
    packed_ids = torch.cat(all_input_ids)  # [total_tokens]
    total_tokens = packed_ids.shape[0]

    # Build position IDs (reset for each sequence)
    positions = []
    for seq_len in seq_lens:
        positions.append(torch.arange(seq_len, device=device))
    positions = torch.cat(positions)

    # Create prefill metadata
    attn_metadata = create_prefill_metadata(
        seq_lens=seq_lens,
        device=device,
    )
    print(f"AttentionMetadata:")
    print(f"  num_prefill_tokens: {attn_metadata.num_prefill_tokens}")
    print(f"  num_prefill_seqs: {attn_metadata.num_prefill_seqs}")
    print(f"  prefill_cu_seqlens_q: {attn_metadata.prefill_cu_seqlens_q}")
    print(f"  max_prefill_seq_len: {attn_metadata.max_prefill_seq_len}")

    with torch.no_grad():
        logits_unified = model(
            packed_ids,
            positions,
            kv_caches=None,
            attn_metadata=attn_metadata,
        )
    print(f"Unified output shape: {logits_unified.shape}")

    # Split unified output back to sequences
    unified_logits = []
    offset = 0
    for seq_len in seq_lens:
        unified_logits.append(logits_unified[offset:offset + seq_len])
        offset += seq_len

    # Compare each sequence
    print(f"\n--- Comparison ---")
    all_match = True
    for i, (legacy, unified) in enumerate(zip(legacy_logits, unified_logits)):
        diff = torch.abs(legacy - unified)
        max_diff = diff.max().item()

        legacy_top1 = legacy[-1].argmax().item()
        unified_top1 = unified[-1].argmax().item()
        top1_match = legacy_top1 == unified_top1

        status = "✓" if max_diff < 1e-4 and top1_match else "✗"
        print(f"  Seq {i}: max_diff={max_diff:.2e}, top1_match={top1_match} {status}")

        if max_diff >= 1e-4 or not top1_match:
            all_match = False

    print(f"\nResult: {'PASS' if all_match else 'FAIL'}")
    return all_match


def test_attention_metadata_properties(device):
    """Test AttentionMetadata properties and helper functions."""
    print("\n" + "=" * 60)
    print("Test 3: AttentionMetadata Properties")
    print("=" * 60)

    from llm_engine.model.attention_metadata import (
        create_prefill_metadata,
        create_decode_metadata,
        create_empty_metadata,
    )

    # Test 3a: Pure prefill metadata
    print("\n--- Test 3a: Pure Prefill Metadata ---")
    metadata = create_prefill_metadata(seq_lens=[10, 5, 8], device=device)

    checks_3a = [
        ("num_prefill_tokens", metadata.num_prefill_tokens, 23),  # 10 + 5 + 8
        ("num_decode_tokens", metadata.num_decode_tokens, 0),
        ("num_prefill_seqs", metadata.num_prefill_seqs, 3),
        ("max_prefill_seq_len", metadata.max_prefill_seq_len, 10),
        ("has_prefill", metadata.has_prefill, True),
        ("has_decode", metadata.has_decode, False),
        ("is_pure_prefill", metadata.is_pure_prefill, True),
        ("has_chunked_prefill", metadata.has_chunked_prefill, False),
    ]

    all_pass = True
    for name, actual, expected in checks_3a:
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False

    # Verify cu_seqlens
    expected_cu = [0, 10, 15, 23]
    actual_cu = metadata.prefill_cu_seqlens_q.tolist()
    cu_match = actual_cu == expected_cu
    print(f"  prefill_cu_seqlens_q: {actual_cu} (expected {expected_cu}) {'✓' if cu_match else '✗'}")
    if not cu_match:
        all_pass = False

    # Test 3b: Decode metadata
    print("\n--- Test 3b: Decode Metadata ---")
    metadata = create_decode_metadata(
        context_lens=[15, 30],
        block_tables=[[0, 1], [2, 3, 4]],
        slot_mapping=[15, 30],  # slot for each decode token
        device=device,
    )

    checks_3b = [
        ("num_prefill_tokens", metadata.num_prefill_tokens, 0),
        ("num_decode_tokens", metadata.num_decode_tokens, 2),
        ("num_decode_seqs", metadata.num_decode_seqs, 2),
        ("has_prefill", metadata.has_prefill, False),
        ("has_decode", metadata.has_decode, True),
        ("is_pure_decode", metadata.is_pure_decode, True),
    ]

    for name, actual, expected in checks_3b:
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False

    # Check block tables are padded correctly
    expected_shape = (2, 3)  # 2 seqs, max 3 blocks
    actual_shape = tuple(metadata.decode_block_tables.shape)
    shape_match = actual_shape == expected_shape
    print(f"  decode_block_tables.shape: {actual_shape} (expected {expected_shape}) {'✓' if shape_match else '✗'}")
    if not shape_match:
        all_pass = False

    # Test 3c: Empty metadata
    print("\n--- Test 3c: Empty Metadata ---")
    metadata = create_empty_metadata(device=device)

    checks_3c = [
        ("num_tokens", metadata.num_tokens, 0),
        ("has_prefill", metadata.has_prefill, False),
        ("has_decode", metadata.has_decode, False),
    ]

    for name, actual, expected in checks_3c:
        match = actual == expected
        status = "✓" if match else "✗"
        print(f"  {name}: {actual} (expected {expected}) {status}")
        if not match:
            all_pass = False

    print(f"\nResult: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main():
    args = parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("Testing Unified Forward (Option C) - Pure Prefill Mode")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")

    # Load model and tokenizer
    print("\nLoading model...")
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_path,
        device=torch.device(args.device),
        dtype=dtype,
    )
    model.eval()
    print(f"Loaded {model.get_num_params():,} parameters")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Run tests
    results = []

    # Test 1: Single sequence
    results.append(("Single Sequence Prefill",
                   test_single_sequence_prefill(model, tokenizer, args.device, dtype)))

    # Test 2: Multi-sequence packed
    results.append(("Multi-Sequence Packed Prefill",
                   test_multi_sequence_prefill(model, tokenizer, args.device, dtype)))

    # Test 3: Metadata properties (no model needed)
    results.append(("AttentionMetadata Properties",
                   test_attention_metadata_properties(args.device)))

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
        print("Unified forward (Option C) is working correctly for pure prefill.")
        print("\nNext step: Implement KV Cache (Step 4) to test decode and chunked prefill.")
        return 0
    else:
        print("\nSome tests failed! ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
