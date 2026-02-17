#!/usr/bin/env python3
"""
Compare outputs between our custom Qwen3 implementation and HuggingFace.

This script:
1. Loads both models with the same weights
2. Runs the same input through both
3. Compares logits/hidden states and reports differences

Usage:
    python scripts/compare_models.py --model-path /path/to/Qwen3-4B

Requirements:
    - A Qwen3 model downloaded locally
    - GPU with sufficient memory (or CPU with enough RAM)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_engine.model.qwen3 import Qwen3ForCausalLM, Qwen3Config


def parse_args():
    parser = argparse.ArgumentParser(description="Compare custom Qwen3 vs HuggingFace")
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
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Data type (default: float32 for precision)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you today?",
        help="Test prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to compare",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for comparison",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for comparison",
    )
    return parser.parse_args()


def load_hf_model(model_path: str, device: str, dtype: torch.dtype):
    """Load HuggingFace model."""
    print(f"Loading HuggingFace model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    print(f"  Loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def load_custom_model(model_path: str, device: str, dtype: torch.dtype):
    """Load our custom Qwen3 model."""
    print(f"Loading custom Qwen3 model from {model_path}...")
    model = Qwen3ForCausalLM.from_pretrained(
        model_path,
        device=torch.device(device),
        dtype=dtype,
    )
    model.eval()
    print(f"  Loaded with {model.get_num_params():,} parameters")
    return model


def compare_logits(
    hf_logits: torch.Tensor,
    custom_logits: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict:
    """Compare logits from both models."""
    # Ensure same shape
    if hf_logits.shape != custom_logits.shape:
        return {
            "match": False,
            "error": f"Shape mismatch: HF={hf_logits.shape}, Custom={custom_logits.shape}",
        }

    # Compute differences
    abs_diff = torch.abs(hf_logits - custom_logits)
    rel_diff = abs_diff / (torch.abs(hf_logits) + 1e-8)

    # Statistics
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    # Check if within tolerance
    close = torch.allclose(hf_logits, custom_logits, atol=atol, rtol=rtol)

    # Compare top predictions
    hf_top = hf_logits[:, -1, :].argmax(dim=-1)
    custom_top = custom_logits[:, -1, :].argmax(dim=-1)
    top_match = (hf_top == custom_top).all().item()

    # Compare top-5
    hf_top5 = hf_logits[:, -1, :].topk(5, dim=-1).indices
    custom_top5 = custom_logits[:, -1, :].topk(5, dim=-1).indices

    return {
        "match": close,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "top1_match": top_match,
        "hf_top5": hf_top5[0].tolist(),
        "custom_top5": custom_top5[0].tolist(),
    }


def compare_token_by_token(
    hf_model,
    custom_model,
    tokenizer,
    input_ids: torch.Tensor,
    device: str,
    atol: float,
    rtol: float,
):
    """Compare outputs token by token."""
    print("\n" + "=" * 60)
    print("Token-by-Token Comparison")
    print("=" * 60)

    seq_len = input_ids.shape[1]
    all_match = True

    for pos in range(1, seq_len + 1):
        partial_ids = input_ids[:, :pos]

        with torch.no_grad():
            # HuggingFace forward
            hf_out = hf_model(partial_ids)
            hf_logits = hf_out.logits

            # Custom forward (legacy mode for HF comparison)
            custom_logits = custom_model.forward_legacy(partial_ids)

        # Compare
        result = compare_logits(hf_logits, custom_logits, atol, rtol)

        token = tokenizer.decode([input_ids[0, pos - 1].item()])
        status = "✓" if result["match"] else "✗"

        print(f"  Position {pos:2d} [{token:12s}]: {status} "
              f"max_diff={result['max_abs_diff']:.2e}, "
              f"top1_match={result['top1_match']}")

        if not result["match"]:
            all_match = False

    return all_match


def main():
    args = parse_args()

    # Map dtype string
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 60)
    print("Qwen3 Model Comparison: Custom vs HuggingFace")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Prompt: {args.prompt}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Tokenize input
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    input_ids = input_ids[:, :args.max_tokens]  # Limit length
    input_ids = input_ids.to(args.device)
    print(f"Input tokens: {input_ids.shape[1]}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print()

    # Load models
    hf_model = load_hf_model(args.model_path, args.device, dtype)
    custom_model = load_custom_model(args.model_path, args.device, dtype)
    print()

    # Run forward pass
    print("=" * 60)
    print("Full Sequence Comparison")
    print("=" * 60)

    with torch.no_grad():
        # HuggingFace forward
        print("Running HuggingFace forward...")
        hf_out = hf_model(input_ids, output_hidden_states=True)
        hf_logits = hf_out.logits
        hf_hidden = hf_out.hidden_states[-1]  # Last layer hidden states

        # Custom forward (legacy mode for HF comparison)
        print("Running custom forward...")
        custom_logits = custom_model.forward_legacy(input_ids)

    # Compare logits
    print("\nLogits Comparison:")
    result = compare_logits(hf_logits, custom_logits, args.atol, args.rtol)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
    else:
        print(f"  Match (within tolerance): {'Yes' if result['match'] else 'No'}")
        print(f"  Max absolute diff: {result['max_abs_diff']:.6e}")
        print(f"  Mean absolute diff: {result['mean_abs_diff']:.6e}")
        print(f"  Max relative diff: {result['max_rel_diff']:.6e}")
        print(f"  Mean relative diff: {result['mean_rel_diff']:.6e}")
        print(f"  Top-1 prediction match: {'Yes' if result['top1_match'] else 'No'}")
        print(f"  HF top-5 tokens: {result['hf_top5']}")
        print(f"  Custom top-5 tokens: {result['custom_top5']}")

        # Decode top predictions
        print(f"  HF top-5 decoded: {[tokenizer.decode([t]) for t in result['hf_top5']]}")
        print(f"  Custom top-5 decoded: {[tokenizer.decode([t]) for t in result['custom_top5']]}")

    # Token-by-token comparison (optional, more detailed)
    if input_ids.shape[1] <= 20:
        all_match = compare_token_by_token(
            hf_model, custom_model, tokenizer, input_ids,
            args.device, args.atol, args.rtol
        )
    else:
        print("\n(Skipping token-by-token comparison for sequences > 20 tokens)")
        all_match = result.get("match", False)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if result.get("match", False) and result.get("top1_match", False):
        print("SUCCESS: Custom model outputs match HuggingFace within tolerance!")
        print("The implementation is correct.")
        return 0
    elif result.get("top1_match", False):
        print("PARTIAL MATCH: Top predictions match, but numerical differences exist.")
        print("This may be acceptable depending on use case.")
        return 0
    else:
        print("MISMATCH: Outputs differ significantly.")
        print("Please investigate the differences.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
