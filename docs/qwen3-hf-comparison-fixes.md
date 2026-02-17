# Qwen3 Custom Implementation: HuggingFace Comparison Fixes

This document summarizes the bugs and improvements found while validating our custom Qwen3 implementation against HuggingFace's reference using `scripts/compare_models.py`.

Model tested: **Qwen3-4B** (`Qwen/Qwen3-4B`)

---

## Fix 1: Initialize Q/K Norms (Critical — model failed to load)

**File:** `llm_engine/model/attention.py`

**Problem:** `Qwen3Attention.__init__` set `q_norm` and `k_norm` to `None`, but the Qwen3-4B checkpoint contains `q_norm.weight` and `k_norm.weight` (RMSNorm with shape `[head_dim=128]`). The weight loader crashed when navigating `model.layers.0.self_attn.q_norm.weight` because `q_norm` was `None`.

```
AttributeError: 'NoneType' object has no attribute 'weight'
```

**Fix:** Initialize both norms as `RMSNorm(head_dim)` and apply them in all three forward paths (packed prefill, padded prefill, decode) after reshaping Q/K to head dimensions but before RoPE — matching the Qwen3 architecture.

```python
# Before
self.q_norm: Optional[nn.Module] = None
self.k_norm: Optional[nn.Module] = None

# After
self.q_norm = RMSNorm(head_dim)
self.k_norm = RMSNorm(head_dim)
```

**Result:** Model loads and runs. Max diff ~5.7e-04 in float32 (top-5 predictions match).

---

## Fix 2: Use custom RMSNorm instead of nn.RMSNorm (Precision — 5.7e-04 → 2.3e-05)

**File:** `llm_engine/model/attention.py`

**Problem:** The initial fix used PyTorch's built-in `nn.RMSNorm(head_dim)` which defaults to `eps=1e-8`. HuggingFace's `Qwen3RMSNorm` uses `eps=1e-6` (from `config.rms_norm_eps`). The eps difference created ~1e-6 per-element errors in Q/K vectors, which were amplified through attention score computation (128-dim dot product) and 36 transformer layers to ~5.7e-04.

**Fix:** Use our custom `RMSNorm(head_dim)` which defaults to `eps=1e-6`, matching HF.

```python
# Before
self.q_norm = nn.RMSNorm(head_dim)   # eps=1e-8
self.k_norm = nn.RMSNorm(head_dim)   # eps=1e-8

# After
from llm_engine.model.layers import RMSNorm
self.q_norm = RMSNorm(head_dim)      # eps=1e-6
self.k_norm = RMSNorm(head_dim)      # eps=1e-6
```

**Result:** Max diff reduced from ~5.7e-04 to ~2.3e-05 in float32.

---

## Fix 3: Replace naive attention with SDPA (Alignment)

**File:** `llm_engine/model/attention.py`

**Problem:** The padded prefill path used `naive_attention` (manual matmul → causal mask → softmax → matmul), while HF uses `F.scaled_dot_product_attention` (SDPA). Also:
- `is_causal` flag: HF sets `is_causal=True` only for `seq_len > 1`; we always set `True`.
- GQA expansion: HF uses `expand` + `reshape`; we used `repeat_interleave`, producing different memory layouts that affect CUDA kernel accumulation order.

**Fix:**
```python
# Replace naive_attention with SDPA
output = F.scaled_dot_product_attention(
    q, k, v, is_causal=(seq_len > 1),
)

# Match HF's GQA expansion (expand+reshape instead of repeat_interleave)
k = k[:, :, None, :, :].expand(
    batch_size, self.num_kv_heads, n_rep, seq_len, self.head_dim
).reshape(batch_size, self.num_heads, seq_len, self.head_dim)
```

---

## Fix 4: Cast RoPE cos/sin to model dtype (Critical — bf16/fp16 crashed)

**File:** `llm_engine/model/qwen3.py`

**Problem:** `compute_rope_cos_sin` always outputs cos/sin in float32. In bf16/fp16 mode, `apply_rope_to_qk` promoted Q/K to float32 (bf16 * float32 → float32), while V stayed in bf16. SDPA then received mixed dtypes and crashed.

```
RuntimeError: Expected query, key, and value to have the same dtype
```

**Fix:** Cast cos/sin to match hidden states dtype before passing to layers.

```python
cos = cos.to(hidden_states.dtype)
sin = sin.to(hidden_states.dtype)
```

This matches HF's approach: `return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)`.

---

## Final Results

### float32
```
Max absolute diff: 2.288818e-05
Mean absolute diff: 2.436015e-06
Top-1 match: Yes | Top-5 match: Yes
All positions: ✓ (within tolerance)
```

The remaining ~1e-05 gap is expected float32 numerical noise from:
- Different SDPA kernel dispatch (flash / memory-efficient / math backends)
- Float32 matmul accumulation order differences across CUDA thread blocks
- Minor RoPE intermediate precision differences

### bf16 / fp16
```
Max absolute diff: 0.000000e+00
All positions: ✓ (bit-exact match)
```

Bit-exact in reduced precision because the ~1e-05 float32 differences fall below the representable precision of bf16/fp16, so both implementations round to the same value.

---

## Files Modified

| File | Changes |
|------|---------|
| `llm_engine/model/attention.py` | Initialize q/k norms, apply QK-norm in all forward paths, switch to SDPA, match HF GQA expansion |
| `llm_engine/model/qwen3.py` | Cast RoPE cos/sin to model dtype |
