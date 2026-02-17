"""
KV Cache tensor allocation and management.

This module provides GPU memory allocation for storing Key and Value tensors
across all transformer layers. The cache is organized as fixed-size blocks
that can be dynamically assigned to sequences.

Block-based Layout:
    Shape per layer: [num_blocks, block_size, num_kv_heads, head_dim]

The block-based design enables:
- Dynamic allocation (sequences get blocks as they grow)
- Non-contiguous storage (blocks don't need to be adjacent)
- Memory sharing (prefix caching)
- Efficient memory utilization (no pre-allocation per sequence)

Reference: vLLM PagedAttention
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class KVCacheConfig:
    """Configuration for KV cache allocation."""

    num_layers: int
    """Number of transformer layers."""

    num_kv_heads: int
    """Number of KV heads per layer."""

    head_dim: int
    """Dimension of each attention head."""

    block_size: int = 16
    """Number of tokens per block."""

    dtype: torch.dtype = torch.bfloat16
    """Data type for cache tensors."""

    device: torch.device = None
    """Device for cache tensors (default: cuda)."""

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KVCache:
    """
    Block-based KV cache storage.

    Manages GPU tensors for storing K/V across all layers.
    Each layer has a separate K cache and V cache tensor.

    Cache Layout:
        k_cache[layer]: [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache[layer]: [num_blocks, block_size, num_kv_heads, head_dim]

    Usage:
        1. Create cache with total number of blocks
        2. BlockManager assigns block IDs to sequences
        3. Model writes to cache using slot_mapping
        4. Model reads from cache using block_tables

    Example:
        >>> config = KVCacheConfig(num_layers=36, num_kv_heads=4, head_dim=128)
        >>> cache = KVCache(config, num_blocks=1000)
        >>> k_cache, v_cache = cache.get_layer_cache(layer_idx=0)
    """

    def __init__(self, config: KVCacheConfig, num_blocks: int):
        """
        Initialize KV cache with pre-allocated GPU memory.

        Args:
            config: Cache configuration
            num_blocks: Total number of blocks to allocate
        """
        self.config = config
        self.num_blocks = num_blocks
        self.num_layers = config.num_layers
        self.block_size = config.block_size
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.dtype = config.dtype
        self.device = config.device

        # Allocate cache tensors for all layers
        self.k_caches: List[Tensor] = []
        self.v_caches: List[Tensor] = []

        self._allocate_cache()

    def _allocate_cache(self) -> None:
        """Allocate GPU memory for all layers."""
        cache_shape = (
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
        )

        for _ in range(self.num_layers):
            k_cache = torch.zeros(
                cache_shape, dtype=self.dtype, device=self.device
            )
            v_cache = torch.zeros(
                cache_shape, dtype=self.dtype, device=self.device
            )
            self.k_caches.append(k_cache)
            self.v_caches.append(v_cache)

    def get_layer_cache(self, layer_idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get KV cache tensors for a specific layer.

        Args:
            layer_idx: Index of the layer (0-indexed)

        Returns:
            Tuple of (k_cache, v_cache) tensors
            Shape: [num_blocks, block_size, num_kv_heads, head_dim]
        """
        return self.k_caches[layer_idx], self.v_caches[layer_idx]

    def get_all_layer_caches(self) -> List[Tuple[Tensor, Tensor]]:
        """
        Get KV cache tensors for all layers.

        Returns:
            List of (k_cache, v_cache) tuples, one per layer
        """
        return list(zip(self.k_caches, self.v_caches))

    def clear(self) -> None:
        """Zero out all cache tensors."""
        for k_cache, v_cache in zip(self.k_caches, self.v_caches):
            k_cache.zero_()
            v_cache.zero_()

    def clear_blocks(self, block_ids: List[int]) -> None:
        """
        Zero out specific blocks across all layers.

        Args:
            block_ids: List of block IDs to clear
        """
        if not block_ids:
            return

        block_ids_tensor = torch.tensor(block_ids, device=self.device)

        for k_cache, v_cache in zip(self.k_caches, self.v_caches):
            k_cache[block_ids_tensor] = 0
            v_cache[block_ids_tensor] = 0

    def copy_blocks(
        self,
        src_to_dst: List[Tuple[int, int]],
    ) -> None:
        """
        Copy blocks from source to destination (for copy-on-write).

        Args:
            src_to_dst: List of (src_block_id, dst_block_id) pairs
        """
        if not src_to_dst:
            return

        src_ids = [s for s, _ in src_to_dst]
        dst_ids = [d for _, d in src_to_dst]

        src_tensor = torch.tensor(src_ids, device=self.device)
        dst_tensor = torch.tensor(dst_ids, device=self.device)

        for k_cache, v_cache in zip(self.k_caches, self.v_caches):
            k_cache[dst_tensor] = k_cache[src_tensor]
            v_cache[dst_tensor] = v_cache[src_tensor]

    def get_memory_usage(self) -> int:
        """
        Get total GPU memory usage in bytes.

        Returns:
            Total bytes allocated for KV cache
        """
        if not self.k_caches:
            return 0

        # Each tensor: num_blocks * block_size * num_kv_heads * head_dim * dtype_size
        tensor_size = self.k_caches[0].numel() * self.k_caches[0].element_size()
        # K + V for each layer
        return tensor_size * 2 * self.num_layers

    def get_memory_usage_mb(self) -> float:
        """Get total GPU memory usage in megabytes."""
        return self.get_memory_usage() / (1024 * 1024)

    def get_memory_usage_gb(self) -> float:
        """Get total GPU memory usage in gigabytes."""
        return self.get_memory_usage() / (1024 * 1024 * 1024)

    @staticmethod
    def compute_num_blocks(
        available_memory_bytes: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """
        Compute number of blocks that fit in available memory.

        Args:
            available_memory_bytes: Available GPU memory in bytes
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads
            head_dim: Head dimension
            block_size: Tokens per block
            dtype: Data type

        Returns:
            Number of blocks that can be allocated
        """
        dtype_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else dtype.itemsize

        # Bytes per block (K + V)
        bytes_per_block = 2 * block_size * num_kv_heads * head_dim * dtype_size

        # Total bytes per block across all layers
        bytes_per_block_all_layers = bytes_per_block * num_layers

        return available_memory_bytes // bytes_per_block_all_layers

    @staticmethod
    def compute_bytes_per_token(
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """
        Compute KV cache bytes needed per token.

        Args:
            num_layers: Number of transformer layers
            num_kv_heads: Number of KV heads
            head_dim: Head dimension
            dtype: Data type

        Returns:
            Bytes per token for KV cache
        """
        dtype_size = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else dtype.itemsize

        # K + V for each layer
        return 2 * num_layers * num_kv_heads * head_dim * dtype_size

    def __repr__(self) -> str:
        return (
            f"KVCache(num_layers={self.num_layers}, num_blocks={self.num_blocks}, "
            f"block_size={self.block_size}, memory={self.get_memory_usage_gb():.2f}GB)"
        )


def create_kv_cache(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    num_blocks: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> KVCache:
    """
    Convenience function to create a KV cache.

    Args:
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV heads per layer
        head_dim: Dimension of each attention head
        num_blocks: Total number of blocks to allocate
        block_size: Tokens per block (default: 16)
        dtype: Data type (default: bfloat16)
        device: Device (default: cuda)

    Returns:
        Initialized KVCache
    """
    config = KVCacheConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )
    return KVCache(config, num_blocks)


def compute_slot_mapping(
    block_table: List[int],
    context_len: int,
    block_size: int,
    num_new_tokens: int = 1,
) -> List[int]:
    """
    Compute slot mapping for writing new KV to cache.

    The slot is a linear index into the flattened cache:
        slot = block_id * block_size + offset_within_block

    Args:
        block_table: List of physical block IDs for the sequence
        context_len: Number of tokens already in cache (before new tokens)
        block_size: Tokens per block
        num_new_tokens: Number of new tokens to write

    Returns:
        List of slot indices, one per new token
    """
    slots = []
    for i in range(num_new_tokens):
        token_pos = context_len + i
        block_idx = token_pos // block_size
        offset = token_pos % block_size

        if block_idx < len(block_table):
            block_id = block_table[block_idx]
            slot = block_id * block_size + offset
            slots.append(slot)

    return slots
