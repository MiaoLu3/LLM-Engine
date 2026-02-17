"""
Full Qwen3 model implementation.

This module provides a complete Qwen3 model that:
1. Defines all components from scratch (using our custom layers)
2. Loads weights from HuggingFace checkpoints
3. Supports unified forward (Option C) for prefill, chunked prefill, and decode

Architecture (Qwen3-4B example):
- embed_tokens: [vocab_size=151936, hidden_size=2560]
- 36 decoder layers with GQA (num_heads=20, num_kv_heads=4)
- RoPE with theta=1000000
- SwiGLU MLP with intermediate_size=6912
- Final RMSNorm + lm_head

Reference:
- https://huggingface.co/Qwen/Qwen3-4B
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, TYPE_CHECKING
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from llm_engine.model.layers import RMSNorm, Qwen3MLP
from llm_engine.model.attention import Qwen3Attention
from llm_engine.model.rope import compute_rope_cos_sin, RotaryEmbedding

if TYPE_CHECKING:
    from llm_engine.model.attention_metadata import AttentionMetadata


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 model."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 36
    num_attention_heads: int = 20
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Qwen3Config":
        """Load config from HuggingFace checkpoint."""
        config_path = Path(model_path) / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        return cls(
            vocab_size=config_dict.get("vocab_size", 151936),
            hidden_size=config_dict.get("hidden_size", 2560),
            intermediate_size=config_dict.get("intermediate_size", 6912),
            num_hidden_layers=config_dict.get("num_hidden_layers", 36),
            num_attention_heads=config_dict.get("num_attention_heads", 20),
            num_key_value_heads=config_dict.get("num_key_value_heads", 4),
            head_dim=config_dict.get(
                "head_dim",
                config_dict.get("hidden_size", 2560)
                // config_dict.get("num_attention_heads", 20),
            ),
            max_position_embeddings=config_dict.get("max_position_embeddings", 32768),
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-6),
            rope_theta=config_dict.get("rope_theta", 1000000.0),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
        )


class Qwen3DecoderLayer(nn.Module):
    """
    Single Qwen3 decoder layer.

    Pre-norm architecture:
    1. input_layernorm -> self_attn -> residual
    2. post_attention_layernorm -> mlp -> residual
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Layer norms
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Self-attention
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            layer_idx=layer_idx,
        )

        # MLP
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Tuple[Tensor, Tensor],
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> Tensor:
        """
        Forward pass through decoder layer.

        Args:
            hidden_states: [total_tokens, hidden_size] or [batch, seq, hidden_size]
            position_embeddings: (cos, sin) for RoPE
            kv_cache: Optional (k_cache, v_cache) for this layer
            attn_metadata: Optional metadata for unified attention

        Returns:
            Output hidden states
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        hidden_states = residual + attn_output

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model(nn.Module):
    """
    Qwen3 transformer model (without lm_head).

    This is the core transformer stack:
    - Token embedding
    - N decoder layers
    - Final RMSNorm
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Decoder layers
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
        )

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> Tensor:
        """
        Unified forward pass through transformer.

        Args:
            input_ids: Token IDs [total_tokens] for packed input.
            positions: Position IDs [total_tokens] for each token.
            kv_caches: List of (k_cache, v_cache) per layer.
                      Shape: [num_blocks, block_size, num_kv_heads, head_dim]
            attn_metadata: AttentionMetadata describing batch composition.

        Returns:
            Hidden states [total_tokens, hidden_size]
        """
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Compute RoPE embeddings
        # positions: [total_tokens]
        cos, sin = compute_rope_cos_sin(
            positions.unsqueeze(0),  # [1, total_tokens]
            self.config.head_dim,
            self.config.rope_theta,
        )

        # Cast to match hidden states dtype
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # Squeeze for packed format: [total_tokens, head_dim]
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)

        position_embeddings = (cos, sin)

        # Process through layers
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None

            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def forward_legacy(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Legacy forward for HuggingFace comparison (pure prefill, no cache).

        Args:
            input_ids: [batch, seq_len] for batched input
            position_ids: Optional [batch, seq_len]

        Returns:
            Hidden states [batch, seq_len, hidden_size]
        """
        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        cos, sin = compute_rope_cos_sin(
            position_ids,
            self.config.head_dim,
            self.config.rope_theta,
        )

        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        position_embeddings = (cos, sin)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                kv_cache=None,
                attn_metadata=None,
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 for causal language modeling.

    This is the full model including the lm_head for next-token prediction.
    Supports unified forward (Option C) for prefill, chunked prefill, and decode.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        # Core transformer
        self.model = Qwen3Model(config)

        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
        attn_metadata: Optional["AttentionMetadata"] = None,
    ) -> Tensor:
        """
        Unified forward pass returning logits.

        Args:
            input_ids: Token IDs [total_tokens].
            positions: Position IDs [total_tokens].
            kv_caches: List of (k_cache, v_cache) per layer.
            attn_metadata: AttentionMetadata describing batch composition.

        Returns:
            Logits [total_tokens, vocab_size]
        """
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )

        # Compute logits
        logits = self.lm_head(hidden_states)

        return logits

    def forward_legacy(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Legacy forward for HuggingFace comparison (pure prefill, no cache).

        Args:
            input_ids: [batch, seq_len]
            position_ids: Optional [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        hidden_states = self.model.forward_legacy(
            input_ids=input_ids,
            position_ids=position_ids,
        )

        logits = self.lm_head(hidden_states)

        return logits

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Qwen3ForCausalLM":
        """
        Load model from HuggingFace checkpoint.

        Args:
            model_path: Path to model directory containing:
                - config.json
                - model-*.safetensors or pytorch_model.bin
            device: Device to load model to.
            dtype: Data type for model parameters.

        Returns:
            Loaded Qwen3ForCausalLM model.
        """
        from llm_engine.model.loader import load_qwen3_weights

        # Load config
        config = Qwen3Config.from_pretrained(model_path)

        # Create model
        model = cls(config)

        # Load weights
        load_qwen3_weights(model, model_path)

        # Move to device/dtype
        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype)

        return model

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_layers(self) -> int:
        """Return number of decoder layers."""
        return self.config.num_hidden_layers

    def get_head_dim(self) -> int:
        """Return head dimension."""
        return self.config.head_dim

    def get_num_kv_heads(self) -> int:
        """Return number of KV heads per layer."""
        return self.config.num_key_value_heads
