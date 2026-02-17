"""
Full Qwen3 model implementation.

This module provides a complete Qwen3 model that:
1. Defines all components from scratch (using our custom layers)
2. Loads weights from HuggingFace checkpoints
3. Supports both prefill (FlashAttention) and decode (PagedAttention) modes

Architecture (Qwen3-4B example):
- embed_tokens: [vocab_size=151936, hidden_size=2560]
- 40 decoder layers with GQA (num_heads=20, num_kv_heads=4)
- RoPE with theta=1000000
- SwiGLU MLP with intermediate_size=6912
- Final RMSNorm + lm_head

Reference:
- https://huggingface.co/Qwen/Qwen3-4B
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from llm_engine.model.layers import RMSNorm, Qwen3MLP
from llm_engine.model.attention import Qwen3Attention
from llm_engine.model.rope import compute_rope_cos_sin, RotaryEmbedding


@dataclass
class Qwen3Config:
    """Configuration for Qwen3 model."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_hidden_layers: int = 40
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
            num_hidden_layers=config_dict.get("num_hidden_layers", 40),
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
        is_prefill: bool = True,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
        block_tables: Optional[Tensor] = None,
        context_lens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through decoder layer.

        Returns:
            Tuple of (output_hidden_states, (key, value)).
        """
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, kv = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            is_prefill=is_prefill,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            k_cache=k_cache,
            v_cache=v_cache,
            block_tables=block_tables,
            context_lens=context_lens,
        )

        hidden_states = residual + attn_output

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, kv


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
        position_ids: Optional[Tensor] = None,
        is_prefill: bool = True,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
        block_tables: Optional[Tensor] = None,
        context_lens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Forward pass through transformer.

        Args:
            input_ids: Token IDs [batch, seq_len] or [total_tokens] for packed.
            position_ids: Position IDs [batch, seq_len] or [total_tokens].
            is_prefill: Whether in prefill mode.
            cu_seqlens: Cumulative sequence lengths for packed input.
            max_seqlen: Maximum sequence length.
            kv_caches: List of (k_cache, v_cache) per layer (for decode).
            block_tables: Block tables [batch, max_blocks] (for decode).
            context_lens: Context lengths [batch] (for decode).

        Returns:
            Tuple of (hidden_states, new_kv_list).
        """
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Compute RoPE embeddings
        if position_ids is None:
            if input_ids.dim() == 1:
                # Packed: positions are implicit
                position_ids = torch.arange(
                    input_ids.shape[0], device=input_ids.device
                )
            else:
                # Batched
                seq_len = input_ids.shape[1]
                position_ids = torch.arange(seq_len, device=input_ids.device)
                position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)

        cos, sin = compute_rope_cos_sin(
            position_ids if position_ids.dim() == 2 else position_ids.unsqueeze(0),
            self.config.head_dim,
            self.config.rope_theta,
        )

        # Cast to match hidden states dtype (RoPE is computed in float32 for
        # precision, but must match model dtype for downstream ops)
        cos = cos.to(hidden_states.dtype)
        sin = sin.to(hidden_states.dtype)

        # For packed sequences, squeeze the batch dimension
        if position_ids.dim() == 1:
            cos = cos.squeeze(0)  # [total_tokens, head_dim]
            sin = sin.squeeze(0)

        position_embeddings = (cos, sin)

        # Process through layers
        new_kv_list = []
        for i, layer in enumerate(self.layers):
            # Get KV cache for this layer if available
            if kv_caches is not None and i < len(kv_caches):
                k_cache, v_cache = kv_caches[i]
            else:
                k_cache, v_cache = None, None

            hidden_states, kv = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                is_prefill=is_prefill,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                k_cache=k_cache,
                v_cache=v_cache,
                block_tables=block_tables,
                context_lens=context_lens,
            )
            new_kv_list.append(kv)

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, new_kv_list


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 for causal language modeling.

    This is the full model including the lm_head for next-token prediction.
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
        position_ids: Optional[Tensor] = None,
        is_prefill: bool = True,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
        kv_caches: Optional[List[Tuple[Tensor, Tensor]]] = None,
        block_tables: Optional[Tensor] = None,
        context_lens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Forward pass returning logits.

        Args:
            input_ids: Token IDs.
            position_ids: Position IDs.
            is_prefill: Whether in prefill mode.
            cu_seqlens: Cumulative sequence lengths.
            max_seqlen: Maximum sequence length.
            kv_caches: KV caches per layer.
            block_tables: Block tables for PagedAttention.
            context_lens: Context lengths.

        Returns:
            Tuple of (logits, new_kv_list).
        """
        hidden_states, new_kv_list = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            is_prefill=is_prefill,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            kv_caches=kv_caches,
            block_tables=block_tables,
            context_lens=context_lens,
        )

        # Compute logits
        logits = self.lm_head(hidden_states)

        return logits, new_kv_list

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
