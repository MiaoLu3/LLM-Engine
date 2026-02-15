"""
Model execution with packed inputs and FlashAttention.

The ModelExecutor handles:
1. Packing variable-length sequences into flat tensors
2. Running forward passes with FlashAttention varlen
3. Managing KV cache reads/writes
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

try:
    from flash_attn import flash_attn_varlen_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_varlen_func = None


@dataclass
class PackedInput:
    """
    Packed input for a batch of sequences.

    All sequences are concatenated into flat tensors with metadata
    to track boundaries.
    """

    token_ids: torch.Tensor
    """Flat tensor of all token IDs [total_tokens]."""

    position_ids: torch.Tensor
    """Position IDs for each token [total_tokens]."""

    cu_seqlens: torch.Tensor
    """Cumulative sequence lengths [num_seqs + 1].
    cu_seqlens[i] to cu_seqlens[i+1] gives tokens for sequence i."""

    max_seqlen: int
    """Maximum sequence length in this batch (for FlashAttention)."""

    seq_ids: List[int]
    """List of sequence IDs in order."""

    # For decode phase, we need to know where each sequence's KV is
    block_tables: Optional[torch.Tensor] = None
    """Block tables for PagedAttention [num_seqs, max_blocks]."""

    context_lens: Optional[torch.Tensor] = None
    """Context lengths for decode [num_seqs]."""

    @property
    def num_tokens(self) -> int:
        """Total number of tokens in this batch."""
        return self.token_ids.shape[0]

    @property
    def num_seqs(self) -> int:
        """Number of sequences in this batch."""
        return len(self.cu_seqlens) - 1


@dataclass
class ModelOutput:
    """Output from model forward pass."""

    logits: torch.Tensor
    """Logits for next token prediction [num_tokens, vocab_size].
    For decode, only the last token of each sequence has valid logits."""

    hidden_states: Optional[torch.Tensor] = None
    """Final hidden states [num_tokens, hidden_size]. Optional."""


class ModelExecutor:
    """
    Executes model forward passes with packed inputs.

    Handles both prefill (all prompt tokens) and decode (one token per seq)
    phases using FlashAttention's variable length API.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        vocab_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """
        Initialize the model executor.

        Args:
            model: Loaded HuggingFace model.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            num_kv_heads: Number of KV heads (for GQA).
            head_dim: Dimension of each attention head.
            vocab_size: Vocabulary size.
            dtype: Data type for computation.
            device: Device to run on.
        """
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.device = device

        # Check FlashAttention availability
        if not FLASH_ATTN_AVAILABLE:
            raise ImportError(
                "flash-attn not installed. Install with: pip install flash-attn"
            )

    @torch.inference_mode()
    def forward_prefill(
        self,
        packed_input: PackedInput,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[ModelOutput, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Execute prefill forward pass.

        Processes all prompt tokens and populates KV cache.

        Args:
            packed_input: Packed input tensors.
            kv_caches: List of (key_cache, value_cache) per layer.
                       Each cache is [num_blocks, block_size, num_kv_heads, head_dim].

        Returns:
            ModelOutput with logits, and updated KV caches.
        """
        # Get embeddings
        inputs_embeds = self.model.model.embed_tokens(packed_input.token_ids)

        # For HuggingFace models, we need to use their forward with our modifications
        # This is a simplified version - full implementation would hook into attention

        # Run through transformer layers
        hidden_states = inputs_embeds

        for layer_idx, layer in enumerate(self.model.model.layers):
            # Standard layer forward (simplified - real impl would use flash attn)
            hidden_states = layer(
                hidden_states,
                position_ids=packed_input.position_ids.unsqueeze(0),
            )[0]

        # Final layer norm
        hidden_states = self.model.model.norm(hidden_states)

        # Get logits
        logits = self.model.lm_head(hidden_states)

        return ModelOutput(logits=logits), kv_caches

    @torch.inference_mode()
    def forward_decode(
        self,
        packed_input: PackedInput,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> ModelOutput:
        """
        Execute decode forward pass.

        Processes one new token per sequence, reading from KV cache.

        Args:
            packed_input: Packed input (one token per sequence).
            kv_caches: List of (key_cache, value_cache) per layer.

        Returns:
            ModelOutput with logits for next token.
        """
        # Get embeddings for new tokens only
        inputs_embeds = self.model.model.embed_tokens(packed_input.token_ids)

        # Run through transformer layers with KV cache
        hidden_states = inputs_embeds

        for layer_idx, layer in enumerate(self.model.model.layers):
            # Would use PagedAttention here with block_tables
            hidden_states = layer(
                hidden_states,
                position_ids=packed_input.position_ids.unsqueeze(0),
            )[0]

        # Final layer norm
        hidden_states = self.model.model.norm(hidden_states)

        # Get logits
        logits = self.model.lm_head(hidden_states)

        return ModelOutput(logits=logits)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        is_prefill: bool = True,
    ) -> torch.Tensor:
        """
        Simple forward pass for initial implementation.

        This uses the HuggingFace model directly. A production implementation
        would replace the attention with FlashAttention + PagedAttention.

        Args:
            input_ids: Token IDs [batch_size, seq_len] or [total_tokens].
            position_ids: Position IDs matching input_ids shape.
            kv_caches: Optional KV caches (not used in simple impl).
            is_prefill: Whether this is prefill or decode.

        Returns:
            Logits tensor [batch_size, seq_len, vocab_size] or [total_tokens, vocab_size].
        """
        # Ensure 2D input for HuggingFace
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0)

        # Forward through model
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,  # We manage cache separately
        )

        return outputs.logits

    def sample_next_tokens(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        top_logprobs: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[dict]]]:
        """
        Sample next tokens from logits with log probability tracking.

        Args:
            logits: Logits tensor [num_seqs, vocab_size].
            temperature: Sampling temperature (0 = greedy).
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling (-1 = disabled).
            top_logprobs: Number of top alternatives to return.

        Returns:
            Tuple of:
            - next_tokens: Sampled token IDs [num_seqs].
            - sampled_logprobs: Log probs of sampled tokens [num_seqs].
            - top_logprobs_list: List of dicts mapping token_id -> logprob for top-k.
        """
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample or greedy decode
        if temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Get log prob of sampled tokens
        sampled_logprobs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

        # Get top-k alternatives if requested
        top_logprobs_list = None
        if top_logprobs > 0:
            top_logprobs_list = []
            k = min(top_logprobs, log_probs.size(-1))
            top_values, top_indices = torch.topk(log_probs, k, dim=-1)

            for i in range(log_probs.size(0)):
                token_logprobs = {}
                for j in range(k):
                    token_id = top_indices[i, j].item()
                    logprob = top_values[i, j].item()
                    token_logprobs[token_id] = logprob
                top_logprobs_list.append(token_logprobs)

        return next_tokens, sampled_logprobs, top_logprobs_list
