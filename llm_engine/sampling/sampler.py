"""
Token sampling with log probability tracking.

The Sampler handles:
1. Temperature scaling
2. Top-k and top-p (nucleus) filtering
3. Greedy vs stochastic sampling
4. Per-token log probability extraction
5. Top-k alternatives for each position
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict

from llm_engine.config import SamplingParams


@dataclass
class SamplerOutput:
    """Output from sampling a batch of sequences."""

    next_tokens: torch.Tensor
    """Sampled token IDs [num_seqs]."""

    logprobs: torch.Tensor
    """Log probability of each sampled token [num_seqs]."""

    top_logprobs: Optional[List[Dict[int, float]]]
    """Top-k alternatives for each position. None if top_logprobs=0."""


class Sampler:
    """
    Samples next tokens from model logits.

    Always computes and returns log probabilities of sampled tokens.
    Optionally returns top-k alternative tokens at each position.
    """

    def __init__(self, vocab_size: int, device: str = "cuda"):
        """
        Initialize the sampler.

        Args:
            vocab_size: Size of vocabulary.
            device: Device to run sampling on.
        """
        self.vocab_size = vocab_size
        self.device = device

    def __call__(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
    ) -> SamplerOutput:
        """
        Sample next tokens from logits.

        Args:
            logits: Raw logits [num_seqs, vocab_size].
            params: Sampling parameters.

        Returns:
            SamplerOutput with sampled tokens and log probabilities.
        """
        return self.sample(logits, params)

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
    ) -> SamplerOutput:
        """
        Sample next tokens with full log probability tracking.

        The sampling process:
        1. Apply temperature scaling
        2. Compute log probabilities (before any filtering)
        3. Apply top-k filtering if specified
        4. Apply top-p (nucleus) filtering if specified
        5. Sample from filtered distribution
        6. Extract log prob of sampled token
        7. Extract top-k alternatives if requested

        Args:
            logits: Raw logits [num_seqs, vocab_size].
            params: Sampling parameters.

        Returns:
            SamplerOutput with tokens, logprobs, and optional top-k.
        """
        # Ensure logits are 2D
        if logits.dim() == 3:
            # Take last token position for each sequence
            logits = logits[:, -1, :]

        original_logits = logits.clone()

        # Apply temperature
        temperature = params.temperature
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # Compute log probabilities BEFORE filtering
        # This gives us the true model probabilities
        log_probs = F.log_softmax(original_logits, dim=-1)

        # For sampling, we may apply filtering
        sampling_logits = logits.clone()

        # Apply top-k filtering
        if params.top_k > 0:
            top_k = min(params.top_k, sampling_logits.size(-1))
            top_k_values = torch.topk(sampling_logits, top_k, dim=-1)[0]
            threshold = top_k_values[..., -1, None]
            sampling_logits = sampling_logits.masked_fill(
                sampling_logits < threshold, float("-inf")
            )

        # Apply top-p (nucleus) filtering
        if params.top_p < 1.0:
            sampling_logits = self._apply_top_p(sampling_logits, params.top_p)

        # Sample or greedy decode
        if temperature == 0 or params.is_greedy:
            # Greedy: take argmax of original logits
            next_tokens = original_logits.argmax(dim=-1)
        else:
            # Stochastic sampling from filtered distribution
            probs = F.softmax(sampling_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # Get log probability of sampled tokens (from unfiltered distribution)
        sampled_logprobs = log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)

        # Get top-k alternatives if requested
        top_logprobs_list = None
        if params.top_logprobs > 0:
            top_logprobs_list = self._get_top_logprobs(
                log_probs, params.top_logprobs
            )

        return SamplerOutput(
            next_tokens=next_tokens,
            logprobs=sampled_logprobs,
            top_logprobs=top_logprobs_list,
        )

    def _apply_top_p(
        self,
        logits: torch.Tensor,
        top_p: float,
    ) -> torch.Tensor:
        """
        Apply nucleus (top-p) sampling filter.

        Keeps the smallest set of tokens whose cumulative probability
        exceeds top_p.

        Args:
            logits: Logits to filter [num_seqs, vocab_size].
            top_p: Cumulative probability threshold.

        Returns:
            Filtered logits with low-probability tokens set to -inf.
        """
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Find cutoff point
        sorted_indices_to_remove = cumulative_probs > top_p

        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter back to original order
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )

        return logits.masked_fill(indices_to_remove, float("-inf"))

    def _get_top_logprobs(
        self,
        log_probs: torch.Tensor,
        k: int,
    ) -> List[Dict[int, float]]:
        """
        Extract top-k log probabilities for each position.

        Args:
            log_probs: Log probabilities [num_seqs, vocab_size].
            k: Number of top alternatives to return.

        Returns:
            List of dicts, one per sequence, mapping token_id -> logprob.
        """
        k = min(k, log_probs.size(-1))
        top_values, top_indices = torch.topk(log_probs, k, dim=-1)

        result = []
        for i in range(log_probs.size(0)):
            token_logprobs = {}
            for j in range(k):
                token_id = top_indices[i, j].item()
                logprob = top_values[i, j].item()
                token_logprobs[token_id] = logprob
            result.append(token_logprobs)

        return result

    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        past_tokens: List[List[int]],
        penalty: float,
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits.

        Tokens that appear in past_tokens have their logits divided
        (if positive) or multiplied (if negative) by the penalty.

        Args:
            logits: Logits to penalize [num_seqs, vocab_size].
            past_tokens: List of past token IDs for each sequence.
            penalty: Repetition penalty factor (1.0 = no penalty).

        Returns:
            Penalized logits.
        """
        if penalty == 1.0:
            return logits

        for i, tokens in enumerate(past_tokens):
            if not tokens:
                continue

            unique_tokens = list(set(tokens))
            token_tensor = torch.tensor(unique_tokens, device=logits.device)

            # Get logits for past tokens
            past_logits = logits[i, token_tensor]

            # Apply penalty
            # If logit > 0, divide by penalty (reduce probability)
            # If logit < 0, multiply by penalty (also reduce probability)
            logits[i, token_tensor] = torch.where(
                past_logits > 0,
                past_logits / penalty,
                past_logits * penalty,
            )

        return logits
