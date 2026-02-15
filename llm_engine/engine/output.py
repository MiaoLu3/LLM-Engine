"""
Output data structures for generation results.

RequestOutput: Complete output for a request (may contain multiple completions)
CompletionOutput: Single completion from one sequence
TokenLogProb: Log probability info for a single token position
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TopLogProb:
    """Log probability for a single token (used in top-k alternatives)."""

    token_id: int
    """Token ID."""

    token: str
    """Decoded token text."""

    logprob: float
    """Log probability of this token."""

    def __repr__(self) -> str:
        return f"TopLogProb({self.token!r}: {self.logprob:.4f})"


@dataclass
class TokenLogProb:
    """
    Log probability information for one generated token position.

    Always includes the sampled token's logprob.
    Optionally includes top-k alternative tokens.
    """

    token_id: int
    """The sampled token ID."""

    token: str
    """Decoded token text."""

    logprob: float
    """Log probability of the sampled token."""

    top_logprobs: Optional[list[TopLogProb]] = None
    """Top-k alternative tokens at this position (if requested).
    Includes the sampled token if it's in top-k."""

    def __repr__(self) -> str:
        top_k = f", top_k={len(self.top_logprobs)}" if self.top_logprobs else ""
        return f"TokenLogProb({self.token!r}: {self.logprob:.4f}{top_k})"


@dataclass
class CompletionOutput:
    """
    Output from a single sequence within a request.

    When n > 1, each sequence produces its own CompletionOutput.
    Always includes per-token log probabilities.
    """

    index: int
    """Index of this completion (0 to n-1)."""

    text: str
    """Generated text (decoded from tokens)."""

    token_ids: list[int]
    """Generated token IDs."""

    logprobs: list[TokenLogProb] = field(default_factory=list)
    """Per-token log probability information. Always populated."""

    cumulative_logprob: float = 0.0
    """Sum of log probabilities for all generated tokens."""

    finish_reason: Optional[str] = None
    """Why generation stopped: 'stop', 'length', or None if still generating."""

    @property
    def num_tokens(self) -> int:
        """Number of generated tokens."""
        return len(self.token_ids)

    @property
    def mean_logprob(self) -> float:
        """Average log probability per token."""
        if self.num_tokens == 0:
            return 0.0
        return self.cumulative_logprob / self.num_tokens

    @property
    def perplexity(self) -> float:
        """Perplexity of the generated sequence (exp of negative mean logprob)."""
        import math
        return math.exp(-self.mean_logprob) if self.num_tokens > 0 else float('inf')

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"CompletionOutput(index={self.index}, tokens={self.num_tokens}, "
            f"finish_reason={self.finish_reason}, text={text_preview!r})"
        )


@dataclass
class RequestOutput:
    """
    Complete output for a request.

    Contains the original prompt and all generated completions.
    For n > 1, there will be multiple CompletionOutput entries.
    """

    request_id: str
    """Unique identifier for this request."""

    prompt: str
    """Original input prompt text."""

    prompt_token_ids: list[int]
    """Tokenized prompt."""

    outputs: list[CompletionOutput] = field(default_factory=list)
    """Generated completions (one per sequence in the group)."""

    finished: bool = False
    """Whether all sequences have finished generating."""

    @property
    def prompt_len(self) -> int:
        """Length of prompt in tokens."""
        return len(self.prompt_token_ids)

    @property
    def num_outputs(self) -> int:
        """Number of completions."""
        return len(self.outputs)

    def add_output(self, output: CompletionOutput) -> None:
        """Add a completion output."""
        self.outputs.append(output)

    def get_best_output(self) -> Optional[CompletionOutput]:
        """Get the completion with highest cumulative log probability."""
        if not self.outputs:
            return None
        return max(self.outputs, key=lambda o: o.cumulative_logprob)

    def __repr__(self) -> str:
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt_len={self.prompt_len}, num_outputs={self.num_outputs}, "
            f"finished={self.finished})"
        )


@dataclass
class SchedulerOutputs:
    """
    Output from one scheduler iteration.

    Contains all information needed to execute a forward pass
    and process the results.
    """

    scheduled_seq_groups: list
    """SequenceGroups scheduled for this iteration."""

    num_prefill_tokens: int
    """Total prefill tokens in this iteration."""

    num_decode_tokens: int
    """Total decode tokens in this iteration."""

    blocks_to_swap_in: dict[int, int] = field(default_factory=dict)
    """Physical block ID -> CPU block ID for swap in."""

    blocks_to_swap_out: dict[int, int] = field(default_factory=dict)
    """Physical block ID -> CPU block ID for swap out."""

    blocks_to_copy: dict[int, int] = field(default_factory=dict)
    """Source block ID -> Dest block ID for copy-on-write."""

    @property
    def num_batched_tokens(self) -> int:
        """Total tokens in this batch."""
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_empty(self) -> bool:
        """Check if nothing is scheduled."""
        return len(self.scheduled_seq_groups) == 0

    def __repr__(self) -> str:
        return (
            f"SchedulerOutputs(groups={len(self.scheduled_seq_groups)}, "
            f"prefill={self.num_prefill_tokens}, decode={self.num_decode_tokens})"
        )
