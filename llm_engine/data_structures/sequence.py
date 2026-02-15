"""
Sequence and SequenceGroup data structures.

A Sequence represents a single generation stream with:
- Prompt tokens (input)
- Output tokens (generated so far)
- State (waiting, running, swapped, finished)
- Block table mapping to physical KV cache blocks

A SequenceGroup represents multiple sequences generated from the same prompt
(used when n > 1 in sampling params).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SequenceStatus(Enum):
    """Status of a sequence in the scheduler."""

    WAITING = "waiting"
    """Sequence is in the waiting queue, not yet started."""

    RUNNING = "running"
    """Sequence is actively being processed."""

    SWAPPED = "swapped"
    """Sequence has been preempted, KV cache swapped to CPU."""

    FINISHED_STOPPED = "finished_stopped"
    """Generation stopped due to stop token or max_tokens."""

    FINISHED_LENGTH = "finished_length"
    """Generation stopped due to reaching max_tokens."""

    FINISHED_ABORTED = "finished_aborted"
    """Generation was aborted (e.g., by user request)."""

    @property
    def is_finished(self) -> bool:
        """Check if the sequence has finished generating."""
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


@dataclass
class TokenLogProbInfo:
    """
    Log probability information for a single generated token.

    Stored in Sequence for each output token.
    """

    logprob: float
    """Log probability of the sampled token."""

    top_logprobs: Optional[dict[int, float]] = None
    """Top-k alternative tokens with their log probabilities.
    Maps token_id -> logprob. None if top_k was 0."""


@dataclass
class Sequence:
    """
    A single generation stream.

    This is the core unit that the scheduler operates on.
    Each sequence maintains its own token history and KV cache mapping.
    """

    seq_id: int
    """Unique identifier for this sequence."""

    prompt_tokens: list[int]
    """Token IDs from the original prompt."""

    output_tokens: list[int] = field(default_factory=list)
    """Token IDs generated so far."""

    output_logprobs: list[TokenLogProbInfo] = field(default_factory=list)
    """Per-token log probability info for each generated token.
    output_logprobs[i] corresponds to output_tokens[i]."""

    status: SequenceStatus = SequenceStatus.WAITING
    """Current status in the scheduler."""

    # Block management
    block_table: list[int] = field(default_factory=list)
    """Mapping from logical block index to physical block ID.
    block_table[i] = physical_block_id for logical block i."""

    # Prefix caching
    num_cached_tokens: int = 0
    """Number of tokens whose KV cache was loaded from prefix cache.
    These tokens don't need prefill computation."""

    # Chunked prefill state
    num_prefilled_tokens: int = 0
    """Number of prompt tokens that have been prefilled so far.
    Used for chunked prefill to track progress."""

    # Generation tracking (derived from output_logprobs, but cached for efficiency)
    cumulative_logprob: float = 0.0
    """Sum of log probabilities of generated tokens."""

    @property
    def prompt_len(self) -> int:
        """Length of the prompt in tokens."""
        return len(self.prompt_tokens)

    @property
    def output_len(self) -> int:
        """Number of tokens generated so far."""
        return len(self.output_tokens)

    @property
    def total_len(self) -> int:
        """Total sequence length (prompt + output)."""
        return self.prompt_len + self.output_len

    @property
    def num_tokens_to_prefill(self) -> int:
        """Number of prompt tokens still needing prefill.
        Accounts for both prefix cache hits and chunked prefill progress."""
        return self.prompt_len - self.num_cached_tokens - self.num_prefilled_tokens

    @property
    def is_prefill(self) -> bool:
        """Check if this sequence is still in prefill phase."""
        return self.num_tokens_to_prefill > 0

    @property
    def all_tokens(self) -> list[int]:
        """All tokens (prompt + output) in order."""
        return self.prompt_tokens + self.output_tokens

    def append_token(
        self,
        token_id: int,
        logprob: float = 0.0,
        top_logprobs: Optional[dict[int, float]] = None,
    ) -> None:
        """
        Append a newly generated token with its log probability info.

        Args:
            token_id: The sampled token ID.
            logprob: Log probability of the sampled token.
            top_logprobs: Optional dict mapping token_id -> logprob for top-k alternatives.
        """
        self.output_tokens.append(token_id)
        self.output_logprobs.append(TokenLogProbInfo(
            logprob=logprob,
            top_logprobs=top_logprobs,
        ))
        self.cumulative_logprob += logprob

    def get_last_token(self) -> int:
        """Get the most recently generated token (for decode input)."""
        if self.output_tokens:
            return self.output_tokens[-1]
        # If no output yet, return last prompt token
        return self.prompt_tokens[-1]

    def get_token_ids_for_prefill(self, max_tokens: Optional[int] = None) -> list[int]:
        """
        Get token IDs for prefill phase.

        Args:
            max_tokens: Maximum tokens to return (for chunked prefill).
                        If None, returns all remaining prefill tokens.

        Returns:
            List of token IDs to process in this prefill iteration.
        """
        start = self.num_cached_tokens + self.num_prefilled_tokens
        end = self.prompt_len

        if max_tokens is not None:
            end = min(end, start + max_tokens)

        return self.prompt_tokens[start:end]

    def mark_prefilled(self, num_tokens: int) -> None:
        """Record that some tokens have been prefilled (for chunked prefill)."""
        self.num_prefilled_tokens += num_tokens

    def finish(self, reason: str = "stopped") -> None:
        """Mark the sequence as finished."""
        if reason == "length":
            self.status = SequenceStatus.FINISHED_LENGTH
        elif reason == "aborted":
            self.status = SequenceStatus.FINISHED_ABORTED
        else:
            self.status = SequenceStatus.FINISHED_STOPPED

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, prompt_len={self.prompt_len}, "
            f"output_len={self.output_len}, status={self.status.value})"
        )


@dataclass
class SequenceGroup:
    """
    A group of sequences generated from the same prompt.

    When n > 1 in sampling params, we generate multiple sequences
    from the same prompt. They share the prompt's KV cache.

    The SequenceGroup tracks:
    - The original request
    - All sequences being generated
    - Sampling parameters
    """

    request_id: str
    """Unique identifier for this request."""

    sequences: list[Sequence]
    """All sequences in this group."""

    prompt_tokens: list[int]
    """Original prompt tokens (shared by all sequences)."""

    arrival_time: float = 0.0
    """Timestamp when this request arrived (for scheduling priority)."""

    # Sampling config (stored here for convenience)
    max_tokens: int = 256
    """Maximum tokens to generate per sequence."""

    stop_token_ids: list[int] = field(default_factory=list)
    """Token IDs that trigger generation stop."""

    @property
    def num_seqs(self) -> int:
        """Number of sequences in this group."""
        return len(self.sequences)

    @property
    def is_finished(self) -> bool:
        """Check if all sequences in the group are finished."""
        return all(seq.status.is_finished for seq in self.sequences)

    @property
    def num_finished(self) -> int:
        """Number of finished sequences."""
        return sum(1 for seq in self.sequences if seq.status.is_finished)

    @property
    def num_running(self) -> int:
        """Number of running sequences."""
        return sum(1 for seq in self.sequences if seq.status == SequenceStatus.RUNNING)

    def get_unfinished_seqs(self) -> list[Sequence]:
        """Get sequences that haven't finished yet."""
        return [seq for seq in self.sequences if not seq.status.is_finished]

    def get_running_seqs(self) -> list[Sequence]:
        """Get sequences currently in running state."""
        return [seq for seq in self.sequences if seq.status == SequenceStatus.RUNNING]

    @classmethod
    def from_prompt(
        cls,
        request_id: str,
        prompt_tokens: list[int],
        n: int = 1,
        max_tokens: int = 256,
        stop_token_ids: Optional[list[int]] = None,
        arrival_time: float = 0.0,
    ) -> "SequenceGroup":
        """
        Create a SequenceGroup from a prompt.

        Args:
            request_id: Unique request identifier.
            prompt_tokens: Tokenized prompt.
            n: Number of sequences to generate.
            max_tokens: Maximum generation length.
            stop_token_ids: Tokens that stop generation.
            arrival_time: Request arrival timestamp.

        Returns:
            A new SequenceGroup with n sequences initialized.
        """
        sequences = [
            Sequence(
                seq_id=i,
                prompt_tokens=prompt_tokens.copy(),
            )
            for i in range(n)
        ]

        return cls(
            request_id=request_id,
            sequences=sequences,
            prompt_tokens=prompt_tokens,
            arrival_time=arrival_time,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids or [],
        )

    def __repr__(self) -> str:
        return (
            f"SequenceGroup(request_id={self.request_id}, "
            f"num_seqs={self.num_seqs}, finished={self.num_finished}/{self.num_seqs})"
        )
