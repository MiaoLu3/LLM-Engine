"""Tests for Sequence and SequenceGroup data structures."""

import pytest
from llm_engine.data_structures.sequence import (
    Sequence,
    SequenceGroup,
    SequenceStatus,
    TokenLogProbInfo,
)


class TestSequenceStatus:
    """Tests for SequenceStatus enum."""

    def test_is_finished_for_finished_states(self):
        """Finished states should return True for is_finished."""
        assert SequenceStatus.FINISHED_STOPPED.is_finished is True
        assert SequenceStatus.FINISHED_LENGTH.is_finished is True
        assert SequenceStatus.FINISHED_ABORTED.is_finished is True

    def test_is_finished_for_active_states(self):
        """Active states should return False for is_finished."""
        assert SequenceStatus.WAITING.is_finished is False
        assert SequenceStatus.RUNNING.is_finished is False
        assert SequenceStatus.SWAPPED.is_finished is False


class TestTokenLogProbInfo:
    """Tests for TokenLogProbInfo dataclass."""

    def test_basic_creation(self):
        """Test creating TokenLogProbInfo with just logprob."""
        info = TokenLogProbInfo(logprob=-0.5)
        assert info.logprob == -0.5
        assert info.top_logprobs is None

    def test_with_top_logprobs(self):
        """Test creating TokenLogProbInfo with top-k alternatives."""
        top_k = {100: -0.5, 200: -1.0, 300: -1.5}
        info = TokenLogProbInfo(logprob=-0.5, top_logprobs=top_k)
        assert info.logprob == -0.5
        assert info.top_logprobs == top_k
        assert len(info.top_logprobs) == 3


class TestSequence:
    """Tests for Sequence dataclass."""

    def test_basic_creation(self):
        """Test creating a basic sequence."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3, 4, 5])
        assert seq.seq_id == 0
        assert seq.prompt_tokens == [1, 2, 3, 4, 5]
        assert seq.output_tokens == []
        assert seq.output_logprobs == []
        assert seq.status == SequenceStatus.WAITING
        assert seq.cumulative_logprob == 0.0

    def test_prompt_len(self):
        """Test prompt_len property."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        assert seq.prompt_len == 3

    def test_output_len(self):
        """Test output_len property."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        assert seq.output_len == 0

        seq.append_token(10, logprob=-0.5)
        assert seq.output_len == 1

    def test_total_len(self):
        """Test total_len property."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        seq.append_token(10, logprob=-0.5)
        seq.append_token(20, logprob=-0.3)
        assert seq.total_len == 5  # 3 prompt + 2 output

    def test_append_token_basic(self):
        """Test appending a token with logprob."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        seq.append_token(token_id=100, logprob=-0.5)

        assert seq.output_tokens == [100]
        assert len(seq.output_logprobs) == 1
        assert seq.output_logprobs[0].logprob == -0.5
        assert seq.cumulative_logprob == -0.5

    def test_append_token_with_top_logprobs(self):
        """Test appending a token with top-k alternatives."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        top_k = {100: -0.5, 200: -1.0}
        seq.append_token(token_id=100, logprob=-0.5, top_logprobs=top_k)

        assert seq.output_tokens == [100]
        assert seq.output_logprobs[0].top_logprobs == top_k

    def test_append_multiple_tokens(self):
        """Test appending multiple tokens accumulates logprobs."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        seq.append_token(100, logprob=-0.5)
        seq.append_token(200, logprob=-0.3)
        seq.append_token(300, logprob=-0.2)

        assert seq.output_tokens == [100, 200, 300]
        assert len(seq.output_logprobs) == 3
        assert seq.cumulative_logprob == pytest.approx(-1.0)

    def test_get_last_token_with_output(self):
        """Test get_last_token returns last output token."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        seq.append_token(100, logprob=-0.5)
        seq.append_token(200, logprob=-0.3)
        assert seq.get_last_token() == 200

    def test_get_last_token_no_output(self):
        """Test get_last_token returns last prompt token when no output."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        assert seq.get_last_token() == 3

    def test_is_prefill(self):
        """Test is_prefill property."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        assert seq.is_prefill is True  # Not yet prefilled

        seq.num_prefilled_tokens = 3  # Mark as prefilled
        assert seq.is_prefill is False

    def test_num_tokens_to_prefill(self):
        """Test num_tokens_to_prefill calculation."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3, 4, 5])
        assert seq.num_tokens_to_prefill == 5

        seq.num_cached_tokens = 2  # 2 tokens from prefix cache
        assert seq.num_tokens_to_prefill == 3

        seq.num_prefilled_tokens = 2  # 2 more prefilled
        assert seq.num_tokens_to_prefill == 1

    def test_get_token_ids_for_prefill(self):
        """Test getting tokens for prefill phase."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3, 4, 5])
        assert seq.get_token_ids_for_prefill() == [1, 2, 3, 4, 5]

        seq.num_cached_tokens = 2
        assert seq.get_token_ids_for_prefill() == [3, 4, 5]

        # With max_tokens limit (chunked prefill)
        assert seq.get_token_ids_for_prefill(max_tokens=2) == [3, 4]

    def test_mark_prefilled(self):
        """Test marking tokens as prefilled."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3, 4, 5])
        seq.mark_prefilled(3)
        assert seq.num_prefilled_tokens == 3

        seq.mark_prefilled(2)
        assert seq.num_prefilled_tokens == 5

    def test_finish_stopped(self):
        """Test finishing with stop reason."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        seq.finish("stopped")
        assert seq.status == SequenceStatus.FINISHED_STOPPED

    def test_finish_length(self):
        """Test finishing with length reason."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        seq.finish("length")
        assert seq.status == SequenceStatus.FINISHED_LENGTH

    def test_finish_aborted(self):
        """Test finishing with aborted reason."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        seq.finish("aborted")
        assert seq.status == SequenceStatus.FINISHED_ABORTED

    def test_all_tokens(self):
        """Test all_tokens property."""
        seq = Sequence(seq_id=0, prompt_tokens=[1, 2, 3])
        seq.append_token(10, logprob=-0.5)
        seq.append_token(20, logprob=-0.3)
        assert seq.all_tokens == [1, 2, 3, 10, 20]


class TestSequenceGroup:
    """Tests for SequenceGroup dataclass."""

    def test_from_prompt_single_sequence(self):
        """Test creating SequenceGroup with n=1."""
        group = SequenceGroup.from_prompt(
            request_id="req_001",
            prompt_tokens=[1, 2, 3, 4, 5],
            n=1,
            max_tokens=100,
        )

        assert group.request_id == "req_001"
        assert group.prompt_tokens == [1, 2, 3, 4, 5]
        assert group.num_seqs == 1
        assert group.max_tokens == 100
        assert len(group.sequences) == 1
        assert group.sequences[0].seq_id == 0

    def test_from_prompt_multiple_sequences(self):
        """Test creating SequenceGroup with n>1."""
        group = SequenceGroup.from_prompt(
            request_id="req_002",
            prompt_tokens=[1, 2, 3],
            n=3,
            max_tokens=50,
        )

        assert group.num_seqs == 3
        assert [seq.seq_id for seq in group.sequences] == [0, 1, 2]
        # Each sequence should have its own copy of prompt tokens
        for seq in group.sequences:
            assert seq.prompt_tokens == [1, 2, 3]

    def test_is_finished_none_finished(self):
        """Test is_finished when no sequences are done."""
        group = SequenceGroup.from_prompt(
            request_id="req_003",
            prompt_tokens=[1, 2, 3],
            n=2,
        )
        assert group.is_finished is False

    def test_is_finished_partial(self):
        """Test is_finished when some sequences are done."""
        group = SequenceGroup.from_prompt(
            request_id="req_004",
            prompt_tokens=[1, 2, 3],
            n=2,
        )
        group.sequences[0].finish("stopped")
        assert group.is_finished is False

    def test_is_finished_all_done(self):
        """Test is_finished when all sequences are done."""
        group = SequenceGroup.from_prompt(
            request_id="req_005",
            prompt_tokens=[1, 2, 3],
            n=2,
        )
        group.sequences[0].finish("stopped")
        group.sequences[1].finish("length")
        assert group.is_finished is True

    def test_num_finished(self):
        """Test num_finished counter."""
        group = SequenceGroup.from_prompt(
            request_id="req_006",
            prompt_tokens=[1, 2, 3],
            n=3,
        )
        assert group.num_finished == 0

        group.sequences[0].finish("stopped")
        assert group.num_finished == 1

        group.sequences[2].finish("stopped")
        assert group.num_finished == 2

    def test_get_unfinished_seqs(self):
        """Test getting unfinished sequences."""
        group = SequenceGroup.from_prompt(
            request_id="req_007",
            prompt_tokens=[1, 2, 3],
            n=3,
        )
        group.sequences[1].finish("stopped")

        unfinished = group.get_unfinished_seqs()
        assert len(unfinished) == 2
        assert group.sequences[0] in unfinished
        assert group.sequences[2] in unfinished

    def test_get_running_seqs(self):
        """Test getting running sequences."""
        group = SequenceGroup.from_prompt(
            request_id="req_008",
            prompt_tokens=[1, 2, 3],
            n=3,
        )
        # Initially all are WAITING
        assert len(group.get_running_seqs()) == 0

        # Set some to RUNNING
        group.sequences[0].status = SequenceStatus.RUNNING
        group.sequences[2].status = SequenceStatus.RUNNING

        running = group.get_running_seqs()
        assert len(running) == 2

    def test_stop_token_ids(self):
        """Test stop token IDs are stored."""
        group = SequenceGroup.from_prompt(
            request_id="req_009",
            prompt_tokens=[1, 2, 3],
            stop_token_ids=[100, 101, 102],
        )
        assert group.stop_token_ids == [100, 101, 102]
