"""Tests for output data structures."""

import pytest
import math
from llm_engine.engine.output import (
    TopLogProb,
    TokenLogProb,
    CompletionOutput,
    RequestOutput,
    SchedulerOutputs,
)


class TestTopLogProb:
    """Tests for TopLogProb dataclass."""

    def test_basic_creation(self):
        """Test creating a TopLogProb."""
        tlp = TopLogProb(token_id=100, token="hello", logprob=-0.5)
        assert tlp.token_id == 100
        assert tlp.token == "hello"
        assert tlp.logprob == -0.5

    def test_repr(self):
        """Test string representation."""
        tlp = TopLogProb(token_id=100, token="the", logprob=-0.1234)
        repr_str = repr(tlp)
        assert "the" in repr_str
        assert "-0.1234" in repr_str


class TestTokenLogProb:
    """Tests for TokenLogProb dataclass."""

    def test_basic_creation(self):
        """Test creating a TokenLogProb without top-k."""
        tlp = TokenLogProb(token_id=100, token="hello", logprob=-0.5)
        assert tlp.token_id == 100
        assert tlp.token == "hello"
        assert tlp.logprob == -0.5
        assert tlp.top_logprobs is None

    def test_with_top_logprobs(self):
        """Test creating TokenLogProb with top-k alternatives."""
        top_k = [
            TopLogProb(token_id=100, token="the", logprob=-0.1),
            TopLogProb(token_id=200, token="a", logprob=-0.5),
            TopLogProb(token_id=300, token="an", logprob=-0.8),
        ]
        tlp = TokenLogProb(
            token_id=100,
            token="the",
            logprob=-0.1,
            top_logprobs=top_k,
        )
        assert len(tlp.top_logprobs) == 3
        assert tlp.top_logprobs[0].token == "the"


class TestCompletionOutput:
    """Tests for CompletionOutput dataclass."""

    def test_basic_creation(self):
        """Test creating a basic completion output."""
        output = CompletionOutput(
            index=0,
            text="Hello, world!",
            token_ids=[100, 200, 300],
        )
        assert output.index == 0
        assert output.text == "Hello, world!"
        assert output.token_ids == [100, 200, 300]
        assert output.logprobs == []
        assert output.cumulative_logprob == 0.0
        assert output.finish_reason is None

    def test_num_tokens(self):
        """Test num_tokens property."""
        output = CompletionOutput(
            index=0,
            text="test",
            token_ids=[1, 2, 3, 4, 5],
        )
        assert output.num_tokens == 5

    def test_with_logprobs(self):
        """Test completion with log probabilities."""
        logprobs = [
            TokenLogProb(token_id=100, token="Hello", logprob=-0.5),
            TokenLogProb(token_id=200, token=",", logprob=-0.3),
            TokenLogProb(token_id=300, token=" world", logprob=-0.2),
        ]
        output = CompletionOutput(
            index=0,
            text="Hello, world",
            token_ids=[100, 200, 300],
            logprobs=logprobs,
            cumulative_logprob=-1.0,
        )
        assert len(output.logprobs) == 3
        assert output.cumulative_logprob == -1.0

    def test_mean_logprob(self):
        """Test mean_logprob property."""
        output = CompletionOutput(
            index=0,
            text="test",
            token_ids=[1, 2, 3, 4],
            cumulative_logprob=-2.0,
        )
        assert output.mean_logprob == pytest.approx(-0.5)

    def test_mean_logprob_empty(self):
        """Test mean_logprob with no tokens."""
        output = CompletionOutput(
            index=0,
            text="",
            token_ids=[],
        )
        assert output.mean_logprob == 0.0

    def test_perplexity(self):
        """Test perplexity calculation."""
        # perplexity = exp(-mean_logprob)
        output = CompletionOutput(
            index=0,
            text="test",
            token_ids=[1, 2],
            cumulative_logprob=-2.0,  # mean = -1.0
        )
        expected_perplexity = math.exp(1.0)  # exp(-(-1.0)) = e
        assert output.perplexity == pytest.approx(expected_perplexity)

    def test_perplexity_empty(self):
        """Test perplexity with no tokens."""
        output = CompletionOutput(
            index=0,
            text="",
            token_ids=[],
        )
        assert output.perplexity == float('inf')

    def test_finish_reason(self):
        """Test finish reason."""
        output = CompletionOutput(
            index=0,
            text="test",
            token_ids=[1, 2, 3],
            finish_reason="stop",
        )
        assert output.finish_reason == "stop"


class TestRequestOutput:
    """Tests for RequestOutput dataclass."""

    def test_basic_creation(self):
        """Test creating a basic request output."""
        output = RequestOutput(
            request_id="req_001",
            prompt="Hello",
            prompt_token_ids=[100, 200],
        )
        assert output.request_id == "req_001"
        assert output.prompt == "Hello"
        assert output.prompt_token_ids == [100, 200]
        assert output.outputs == []
        assert output.finished is False

    def test_prompt_len(self):
        """Test prompt_len property."""
        output = RequestOutput(
            request_id="req_001",
            prompt="Hello",
            prompt_token_ids=[100, 200, 300],
        )
        assert output.prompt_len == 3

    def test_num_outputs(self):
        """Test num_outputs property."""
        output = RequestOutput(
            request_id="req_001",
            prompt="Hello",
            prompt_token_ids=[100],
        )
        assert output.num_outputs == 0

        output.add_output(CompletionOutput(index=0, text="Hi", token_ids=[1]))
        assert output.num_outputs == 1

    def test_add_output(self):
        """Test adding completion outputs."""
        output = RequestOutput(
            request_id="req_001",
            prompt="Hello",
            prompt_token_ids=[100],
        )

        comp1 = CompletionOutput(index=0, text="Hi", token_ids=[1, 2])
        comp2 = CompletionOutput(index=1, text="Hey", token_ids=[3, 4])

        output.add_output(comp1)
        output.add_output(comp2)

        assert len(output.outputs) == 2
        assert output.outputs[0].text == "Hi"
        assert output.outputs[1].text == "Hey"

    def test_get_best_output(self):
        """Test getting best output by cumulative logprob."""
        output = RequestOutput(
            request_id="req_001",
            prompt="Hello",
            prompt_token_ids=[100],
        )

        comp1 = CompletionOutput(
            index=0, text="Hi", token_ids=[1], cumulative_logprob=-1.5
        )
        comp2 = CompletionOutput(
            index=1, text="Hey", token_ids=[2], cumulative_logprob=-0.8
        )
        comp3 = CompletionOutput(
            index=2, text="Hello", token_ids=[3], cumulative_logprob=-2.0
        )

        output.add_output(comp1)
        output.add_output(comp2)
        output.add_output(comp3)

        best = output.get_best_output()
        assert best.index == 1  # Highest logprob = -0.8
        assert best.text == "Hey"

    def test_get_best_output_empty(self):
        """Test get_best_output with no outputs."""
        output = RequestOutput(
            request_id="req_001",
            prompt="Hello",
            prompt_token_ids=[100],
        )
        assert output.get_best_output() is None


class TestSchedulerOutputs:
    """Tests for SchedulerOutputs dataclass."""

    def test_basic_creation(self):
        """Test creating scheduler outputs."""
        outputs = SchedulerOutputs(
            scheduled_seq_groups=[],
            num_prefill_tokens=100,
            num_decode_tokens=50,
        )
        assert outputs.scheduled_seq_groups == []
        assert outputs.num_prefill_tokens == 100
        assert outputs.num_decode_tokens == 50

    def test_num_batched_tokens(self):
        """Test num_batched_tokens property."""
        outputs = SchedulerOutputs(
            scheduled_seq_groups=[],
            num_prefill_tokens=100,
            num_decode_tokens=50,
        )
        assert outputs.num_batched_tokens == 150

    def test_is_empty(self):
        """Test is_empty property."""
        outputs = SchedulerOutputs(
            scheduled_seq_groups=[],
            num_prefill_tokens=0,
            num_decode_tokens=0,
        )
        assert outputs.is_empty is True

        outputs = SchedulerOutputs(
            scheduled_seq_groups=["dummy"],
            num_prefill_tokens=10,
            num_decode_tokens=0,
        )
        assert outputs.is_empty is False

    def test_default_block_operations(self):
        """Test default empty block operation dicts."""
        outputs = SchedulerOutputs(
            scheduled_seq_groups=[],
            num_prefill_tokens=0,
            num_decode_tokens=0,
        )
        assert outputs.blocks_to_swap_in == {}
        assert outputs.blocks_to_swap_out == {}
        assert outputs.blocks_to_copy == {}

    def test_with_block_operations(self):
        """Test scheduler outputs with block operations."""
        outputs = SchedulerOutputs(
            scheduled_seq_groups=[],
            num_prefill_tokens=0,
            num_decode_tokens=10,
            blocks_to_swap_in={0: 100, 1: 101},
            blocks_to_swap_out={5: 200},
            blocks_to_copy={10: 20},
        )
        assert outputs.blocks_to_swap_in == {0: 100, 1: 101}
        assert outputs.blocks_to_swap_out == {5: 200}
        assert outputs.blocks_to_copy == {10: 20}
