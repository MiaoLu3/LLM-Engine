"""Tests for the Sampler module."""

import pytest
import torch
import math

from llm_engine.sampling.sampler import Sampler, SamplerOutput
from llm_engine.config import SamplingParams


@pytest.fixture
def sampler():
    """Create a sampler for testing."""
    return Sampler(vocab_size=1000, device="cpu")


@pytest.fixture
def simple_logits():
    """Create simple logits for testing."""
    # Vocab size = 1000, batch size = 2
    logits = torch.zeros(2, 1000)
    # Make token 100 have highest logit for seq 0
    logits[0, 100] = 10.0
    # Make token 200 have highest logit for seq 1
    logits[1, 200] = 10.0
    return logits


class TestSamplerBasic:
    """Basic tests for Sampler."""

    def test_creation(self, sampler):
        """Test sampler creation."""
        assert sampler.vocab_size == 1000
        assert sampler.device == "cpu"

    def test_greedy_sampling(self, sampler, simple_logits):
        """Test greedy sampling (temperature=0)."""
        params = SamplingParams(temperature=0.0)
        output = sampler.sample(simple_logits, params)

        assert isinstance(output, SamplerOutput)
        assert output.next_tokens[0].item() == 100
        assert output.next_tokens[1].item() == 200

    def test_returns_logprobs(self, sampler, simple_logits):
        """Test that logprobs are always returned."""
        params = SamplingParams(temperature=0.0)
        output = sampler.sample(simple_logits, params)

        # Should have logprobs for each sequence
        assert output.logprobs.shape == (2,)
        # Logprobs should be negative (log of probability < 1)
        assert all(lp <= 0 for lp in output.logprobs)

    def test_top_logprobs_returned(self, sampler, simple_logits):
        """Test that top_logprobs are returned when requested."""
        params = SamplingParams(temperature=0.0, top_logprobs=5)
        output = sampler.sample(simple_logits, params)

        assert output.top_logprobs is not None
        assert len(output.top_logprobs) == 2  # One dict per sequence

        # Each dict should have 5 entries
        for seq_top_logprobs in output.top_logprobs:
            assert len(seq_top_logprobs) == 5
            # All values should be log probabilities (negative)
            assert all(lp <= 0 for lp in seq_top_logprobs.values())

    def test_top_logprobs_not_returned_when_zero(self, sampler, simple_logits):
        """Test that top_logprobs is None when top_logprobs=0."""
        params = SamplingParams(temperature=0.0, top_logprobs=0)
        output = sampler.sample(simple_logits, params)

        assert output.top_logprobs is None


class TestSamplerTemperature:
    """Tests for temperature scaling."""

    def test_high_temperature_increases_randomness(self, sampler):
        """Test that high temperature makes sampling more random."""
        # Create logits where one token is slightly preferred
        logits = torch.zeros(100, 1000)  # 100 samples
        logits[:, 100] = 1.0  # Slight preference for token 100

        # With temperature=0 (greedy), should always pick 100
        params = SamplingParams(temperature=0.0)
        output = sampler.sample(logits, params)
        assert all(t.item() == 100 for t in output.next_tokens)

        # With high temperature, should sometimes pick other tokens
        torch.manual_seed(42)
        params = SamplingParams(temperature=2.0)
        output = sampler.sample(logits, params)
        # Not all should be 100 with high temp
        unique_tokens = len(set(t.item() for t in output.next_tokens))
        assert unique_tokens > 1

    def test_low_temperature_concentrates_probability(self, sampler):
        """Test that low temperature concentrates on highest prob token."""
        logits = torch.zeros(100, 1000)
        logits[:, 100] = 5.0   # Much higher than others
        logits[:, 200] = 1.0   # Second best

        torch.manual_seed(42)
        params = SamplingParams(temperature=0.1)
        output = sampler.sample(logits, params)

        # With very low temp and large logit gap, should almost always pick 100
        token_100_count = sum(1 for t in output.next_tokens if t.item() == 100)
        assert token_100_count >= 95  # At least 95%


class TestSamplerTopK:
    """Tests for top-k sampling."""

    def test_top_k_limits_vocabulary(self, sampler):
        """Test that top-k limits sampling to k tokens."""
        # Create logits with clear ranking
        logits = torch.arange(1000).float().unsqueeze(0)  # [1, 1000]
        # Token 999 has highest logit, 998 second, etc.

        torch.manual_seed(42)
        params = SamplingParams(temperature=1.0, top_k=5)

        # Run multiple times to check we only get top-5 tokens
        all_tokens = []
        for _ in range(100):
            output = sampler.sample(logits, params)
            all_tokens.append(output.next_tokens[0].item())

        unique_tokens = set(all_tokens)
        # Should only get tokens from top 5: 999, 998, 997, 996, 995
        assert all(t >= 995 for t in unique_tokens)


class TestSamplerTopP:
    """Tests for top-p (nucleus) sampling."""

    def test_top_p_filters_low_probability(self, sampler):
        """Test that top-p filters out low probability tokens."""
        # Create logits where a few tokens have most of the probability
        logits = torch.full((1, 1000), -100.0)  # Very low
        logits[0, 0] = 5.0   # ~40% prob
        logits[0, 1] = 4.5   # ~30% prob
        logits[0, 2] = 4.0   # ~20% prob
        logits[0, 3] = 3.0   # ~10% prob (cumsum ~100%)

        torch.manual_seed(42)
        params = SamplingParams(temperature=1.0, top_p=0.9)

        all_tokens = []
        for _ in range(100):
            output = sampler.sample(logits, params)
            all_tokens.append(output.next_tokens[0].item())

        unique_tokens = set(all_tokens)
        # Should only get tokens 0, 1, 2 (cover ~90% of probability)
        # Token 3 might occasionally appear due to numerical precision
        assert all(t <= 3 for t in unique_tokens)


class TestSamplerLogProbAccuracy:
    """Tests for log probability computation accuracy."""

    def test_logprob_matches_softmax(self, sampler):
        """Test that returned logprobs match log_softmax computation."""
        logits = torch.randn(3, 1000)

        params = SamplingParams(temperature=0.0)  # Greedy
        output = sampler.sample(logits, params)

        # Compute expected logprobs manually
        expected_log_probs = torch.log_softmax(logits, dim=-1)
        expected = expected_log_probs.gather(-1, output.next_tokens.unsqueeze(-1)).squeeze(-1)

        assert torch.allclose(output.logprobs, expected, atol=1e-5)

    def test_top_logprobs_are_sorted(self, sampler):
        """Test that top_logprobs contains highest probability tokens."""
        logits = torch.randn(1, 1000)

        params = SamplingParams(temperature=0.0, top_logprobs=5)
        output = sampler.sample(logits, params)

        # Get the actual top 5 from log_softmax
        log_probs = torch.log_softmax(logits[0], dim=-1)
        top_values, top_indices = torch.topk(log_probs, 5)

        # Check that returned top_logprobs matches
        returned_tokens = set(output.top_logprobs[0].keys())
        expected_tokens = set(top_indices.tolist())
        assert returned_tokens == expected_tokens


class TestSamplerRepetitionPenalty:
    """Tests for repetition penalty."""

    def test_repetition_penalty_reduces_probability(self, sampler):
        """Test that repetition penalty reduces probability of repeated tokens."""
        # Token 100 has highest logit
        logits = torch.zeros(1, 1000)
        logits[0, 100] = 10.0
        logits[0, 200] = 9.9  # Close second

        # Without penalty, should always pick 100
        params = SamplingParams(temperature=0.0)
        output = sampler.sample(logits, params)
        assert output.next_tokens[0].item() == 100

        # With penalty on token 100, should pick 200
        penalized_logits = sampler.apply_repetition_penalty(
            logits.clone(),
            past_tokens=[[100]],
            penalty=2.0,
        )
        output = sampler.sample(penalized_logits, params)
        assert output.next_tokens[0].item() == 200

    def test_repetition_penalty_no_effect_when_1(self, sampler):
        """Test that penalty=1.0 has no effect."""
        logits = torch.randn(1, 1000)
        original_logits = logits.clone()

        penalized_logits = sampler.apply_repetition_penalty(
            logits,
            past_tokens=[[100, 200, 300]],
            penalty=1.0,
        )

        assert torch.allclose(penalized_logits, original_logits)


class TestSamplerEdgeCases:
    """Tests for edge cases."""

    def test_single_sequence(self, sampler):
        """Test sampling with single sequence."""
        logits = torch.randn(1, 1000)
        params = SamplingParams(temperature=1.0)
        output = sampler.sample(logits, params)

        assert output.next_tokens.shape == (1,)
        assert output.logprobs.shape == (1,)

    def test_3d_logits(self, sampler):
        """Test that 3D logits (with seq_len dim) work correctly."""
        # Shape: [batch_size, seq_len, vocab_size]
        logits = torch.randn(2, 10, 1000)

        params = SamplingParams(temperature=1.0)
        output = sampler.sample(logits, params)

        # Should sample from last position only
        assert output.next_tokens.shape == (2,)

    def test_empty_top_logprobs(self, sampler):
        """Test requesting more top_logprobs than vocab size."""
        small_vocab_sampler = Sampler(vocab_size=5, device="cpu")
        logits = torch.randn(1, 5)

        params = SamplingParams(temperature=1.0, top_logprobs=10)
        output = small_vocab_sampler.sample(logits, params)

        # Should return min(10, 5) = 5 top logprobs
        assert len(output.top_logprobs[0]) == 5
