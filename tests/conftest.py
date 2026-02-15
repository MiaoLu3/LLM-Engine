"""Pytest configuration and shared fixtures."""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


@pytest.fixture
def sample_prompt_tokens():
    """Sample prompt token IDs for testing."""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def sample_output_tokens():
    """Sample output token IDs for testing."""
    return [100, 101, 102, 103, 104]


@pytest.fixture
def sample_logprobs():
    """Sample log probabilities for testing."""
    return [-0.5, -0.3, -0.7, -0.2, -0.4]
