"""
Pytest configuration and fixtures
"""

import sys
import os
import pytest
from pathlib import Path

# Add src to path so tests can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

@pytest.fixture
def sample_question():
    """Return a sample question for testing."""
    return "What is the penalty for late payment?"

@pytest.fixture
def expected_answer():
    """Return expected answer pattern."""
    return "2.40%"