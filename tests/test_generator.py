"""
Unit tests for generator component
"""

import pytest
from day18_query_optimizer import QueryOptimizer


class TestGenerator:
    """Test answer generator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.generator = QueryOptimizer()

    def test_generator_returns_240(self, sample_question, expected_answer):
        """Test that generator returns 2.40% in answer."""
        # Create mock chunks with target
        chunks = [{
            'chunk_id': 'sbi_home_loan_terms_p2_c0935',
            'doc': 'sbi_home_loan_terms',
            'page': 2,
            'text': 'Penalty: 2.40% per annum',
            'rerank_score': 9989.6
        }]

        result = self.generator.generate(sample_question, chunks)

        assert expected_answer in result['answer'], \
            f"Expected {expected_answer} not in answer"
        assert result['confidence'] > 0.9, \
            f"Confidence too low: {result['confidence']}"

    def test_generator_handles_no_context(self):
        """Test generator handles empty chunks."""
        result = self.generator.generate("test question", [])
        assert 'NOT FOUND' in result['answer'].upper()
        assert result['confidence'] == 0.1