"""
Unit tests for hallucination detector
"""

import pytest
from day8_detector import HallucinationDetector


class TestDetector:
    """Test hallucination detection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.detector = HallucinationDetector()
        self.sample_chunks = [{
            'doc': 'sbi_home_loan_terms',
            'page': 2,
            'text': 'Penalty: 2.40% per annum'
        }]

    def test_detector_accepts_valid_answer(self):
        """Test valid answer passes detector."""
        answer = "According to sbi_home_loan_terms, penalty is 2.40%"
        result = self.detector.detect(answer, self.sample_chunks, "test")

        assert not result['has_hallucination'], \
            f"Valid answer flagged as hallucination (score: {result['hallucination_score']})"
        assert result['hallucination_score'] < 0.3

    def test_detector_rejects_hallucinated_answer(self):
        """Test hallucinated answer fails detector."""
        answer = "I think the penalty is probably 5% for all loans"
        result = self.detector.detect(answer, self.sample_chunks, "test")

        assert result['has_hallucination'], \
            "Hallucinated answer not detected"
        assert result['hallucination_score'] > 0.3