"""
Unit tests for retrieval components
"""

import pytest
from day10_hybrid import HybridSearch
from day6_reranker import TransformerReranker


class TestRetrieval:
    """Test retrieval pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.hybrid = HybridSearch()
        self.reranker = TransformerReranker()
        self.target_chunk = 'sbi_home_loan_terms_p2_c0935'

    def test_hybrid_search_finds_target(self, sample_question):
        """Test that hybrid search finds the 2.40% penalty chunk."""
        chunks = self.hybrid.adaptive_search(sample_question, k=20)

        found = False
        for c in chunks:
            if c.get('chunk_id') == self.target_chunk:
                found = True
                break

        assert found, "Target chunk not found in hybrid search"

    def test_target_in_top_5(self, sample_question):
        """Test that target chunk is in top 5 results."""
        chunks = self.hybrid.adaptive_search(sample_question, k=20)

        for i, c in enumerate(chunks[:5]):
            if c.get('chunk_id') == self.target_chunk:
                assert True
                return

        pytest.fail("Target chunk not in top 5")

    def test_reranker_prioritizes_target(self, sample_question):
        """Test that reranker puts target chunk at top."""
        chunks = self.hybrid.adaptive_search(sample_question, k=20)
        reranked = self.reranker.rerank(sample_question, chunks, top_k=5)

        assert reranked[0].get('chunk_id') == self.target_chunk, \
            "Target chunk not at rank 1"