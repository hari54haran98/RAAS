"""
DAY 5: FAISS Retrieval for RAAS
Complete retrieval system for all banking queries
"""

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time


class FAISSRetriever:
    """
    Complete FAISS-based retriever for all banking queries.
    Used by Day 10 hybrid search and all downstream components.
    """

    def __init__(self):
        print("=" * 60)
        print("DAY 5: COMPLETE FAISS RETRIEVER")
        print("=" * 60)

        # Load Day 4 artifacts
        print("\n📂 Loading FAISS index...")
        self.index = faiss.read_index("models/faiss_index.bin")

        print("\n📂 Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        print("\n📂 Loading chunks...")
        self.chunks_df = pd.read_csv("data/text_blocks_enriched.csv")
        self.chunk_id_to_idx = {row['chunk_id']: idx for idx, row in self.chunks_df.iterrows()}

        print(f"\n✅ Loaded {len(self.chunks_df)} chunks")
        print(f"✅ FAISS index with {self.index.ntotal} vectors")
        print("=" * 60)

    def retrieve(self, question, k=10):
        """
        Retrieve top k chunks for any question.

        Args:
            question: User question (any banking query)
            k: Number of chunks to return

        Returns:
            List of chunks with metadata and FAISS distances
        """
        start = time.time()

        # Encode question
        q_vector = self.embedder.encode([question]).astype('float32')

        # Search FAISS
        distances, indices = self.index.search(q_vector, k)

        # Get chunks
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.chunks_df):
                chunk = self.chunks_df.iloc[idx].to_dict()
                chunk['faiss_distance'] = float(dist)
                chunk['retrieval_method'] = 'faiss'
                results.append(chunk)

        elapsed = (time.time() - start) * 1000
        print(f"   ⚡ FAISS retrieval: {elapsed:.1f}ms for {len(results)} chunks")

        return results

    def retrieve_by_id(self, chunk_id):
        """Get a specific chunk by its ID."""
        if chunk_id in self.chunk_id_to_idx:
            idx = self.chunk_id_to_idx[chunk_id]
            return self.chunks_df.iloc[idx].to_dict()
        return None

    def batch_retrieve(self, questions, k=5):
        """Retrieve for multiple questions at once."""
        results = []
        for q in questions:
            results.append(self.retrieve(q, k))
        return results

    def get_stats(self):
        """Get retrieval system statistics."""
        return {
            'total_chunks': len(self.chunks_df),
            'faiss_vectors': self.index.ntotal,
            'documents': self.chunks_df['doc'].nunique(),
            'avg_chunk_size': self.chunks_df['char_count'].mean(),
            'tag_distribution': self.chunks_df['tags'].value_counts().to_dict()
        }

    def quick_test(self):
        """Test retrieval on various query types."""
        print("\n🧪 TESTING COMPLETE RETRIEVAL SYSTEM")
        print("-" * 60)

        test_queries = [
            "What is the penalty for late payment?",
            "What documents are required for loan?",
            "What is the interest rate?",
            "Is there COVID relief?"
        ]

        for q in test_queries:
            print(f"\n🔍 Query: '{q}'")
            results = self.retrieve(q, k=3)

            for i, r in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"     Doc: {r['doc']} p{r['page']}")
                print(f"     Distance: {r['faiss_distance']:.3f}")
                print(f"     Tags: {r.get('tags', 'N/A')}")
                print(f"     Text: {r['text'][:100]}...")


if __name__ == "__main__":
    # Quick test
    retriever = FAISSRetriever()
    retriever.quick_test()

    # Show stats
    print("\n📊 SYSTEM STATISTICS")
    print("-" * 60)
    stats = retriever.get_stats()
    for k, v in stats.items():
        print(f"   {k}: {v}")