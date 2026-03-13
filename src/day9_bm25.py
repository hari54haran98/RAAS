"""
DAY 9: BM25 Keyword Index for RAAS
Exact keyword search to complement FAISS
"""

import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
import re
from pathlib import Path


class BM25Index:
    """BM25 keyword search index for banking chunks."""

    def __init__(self, chunks_path="data/text_blocks_enriched.csv"):
        print("=" * 60)
        print("DAY 9: BM25 KEYWORD INDEX")
        print("=" * 60)

        # Load chunks
        print("\n📂 Loading chunks...")
        self.chunks_df = pd.read_csv(chunks_path)
        print(f"✓ Loaded {len(self.chunks_df)} chunks")

        # Prepare texts
        print("\n🔧 Tokenizing chunks for BM25...")
        self.chunk_texts = self.chunks_df['text'].tolist()
        self.tokenized_chunks = [self._tokenize(text) for text in self.chunk_texts]

        # Build BM25 index
        print("\n🏗️ Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        print("✓ BM25 index built successfully")

        # Save index
        self._save_index()

    def _tokenize(self, text):
        """Simple tokenizer for BM25."""
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def search(self, query, k=10):
        """Search BM25 index for top k chunks."""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            chunk = self.chunks_df.iloc[idx]
            results.append({
                'chunk_id': chunk['chunk_id'],
                'doc': chunk['doc'],
                'page': chunk['page'],
                'text': chunk['text'],
                'tags': chunk['tags'],
                'bm25_score': float(scores[idx]),
                'method': 'bm25'
            })

        return results

    def _save_index(self):
        """Save BM25 index to file."""
        save_path = Path("models/bm25_index.pkl")
        save_path.parent.mkdir(exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunks_df': self.chunks_df
            }, f)

        print(f"\n💾 BM25 index saved to {save_path}")

    def load_index(self):
        """Load BM25 index from file."""
        load_path = Path("models/bm25_index.pkl")
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks_df = data['chunks_df']
        print(f"📂 BM25 index loaded from {load_path}")
        return self


# Quick test
if __name__ == "__main__":
    print("\n🧪 TESTING BM25 INDEX")
    print("-" * 40)

    # Create index
    bm25 = BM25Index()

    # Test search
    test_queries = [
        "2.40% penalty",
        "PAN card document",
        "interest rate"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        results = bm25.search(query, k=3)

        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['doc']} p{r['page']} (score: {r['bm25_score']:.2f})")

    print("\n" + "=" * 60)
    print("✅ DAY 9 COMPLETE: BM25 Index Ready")
    print("=" * 60)