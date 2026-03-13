"""
DAY 4: Embeddings + FAISS for RAAS
Convert chunks to vectors and build search index
"""

import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import time


class EmbeddingSystem:
    """Professional embedding system for RAAS."""

    def __init__(self, input_file="data/text_blocks_enriched.csv"):
        self.input_file = Path(input_file)
        self.chunks_df = None
        self.embeddings = None
        self.index = None
        self.embedder = None

    def load_chunks(self):
        """Load Day 3 chunks."""
        print("\n📂 LOADING DAY 3 CHUNKS")
        print("=" * 50)

        self.chunks_df = pd.read_csv(self.input_file)

        print(f"✅ Loaded {len(self.chunks_df)} chunks")
        print(f"   Avg length: {self.chunks_df['char_count'].mean():.0f} chars")
        print(f"   Documents: {self.chunks_df['doc'].nunique()}")

        # Show sample
        print(f"\n🔍 SAMPLE CHUNK:")
        sample = self.chunks_df.iloc[0]
        print(f"   ID: {sample['chunk_id']}")
        print(f"   Tags: {sample['tags']}")
        print(f"   Text: {sample['text'][:80]}...")

        return self.chunks_df

    def create_embeddings(self):
        """Create embeddings for all chunks."""
        print(f"\n🔧 CREATING EMBEDDINGS")
        print("-" * 40)

        # Load embedding model
        print("   Loading model: all-MiniLM-L6-v2...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Create embeddings
        print(f"   Encoding {len(self.chunks_df)} chunks...")
        start_time = time.time()

        texts = self.chunks_df['text'].tolist()
        self.embeddings = self.embedder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        ).astype('float32')

        elapsed = time.time() - start_time

        print(f"\n✅ Embeddings created in {elapsed:.1f}s")
        print(f"   Shape: {self.embeddings.shape} (chunks × dimensions)")
        print(f"   Dimensions: {self.embeddings.shape[1]}")

        return self.embeddings

    def build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        print(f"\n🏗️  BUILDING FAISS INDEX")
        print("-" * 40)

        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for exact search

        # Add embeddings to index
        print(f"   Adding {len(self.embeddings)} vectors to index...")
        self.index.add(self.embeddings)

        print(f"✅ FAISS index built")
        print(f"   Index size: {self.index.ntotal} vectors")
        print(f"   Dimensions: {dimension}")

        return self.index

    def save_artifacts(self):
        """Save embeddings and index."""
        print(f"\n💾 SAVING ARTIFACTS")
        print("-" * 40)

        # Create data directory if needed
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

        # Save embeddings
        np.save("models/embeddings.npy", self.embeddings)
        print(f"   Saved: models/embeddings.npy")

        # Save FAISS index
        faiss.write_index(self.index, "models/faiss_index.bin")
        print(f"   Saved: models/faiss_index.bin")

        # Save mapping (lightweight copy for retrieval)
        mapping_df = self.chunks_df[['chunk_id', 'doc', 'page', 'tags', 'char_count']].copy()
        mapping_df.to_csv("data/embedding_mapping.csv", index=False)
        print(f"   Saved: data/embedding_mapping.csv")

        # Quick test
        self._quick_test()

    def _quick_test(self):
        """Quick test of the index."""
        print(f"\n🧪 QUICK INDEX TEST")
        print("-" * 40)

        # Test query
        test_query = "What is the penalty for late payment?"

        # Encode query
        query_embedding = self.embedder.encode([test_query]).astype('float32')

        # Search
        k = 3
        distances, indices = self.index.search(query_embedding, k)

        print(f"   Query: '{test_query}'")
        print(f"   Found {len(indices[0])} results:")

        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.chunks_df):
                chunk = self.chunks_df.iloc[idx]
                print(f"\n   Result {i + 1} (distance: {dist:.3f}):")
                print(f"      Doc: {chunk['doc']}, Page: {chunk['page']}")
                print(f"      Tags: {chunk['tags']}")
                print(f"      Text: {chunk['text'][:80]}...")

    def run(self):
        """Execute complete Day 4 pipeline."""
        print("🚀 DAY 4: EMBEDDINGS + FAISS")
        print("=" * 50)

        try:
            # Step 1: Load chunks
            self.load_chunks()

            # Step 2: Create embeddings
            self.create_embeddings()

            # Step 3: Build FAISS index
            self.build_faiss_index()

            # Step 4: Save artifacts
            self.save_artifacts()

            print(f"\n" + "=" * 50)
            print("✅ DAY 4 COMPLETE!")
            print("=" * 50)
            print(f"\n🎯 WHAT'S READY:")
            print(f"   1. {len(self.embeddings)} embeddings (384-dim each)")
            print(f"   2. FAISS index for <50ms similarity search")
            print(f"   3. Full traceability mapping")
            print(f"\n📁 FILES CREATED:")
            print(f"   • models/embeddings.npy")
            print(f"   • models/faiss_index.bin")
            print(f"   • data/embedding_mapping.csv")
            print(f"\n🚀 READY FOR DAY 5: Baseline Retrieval")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


# Execute
if __name__ == "__main__":
    system = EmbeddingSystem()
    system.run()