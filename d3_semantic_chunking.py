"""
DAY 3: Semantic Chunking for RAAS
Split pages into smart, RAG-optimized chunks with banking tags
"""

import pandas as pd
import re
from pathlib import Path


class BankingChunker:
    """Smart chunking for banking documents with header filtering."""

    def __init__(self, input_file="data/pdf_pages_raw.csv", output_file="data/text_blocks_enriched.csv"):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.df = None
        self.chunks_data = []

    def load_pages(self):
        """Load Day 1 output."""
        print("📂 LOADING DAY 1 PAGES")
        print("=" * 50)

        self.df = pd.read_csv(self.input_file)
        print(f"✅ Loaded {len(self.df)} pages from {self.df['doc'].nunique()} documents")
        return self.df

    def is_header(self, text):
        """Filter out headers, keep actual content."""
        text = text.strip()
        if len(text) < 50:
            return True

        text_lower = text.lower()

        # Common banking headers to filter
        headers = ['location:', 'sr.no.', 'page', 'index', 'table of contents']
        if any(text_lower.startswith(h) for h in headers):
            return True

        # Skip ALL CAPS headers (short text, all uppercase)
        if len(text) < 100 and text.isupper():
            return True

        return False

    def chunk_by_sentences(self, text, max_sentences=4):
        """Split text into sentence-based chunks."""
        # Smart sentence splitting (preserve numbers like 2.40%)
        sentences = re.split(r'(?<=[.!?])\s+(?![0-9])', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            # Start new chunk if we have enough sentences or length
            if (len(current_chunk) >= max_sentences or
                    (current_length + sentence_length > 500 and current_chunk)):

                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) > 100:  # Avoid tiny chunks
                        chunks.append(chunk_text)
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) > 100:
                chunks.append(chunk_text)

        return chunks

    def extract_tags(self, text):
        """Extract banking-specific tags from text."""
        tags = []
        text_lower = text.lower()

        # Banking categories
        if any(word in text_lower for word in ['penalty', 'late payment', 'fine', 'charge']):
            tags.append('penalty')
        if any(word in text_lower for word in ['interest', 'rate', '% per annum', 'apr']):
            tags.append('interest')
        if any(word in text_lower for word in ['document', 'required', 'submit', 'provide']):
            tags.append('documents')
        if re.search(r'\d+\.?\d*%', text):
            tags.append('has_percentage')
        if 'clause' in text_lower or 'section' in text_lower:
            tags.append('legal')

        return '|'.join(tags) if tags else 'general'

    def process_all_pages(self):
        """Process all pages into semantic chunks."""
        print("\n🔪 SEMANTIC CHUNKING STARTED")
        print("=" * 50)

        chunk_counter = 0

        for idx, row in self.df.iterrows():
            # Skip headers
            if self.is_header(row['text']):
                continue

            # Split into sentence-based chunks
            text_chunks = self.chunk_by_sentences(row['text'])

            for chunk_text in text_chunks:
                chunk_counter += 1
                self.chunks_data.append({
                    'chunk_id': f"{row['doc']}_p{row['page']}_c{chunk_counter:04d}",
                    'doc': row['doc'],
                    'page': row['page'],
                    'text': chunk_text,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'tags': self.extract_tags(chunk_text)
                })

        # Create DataFrame
        chunks_df = pd.DataFrame(self.chunks_data)

        # Save
        chunks_df.to_csv(self.output_file, index=False)

        # Report
        print(f"\n✅ CHUNKING COMPLETE")
        print(f"   Input pages: {len(self.df)}")
        print(f"   Output chunks: {len(chunks_df)}")
        print(f"   Avg chunk size: {chunks_df['char_count'].mean():.0f} chars")

        # RAG readiness
        rag_ready = chunks_df['char_count'].between(200, 500).sum()
        print(f"   RAG-ready (200-500 chars): {rag_ready} ({rag_ready / len(chunks_df) * 100:.1f}%)")

        # Tag distribution
        print(f"\n🏷️  BANKING CONTENT FOUND:")
        all_tags = '|'.join(chunks_df['tags'].tolist()).split('|')
        from collections import Counter
        tag_counts = Counter(all_tags)
        for tag in ['penalty', 'interest', 'documents', 'has_percentage', 'legal', 'general']:
            count = tag_counts.get(tag, 0)
            if count > 0:
                pct = (count / len(chunks_df)) * 100
                print(f"   • {tag}: {count} chunks ({pct:.1f}%)")

        print(f"\n💾 Saved: {self.output_file}")
        print("\n" + "=" * 50)

        return chunks_df

    def run(self):
        """Execute Day 3 pipeline."""
        self.load_pages()
        chunks_df = self.process_all_pages()
        print("\n✅ DAY 3 COMPLETE: Semantic Chunking")
        print("🚀 Ready for Day 4: Embeddings + FAISS")
        return chunks_df


# Execute
if __name__ == "__main__":
    chunker = BankingChunker()
    chunks_df = chunker.run()