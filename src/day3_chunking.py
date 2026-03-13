"""
DAY 3: Semantic Chunking for RAAS (FIXED - Keeps Page Content)
Better preservation of percentages and penalty clauses
"""

import pandas as pd
import re
from pathlib import Path
from collections import Counter


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
        """Filter out headers, but KEEP page content."""
        text = text.strip()
        if len(text) < 30:  # Very short text is likely just a header
            return True

        text_lower = text.lower()

        # List of headers to filter - REMOVED 'page' from this list
        headers = ['location:', 'sr.no.', 'index', 'table of contents']
        if any(text_lower.startswith(h) for h in headers):
            return True

        # Skip ALL CAPS headers (short text, all uppercase)
        if len(text) < 100 and text.isupper() and not any(c.isdigit() for c in text):
            return True

        # Only filter pure page numbers like "Page 2 of 57" with no other content
        if text_lower.startswith('page') and len(text) < 30 and 'of' in text_lower:
            return True

        return False

    def chunk_by_sentences(self, text, max_sentences=3):
        """
        Split text into sentence-based chunks.
        FIXED: Better preservation of percentages and bullet points.
        """
        # First, protect percentage patterns (keep them together)
        text = re.sub(r'(\d+\.?\d*)\s*%', r'\1%', text)

        # Split by sentence endings, but keep bullet points together
        sentences = re.split(r'(?<=[.!?])\s+(?![0-9])', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            # If this sentence contains a percentage, prioritize keeping it whole
            has_percentage = '%' in sentence

            # Start new chunk if:
            # - We have enough sentences AND
            # - Adding this sentence would make chunk too long OR
            # - This sentence has a percentage and current chunk already has one
            if (len(current_chunk) >= max_sentences and
                    (current_length + sentence_length > 500 or
                     (has_percentage and any('%' in s for s in current_chunk)))):

                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) > 100:
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
        print("\n🔪 SEMANTIC CHUNKING STARTED (FIXED VERSION - KEEPS PAGE CONTENT)")
        print("=" * 50)

        chunk_counter = 0

        for idx, row in self.df.iterrows():
            # Skip headers, but KEEP page content even if it starts with "Page"
            if self.is_header(row['text']):
                continue

            # Split into sentence-based chunks with better preservation
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
        print("\n✅ DAY 3 COMPLETE: Semantic Chunking (FIXED - KEEPS PAGE CONTENT)")
        print("🚀 Ready for Day 4: Embeddings + FAISS")
        return chunks_df


# Execute
if __name__ == "__main__":
    chunker = BankingChunker()
    chunks_df = chunker.run()

