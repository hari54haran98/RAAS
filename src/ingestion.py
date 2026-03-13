"""
DAY 1: PDF Ingestion Foundation
Clean, professional, FAANG-level
"""

import fitz
import pandas as pd
from pathlib import Path


class PDFIngestor:
    """Professional PDF ingestion with metadata preservation."""

    def __init__(self, pdf_dir="data/raw"):  # ← CHANGED to data/raw
        self.pdf_dir = Path(pdf_dir)
        self.data = []

    def extract_documents(self):
        """Extract all PDFs with full traceability."""
        print("PDF INGESTION STARTED")
        print("=" * 50)

        for pdf in self.pdf_dir.glob("*.pdf"):
            print(f"   Processing: {pdf.name}")
            with fitz.open(pdf) as doc:
                for page_num in range(len(doc)):
                    text = doc[page_num].get_text().strip()

                    if text and len(text) > 50:
                        self.data.append({
                            "doc": pdf.stem,
                            "page": page_num + 1,
                            "text": text,
                            "char_count": len(text)
                        })

        return pd.DataFrame(self.data)

    def save_and_report(self, df):
        """Save data and generate report."""
        # Save raw pages
        df.to_csv("data/pdf_pages_raw.csv", index=False)

        # Report
        print(f"\n INGESTION COMPLETE")
        print(f"   Documents: {df['doc'].nunique()}")
        print(f"   Pages: {len(df)}")
        print(f"   Avg length: {df['char_count'].mean():.0f} chars")
        print(f"   Saved: data/pdf_pages_raw.csv")
        print("\n" + "=" * 50)


# Execute
if __name__ == "__main__":
    ingestor = PDFIngestor()
    df = ingestor.extract_documents()
    ingestor.save_and_report(df)