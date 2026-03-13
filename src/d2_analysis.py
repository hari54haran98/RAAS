"""
DAY 2: Quick Analysis for RAVEN-RAAS
RAG Readiness Analysis
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/pdf_pages_raw.csv")

print(" DAY 2: RAG READINESS ANALYSIS")
print("=" * 50)

# Basic stats
print(f" Documents: {df['doc'].nunique()}")
print(f" Pages: {len(df)}")
print(f" Avg length: {df['char_count'].mean():.0f} chars")

# RAG optimization
print(f"\n RAG OPTIMIZATION NEED:")
lengths = df['char_count']

categories = [
    ("Ready (100-500)", (100, 500)),
    ("Light chunking (500-1000)", (500, 1000)),
    ("Chunking needed (1000-2000)", (1000, 2000)),
    ("Heavy chunking (>2000)", (2000, np.inf))
]

for label, (low, high) in categories:
    if high == np.inf:
        count = (lengths >= low).sum()
    else:
        count = lengths.between(low, high).sum()

    pct = (count / len(df)) * 100
    print(f"   • {label}: {count} pages ({pct:.1f}%)")

print(f"\n RECOMMENDATION:")
avg_len = df['char_count'].mean()
if avg_len > 2000:
    print("   USE SENTENCE-BASED CHUNKING")
    print("   • Target: 3-5 sentences per chunk")
    print("   • Filter out headers/TOC")
    print("   • Preserve banking clauses")
elif avg_len > 1000:
    print("   Use paragraph-based chunking")
else:
    print("   Pages are RAG-ready")

print("\n" + "=" * 50)