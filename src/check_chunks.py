import pandas as pd

df = pd.read_csv('data/text_blocks_enriched.csv')
page_chunks = df[df['chunk_id'].str.contains('sbi_home_loan_terms_p2', na=False)]

print(f'Found {len(page_chunks)} chunks for sbi_home_loan_terms page 2')
for i, row in page_chunks.iterrows():
    print(f"\nChunk {i}: {row['chunk_id']}")
    print(f"Text: {row['text'][:200]}...")