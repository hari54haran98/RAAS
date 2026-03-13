import pandas as pd

df = pd.read_csv('data/pdf_pages_raw.csv')
matches = df[df['text'].str.contains('2.40%', na=False)]

print(f'Found {len(matches)} pages with 2.40%')
for i, row in matches.iterrows():
    print(f"{row['doc']} p{row['page']}: {row['text'][:200]}")