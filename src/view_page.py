import pandas as pd

df = pd.read_csv('data/pdf_pages_raw.csv')
page = df[(df['doc'] == 'sbi_home_loan_terms') & (df['page'] == 2)]

if len(page) > 0:
    text = page.iloc[0]['text']
    print("Full page text:")
    print("-" * 50)
    print(text)
    print("-" * 50)

    # Search for variations
    variations = ['2.40', '2.40%', '2.40 %', '2.40 per cent']
    for v in variations:
        if v in text:
            print(f"✓ Found '{v}'")
        else:
            print(f"✗ Not found '{v}'")
else:
    print("Page not found")