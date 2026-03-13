from day10_hybrid import HybridSearch

h = HybridSearch()
chunks = h.adaptive_search('What is the penalty for late payment?', k=20)

print('Chunks received by reranker:')
for i, c in enumerate(chunks[:5]):
    print(f'{i+1}. {c["doc"]} p{c["page"]}')
    if c.get('chunk_id') == 'sbi_home_loan_terms_p2_c0935':
        print('   ✅ TARGET CHUNK IS HERE!')