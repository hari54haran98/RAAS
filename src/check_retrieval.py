from day10_hybrid import HybridSearch

h = HybridSearch()
chunks = h.adaptive_search('What is the penalty for late payment?', k=10)

print("Top 10 chunks:")
for i, c in enumerate(chunks[:5]):
    print(f"{i+1}. {c['doc']} p{c['page']} - {c['text'][:50]}...")
    if '2.40%' in c['text']:
        print("   ✅ CONTAINS 2.40%")