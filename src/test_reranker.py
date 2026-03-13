from day10_hybrid import HybridSearch
from day6_reranker import TransformerReranker

print("="*60)
print("TESTING HYBRID → RERANKER PIPELINE")
print("="*60)

# Get chunks from hybrid
h = HybridSearch()
chunks = h.adaptive_search('What is the penalty for late payment?', k=20)

print(f"\n📊 HYBRID SEARCH TOP 5:")
for i, c in enumerate(chunks[:5]):
    print(f"{i+1}. {c['doc']} p{c['page']}")
    if c.get('chunk_id') == 'sbi_home_loan_terms_p2_c0935':
        print(f"   ✅ TARGET CHUNK HERE (ID: {c['chunk_id']})")

# Rerank
r = TransformerReranker()
reranked = r.rerank('What is the penalty for late payment?', chunks, top_k=5)

print(f"\n📊 AFTER RERANKING TOP 5:")
for i, c in enumerate(reranked[:5]):
    print(f"{i+1}. {c['doc']} p{c['page']} - Score: {c['rerank_score']:.1f}")
    if c.get('chunk_id') == 'sbi_home_loan_terms_p2_c0935':
        print(f"   ✅ TARGET CHUNK RETAINED at rank {i+1}!")

print("\n" + "="*60)