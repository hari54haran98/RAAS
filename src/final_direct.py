"""
FINAL DIRECT TEST - No API, pure components
"""

from day10_hybrid import HybridSearch
from day6_reranker import TransformerReranker
from day18_query_optimizer import QueryOptimizer
from day8_detector import HallucinationDetector

print("="*60)
print("FINAL DIRECT PIPELINE TEST")
print("="*60)

# Load components
print("\n📦 Loading components...")
hybrid = HybridSearch()
reranker = TransformerReranker()
generator = QueryOptimizer()
detector = HallucinationDetector()
print("✅ All components loaded")

# Test the critical query
question = "What is the penalty for late payment?"
print(f"\n🔍 Testing: '{question}'")

# Run pipeline
chunks = hybrid.adaptive_search(question, k=20)
print(f"📥 Hybrid search: {len(chunks)} chunks")

reranked = reranker.rerank(question, chunks, top_k=3)
print(f"🔄 Reranker top chunks:")
for i, c in enumerate(reranked):
    print(f"   {i+1}. {c['doc']} p{c['page']} - Score: {c.get('rerank_score', 0):.1f}")

answer = generator.generate(question, reranked)
print(f"🤖 Generator: answer generated")

safety = detector.detect(answer['answer'], reranked, question)
print(f"🛡️ Detector: safety check complete")

# Results
print("\n" + "="*60)
print("📊 FINAL RESULT")
print("="*60)
print(f"\nANSWER: {answer['answer']}")
print(f"\nCONFIDENCE: {answer['confidence']}")
print(f"SOURCES: {answer.get('sources', [])}")
print(f"SAFE: {not safety['has_hallucination']}")
print("="*60)