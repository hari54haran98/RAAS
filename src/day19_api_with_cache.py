"""
DAY 19: API with Cache + Rate Limiting (FIXED)
Returns cached responses immediately — no delay
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time

from day10_hybrid import HybridSearch
from day6_reranker import TransformerReranker
from day18_query_optimizer import QueryOptimizer
from day8_detector import HallucinationDetector
from day19_cache_manager import CacheManager

# Initialize
app = FastAPI(title="RAAS API with Cache", version="3.0.0")
cache = CacheManager()

print("\n🚀 Loading RAAS components...")
hybrid_search = HybridSearch()
reranker = TransformerReranker()
generator = QueryOptimizer()
detector = HallucinationDetector()
print("✓ All components loaded!")


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[dict]
    is_safe: bool
    cached: bool
    processing_time_ms: float


@app.post("/ask")
async def ask_question(request: Request, req: QuestionRequest):
    start_time = time.time()

    # Rate limiting
    client_ip = request.client.host
    if not cache.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # ===== CHECK CACHE FIRST =====
    cached_response = cache.get_cached(req.question)
    if cached_response:
        # ✅ RETURN IMMEDIATELY — NO EXTRA PROCESSING
        cached_response['cached'] = True
        return cached_response

    # ===== NO CACHE — PROCESS QUERY =====

    # Get more chunks to ensure target appears
    chunks = hybrid_search.adaptive_search(req.question, k=req.top_k * 10)

    # Rerank
    reranked = reranker.rerank(req.question, chunks, top_k=req.top_k)

    # Generate answer
    answer_result = generator.generate(req.question, reranked)

    # Detect hallucinations
    safety = detector.detect(answer_result['answer'], reranked, req.question)

    # Prepare response
    response = {
        "question": req.question,
        "answer": answer_result['answer'],
        "confidence": answer_result['confidence'],
        "sources": [
            {
                "document": c['doc'],
                "page": c['page'],
                "score": c.get('rerank_score', 0.0)
            }
            for c in reranked
        ],
        "is_safe": not safety['has_hallucination'],
        "cached": False,
        "processing_time_ms": round((time.time() - start_time) * 1000, 2)
    }

    # Store in cache
    cache.set_cached(req.question, response)

    return response


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache": "connected" if cache.available else "fallback",
        "version": "3.0.0"
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    return cache.get_stats()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)