"""
DAY 19/24/25: API with Enhanced Rate Limiting + Security + Input Validation
"""

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time

from src.day10_hybrid import HybridSearch
from src.day6_reranker import TransformerReranker
from src.day18_query_optimizer import QueryOptimizer
from src.day8_detector import HallucinationDetector
from src.day19_cache_manager import EnhancedCacheManager
from src.day23_prompt_security import PromptSecurity
from src.day25_input_validator import InputValidator

# Initialize FastAPI
app = FastAPI(title="RAAS API with Enhanced Security", version="6.0.0")

# Initialize all components
cache = EnhancedCacheManager()
security = PromptSecurity()
validator = InputValidator()

print("\n🚀 Loading RAAS components...")
hybrid_search = HybridSearch()
reranker = TransformerReranker()
generator = QueryOptimizer()
detector = HallucinationDetector()
print("✓ All components loaded!")
print("✓ Security module loaded!")
print("✓ Input validator loaded!")


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
    security_flagged: bool
    validation_warnings: List[str]
    rate_limit: dict
    processing_time_ms: float


@app.post("/ask")
async def ask_question(request: Request, response: Response, req: QuestionRequest):
    start_time = time.time()
    user_id = request.client.host

    # ===== STEP 1: INPUT VALIDATION =====
    validation_result = validator.validate(req.question)

    if not validation_result.is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid input",
                "errors": validation_result.errors,
                "help": validator.get_validation_help()
            }
        )

    safe_question = validation_result.sanitized_input
    validation_warnings = validation_result.warnings

    # ===== STEP 2: RATE LIMITING =====
    rate_limit_result = cache.check_rate_limit(user_id)
    headers = cache.get_rate_limit_headers(user_id)
    for key, value in headers.items():
        response.headers[key] = value

    if not rate_limit_result["allowed"]:
        raise HTTPException(status_code=429, detail={
            "message": "Rate limit exceeded",
            "limit": rate_limit_result["limit"],
            "reset_in_seconds": rate_limit_result["reset_time"],
            "tier": rate_limit_result["tier"]
        })

    # ===== STEP 3: SECURITY CHECK =====
    security_result = security.detect_injection(safe_question)
    if security_result["risk_level"] == "HIGH":
        raise HTTPException(status_code=400, detail="Suspicious input detected")

    final_question = security_result["sanitized_input"]

    # ===== STEP 4: CHECK CACHE =====
    cached_response = cache.get_cached(req.question)
    if cached_response:
        cached_response['cached'] = True
        cached_response['security_flagged'] = security_result["detected"]
        cached_response['validation_warnings'] = validation_warnings
        cached_response['rate_limit'] = rate_limit_result
        return cached_response

    # ===== STEP 5: PROCESS QUERY =====
    chunks = hybrid_search.adaptive_search(final_question, k=req.top_k * 10)
    reranked = reranker.rerank(final_question, chunks, top_k=req.top_k)
    answer_result = generator.generate(final_question, reranked)
    safety = detector.detect(answer_result['answer'], reranked, final_question)

    # ===== STEP 6: PREPARE RESPONSE =====
    response_data = {
        "question": req.question,
        "answer": answer_result['answer'],
        "confidence": answer_result['confidence'],
        "sources": [{"document": c['doc'], "page": c['page'], "score": c.get('rerank_score', 0.0)} for c in reranked],
        "is_safe": not safety['has_hallucination'],
        "cached": False,
        "security_flagged": security_result["detected"],
        "validation_warnings": validation_warnings,
        "rate_limit": rate_limit_result,
        "processing_time_ms": round((time.time() - start_time) * 1000, 2)
    }

    # ===== STEP 7: CACHE RESPONSE =====
    cache.set_cached(req.question, response_data)
    return response_data


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache": "connected" if cache.available else "fallback",
        "security": "active",
        "validator": "active",
        "version": "6.0.0"
    }


@app.get("/cache/stats")
async def cache_stats():
    return cache.get_stats()


@app.get("/security/stats")
async def security_stats():
    return {
        "patterns_loaded": len(security.injection_patterns),
        "status": "active"
    }


@app.get("/validation/help")
async def validation_help():
    """Get input validation rules."""
    return validator.get_validation_help()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)