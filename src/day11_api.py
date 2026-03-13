"""
DAY 11: FastAPI Backend for RAAS (FIXED)
Uses Day 18 generator and retrieves more chunks for better results
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time

# Import CORRECT components
from day10_hybrid import HybridSearch
from day6_reranker import TransformerReranker
from day18_query_optimizer import QueryOptimizer  # ← FIXED: Using Day 18
from day8_detector import HallucinationDetector    # ← FIXED: Using updated detector

# Initialize FastAPI
app = FastAPI(
    title="RAAS - Retrieval Augmented Answer Safety (FIXED)",
    description="Banking document Q&A with hallucination detection",
    version="2.0.0"
)

# Initialize RAAS components (load once at startup)
print("\n🚀 Loading RAAS components...")
hybrid_search = HybridSearch()
reranker = TransformerReranker()
generator = QueryOptimizer()  # ← FIXED: Using Day 18 generator
detector = HallucinationDetector()  # ← FIXED: Using updated detector
print("✓ All components loaded successfully!")


# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class SourceResponse(BaseModel):
    document: str
    page: int
    score: float


class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[SourceResponse]
    is_safe: bool
    hallucination_score: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    components: dict
    version: str


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RAAS API (FIXED)",
        "version": "2.0.0",
        "description": "Banking document Q&A with hallucination detection",
        "endpoints": {
            "/ask": "POST - Ask a question",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get answer with safety check.
    """
    start_time = time.time()

    try:
        # Step 1: Hybrid search - get MORE chunks (30 instead of 6)
        chunks = hybrid_search.adaptive_search(request.question, k=request.top_k * 10)

        # Step 2: Rerank
        reranked = reranker.rerank(request.question, chunks, top_k=request.top_k)

        # Step 3: Generate answer using FIXED Day 18 generator
        answer_result = generator.generate(request.question, reranked)

        # Step 4: Detect hallucinations using FIXED detector (threshold 0.3)
        safety = detector.detect(answer_result['answer'], reranked, request.question)

        # Calculate processing time
        proc_time = (time.time() - start_time) * 1000

        # Format sources
        sources = [
            SourceResponse(
                document=chunk['doc'],
                page=chunk['page'],
                score=chunk.get('rerank_score', 0.0)
            )
            for chunk in reranked
        ]

        return AnswerResponse(
            question=request.question,
            answer=answer_result['answer'],
            confidence=answer_result['confidence'],
            sources=sources,
            is_safe=not safety['has_hallucination'],
            hallucination_score=safety['hallucination_score'],
            processing_time_ms=round(proc_time, 2)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if all components are healthy."""
    return HealthResponse(
        status="healthy",
        components={
            "hybrid_search": "loaded",
            "reranker": "loaded",
            "generator": "loaded (Day 18)",
            "detector": "loaded (threshold 0.3)"
        },
        version="2.0.0"
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown."""
    # Only needed if generator has a close method
    if hasattr(generator, 'close'):
        generator.close()
    print("✓ Connections closed")


# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)