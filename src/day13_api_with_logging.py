"""
DAY 13: FastAPI with Professional Logging
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time

# Import components
from day10_hybrid import HybridSearch
from day6_reranker import TransformerReranker
from day7_generator import LLMGenerator
from day8_detector import HallucinationDetector
from day13_logging import RAASLogger

# Initialize
app = FastAPI(title="RAAS with Logging", version="1.0.0")
logger = RAASLogger()

# Load components
print("\n🚀 Loading RAAS components...")
hybrid_search = HybridSearch()
reranker = TransformerReranker()
generator = LLMGenerator()
detector = HallucinationDetector()
print("✓ All components loaded!")


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
    query_id: str


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    start_time = time.time()
    error = None

    try:
        # Your existing pipeline
        chunks = hybrid_search.adaptive_search(request.question, k=request.top_k * 2)
        reranked = reranker.rerank(request.question, chunks, top_k=request.top_k)
        answer_result = generator.generate(request.question, reranked)
        safety = detector.detect(answer_result['answer'], reranked, request.question)

        proc_time = (time.time() - start_time) * 1000

        sources = [
            SourceResponse(
                document=chunk['doc'],
                page=chunk['page'],
                score=chunk.get('rerank_score', 0.0)
            )
            for chunk in reranked
        ]

        response = AnswerResponse(
            question=request.question,
            answer=answer_result['answer'],
            confidence=answer_result['confidence'],
            sources=sources,
            is_safe=not safety['has_hallucination'],
            hallucination_score=safety['hallucination_score'],
            processing_time_ms=round(proc_time, 2),
            query_id=f"Q{logger.query_count + 1:06d}"
        )

        # LOG EVERYTHING
        logger.log_query(
            question=request.question,
            answer=answer_result['answer'],
            confidence=answer_result['confidence'],
            sources=[f"{s.document} p{s.page}" for s in sources],
            hallucination_score=safety['hallucination_score'],
            is_safe=not safety['has_hallucination'],
            response_time_ms=proc_time,
            metadata={"top_k": request.top_k}
        )

        return response

    except Exception as e:
        error = str(e)
        logger.log_query(
            question=request.question,
            answer="ERROR",
            confidence=0,
            sources=[],
            hallucination_score=1.0,
            is_safe=False,
            response_time_ms=(time.time() - start_time) * 1000,
            error=error
        )
        raise HTTPException(status_code=500, detail=error)


@app.get("/logs/stats")
async def get_log_stats():
    """Get logging statistics."""
    return logger.get_stats()


@app.get("/logs/report")
async def get_log_report():
    """Generate logging report."""
    logger.generate_report()
    return {"message": "Report generated in console"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)