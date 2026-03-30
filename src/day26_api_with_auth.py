"""
DAY 26: API with JWT Authentication + Rate Limiting + Security + Validation
"""

from fastapi import FastAPI, HTTPException, Request, Response, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time

from day10_hybrid import HybridSearch
from day6_reranker import TransformerReranker
from day18_query_optimizer import QueryOptimizer
from day8_detector import HallucinationDetector
from day19_cache_manager import EnhancedCacheManager
from day23_prompt_security import PromptSecurity
from day25_input_validator import InputValidator
from day26_auth import AuthManager, Token, User, UserInDB

app = FastAPI(title="RAAS API with Authentication", version="7.0.0")
cache = EnhancedCacheManager()
security = PromptSecurity()
validator = InputValidator()
auth = AuthManager()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

print("\n🚀 Loading RAAS components...")
hybrid_search = HybridSearch()
reranker = TransformerReranker()
generator = QueryOptimizer()
detector = HallucinationDetector()
print("✓ All components loaded!")
print("✓ Security module loaded!")
print("✓ Input validator loaded!")
print("✓ Auth module loaded!")


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


class UserResponse(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    tier: str


# ===== AUTHENTICATION ENDPOINTS =====

@app.post("/register", response_model=UserResponse)
async def register(username: str, password: str, email: str, full_name: Optional[str] = None):
    """Register a new user."""
    if auth.get_user(username):
        raise HTTPException(status_code=400, detail="Username already registered")

    user = auth.create_user(username, password, email, full_name)

    from day19_cache_manager import UserTier
    cache.set_user_tier(username, UserTier.FREE)

    return UserResponse(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        tier=user.tier
    )


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token."""
    user = auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = auth.create_access_token(
        data={"sub": user.username, "tier": user.tier}
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=30 * 60
    )


@app.get("/users/me", response_model=UserResponse)
async def read_users_me(token: str = Depends(oauth2_scheme)):
    """Get current user info."""
    user = auth.get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    return UserResponse(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        tier=user.tier
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(
        request: Request,
        response: Response,
        req: QuestionRequest,
        token: str = Depends(oauth2_scheme)
):
    start_time = time.time()

    user = auth.get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    user_id = user.username

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

    security_result = security.detect_injection(safe_question)
    if security_result["risk_level"] == "HIGH":
        raise HTTPException(status_code=400, detail="Suspicious input detected")

    final_question = security_result["sanitized_input"]

    cached_response = cache.get_cached(req.question)
    if cached_response:
        cached_response['cached'] = True
        cached_response['security_flagged'] = security_result["detected"]
        cached_response['validation_warnings'] = validation_warnings
        cached_response['rate_limit'] = rate_limit_result
        return cached_response

    chunks = hybrid_search.adaptive_search(final_question, k=req.top_k * 10)
    reranked = reranker.rerank(final_question, chunks, top_k=req.top_k)
    answer_result = generator.generate(final_question, reranked)
    safety = detector.detect(answer_result['answer'], reranked, final_question)

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

    cache.set_cached(req.question, response_data)
    return response_data


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache": "connected" if cache.available else "fallback",
        "security": "active",
        "validator": "active",
        "auth": "active",
        "version": "7.0.0"
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
    return validator.get_validation_help()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)