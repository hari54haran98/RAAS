"""
DAY 27: RAAS API with JWT Authentication, Rate Limiting, Redis, and HTTPS
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional
import time
import jwt
import bcrypt
from datetime import datetime, timedelta
import os

# RAAS Components
from day10_hybrid import HybridSearch
from day6_reranker import TransformerReranker
from day18_query_optimizer import QueryOptimizer
from day8_detector import HallucinationDetector
from day19_cache_manager import EnhancedCacheManager
from day23_prompt_security import PromptSecurity
from day25_input_validator import InputValidator

# Initialize
app = FastAPI(title="RAAS API", version="8.0.0")
cache = EnhancedCacheManager()
security = PromptSecurity()
validator = InputValidator()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT Configuration
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# In-memory user store (for demo)
users_db = {}

# Create permanent demo user
demo_password = bcrypt.hashpw("demo123".encode(), bcrypt.gensalt()).decode()
users_db["demo"] = {
    "username": "demo",
    "password": demo_password,
    "email": "demo@raas.com",
    "full_name": "Demo User",
    "tier": "free"
}

print("\n" + "=" * 60)
print("DAY 27: RAAS API WITH AUTHENTICATION")
print("=" * 60)
print("✓ Demo User: demo / demo123")
print("=" * 60)

# Load RAAS components
print("\n🚀 Loading RAAS components...")
hybrid_search = HybridSearch()
reranker = TransformerReranker()
generator = QueryOptimizer()
detector = HallucinationDetector()
print("✓ All components loaded!\n")


# ========== MODELS ==========
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
    hallucination_score: float
    processing_time_ms: float


class UserRegister(BaseModel):
    username: str
    password: str
    email: str = ""
    full_name: str = ""


class UserResponse(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    tier: str


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


# ========== AUTH ENDPOINTS ==========
@app.post("/register", response_model=UserResponse)
async def register(user: UserRegister):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already registered")

    hashed = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
    users_db[user.username] = {
        "username": user.username,
        "password": hashed,
        "email": user.email,
        "full_name": user.full_name,
        "tier": "free"
    }

    from day19_cache_manager import UserTier
    cache.set_user_tier(user.username, UserTier.FREE)

    return UserResponse(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        tier="free"
    )


@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_db.get(form_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    if not bcrypt.checkpw(form_data.password.encode(), user["password"].encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    access_token = jwt.encode(
        {"sub": user["username"], "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)},
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username not in users_db:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return users_db[username]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return UserResponse(
        username=current_user["username"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        tier=current_user["tier"]
    )


# ========== MAIN ASK ENDPOINT ==========
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(
        request: Request,
        response: Response,
        req: QuestionRequest,
        current_user: dict = Depends(get_current_user)
):
    start_time = time.time()
    user_id = current_user["username"]

    # Input validation
    validation_result = validator.validate(req.question)
    if not validation_result.is_valid:
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid input", "errors": validation_result.errors}
        )

    safe_question = validation_result.sanitized_input
    validation_warnings = validation_result.warnings

    # Rate limiting
    rate_limit_result = cache.check_rate_limit(user_id)
    for key, value in cache.get_rate_limit_headers(user_id).items():
        response.headers[key] = value

    if not rate_limit_result["allowed"]:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    # Security check
    security_result = security.detect_injection(safe_question)
    if security_result["risk_level"] == "HIGH":
        raise HTTPException(status_code=400, detail="Suspicious input detected")

    final_question = security_result["sanitized_input"]

    # Cache check
    cached_response = cache.get_cached(req.question)
    if cached_response:
        cached_response['cached'] = True
        return cached_response

    # Process query
    chunks = hybrid_search.adaptive_search(final_question, k=req.top_k * 10)
    reranked = reranker.rerank(final_question, chunks, top_k=req.top_k)
    answer_result = generator.generate(final_question, reranked)
    safety = detector.detect(answer_result['answer'], reranked, final_question)

    response_data = {
        "question": req.question,
        "answer": answer_result['answer'],
        "confidence": answer_result['confidence'],
        "sources": [{"document": c['doc'], "page": c['page']} for c in reranked[:3]],
        "is_safe": not safety['has_hallucination'],
        "cached": False,
        "security_flagged": security_result["detected"],
        "validation_warnings": validation_warnings,
        "rate_limit": rate_limit_result,
        "hallucination_score": safety['hallucination_score'],
        "processing_time_ms": round((time.time() - start_time) * 1000, 2)
    }

    cache.set_cached(req.question, response_data)
    return response_data


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "auth": "active",
        "demo_user": "demo/demo123",
        "version": "8.0.0"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)