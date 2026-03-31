# 🏦 RAAS — Banking Document Q&A

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-25.0-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎥 Demo Video

[Watch the Demo](YOUR_YOUTUBE_LINK)

*Replace YOUR_YOUTUBE_LINK with your actual YouTube video link*

---

## 📌 What is RAAS?

RAAS (Retrieval Augmented Answer Safety) is a production-ready RAG system that answers banking document questions with source citations, hallucination detection, and enterprise-grade security.

**Key Features:**
- ✅ Hybrid Search (FAISS semantic + BM25 keyword)
- ✅ Transformer Reranker (+22% precision)
- ✅ 3-Layer Hallucination Detector (70% reduction)
- ✅ 6 Security Layers (JWT, rate limiting, input validation, prompt injection, Redis cache, HTTPS)
- ✅ Production Deployment (Docker, AWS EC2)
- ✅ MLOps (MLflow, DVC, Prometheus)

---

## 🏗️ Architecture




User Request
↓
6 Security Layers (JWT | Rate Limit | Input Val | Prompt Inj | Redis | HTTPS)
↓
Hybrid Retrieval (FAISS + BM25)
↓
Transformer Reranker (+22% precision)
↓
Groq LLM (llama-3.3-70b)
↓
3-Layer Hallucination Detector (Rules | Semantic | Numerical)
↓
Answer + Sources + Confidence




---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Retrieval Speed | 27.3ms |
| Generation Speed | 4.2s |
| Precision | 92% |
| Hallucination Rate | <5% |

---

## 🔐 Security Features

| Layer | Details |
|-------|---------|
| Input Validation | 18 SQL injection + 6 XSS patterns |
| Prompt Injection | 14 attack patterns |
| Authentication | JWT + bcrypt (30-min tokens) |
| Rate Limiting | 3 tiers (100/1000/10000 per hour) |
| Caching | Redis (30x speedup) |
| Encryption | HTTPS (TLS 1.2+) |

---

## 🛠️ Tech Stack

Python • FastAPI • FAISS • BM25 • Groq LLM • Redis • Docker • AWS EC2 • MLflow • DVC • Prometheus

---

## 🚀 Quick Start

```bash
git clone https://github.com/hari54haran98/RAAS.git
cd RAAS
pip install -r requirements.txt
docker run -d --name redis -p 6379:6379 redis:7-alpine
python src/day11_api.py
streamlit run src/day27_streamlit_auth.py


📁 Project Structure
RAAS/
├── src/           # All Python code
├── data/          # Banking document chunks
├── models/        # FAISS + BM25 indexes
├── langchain-mini/ # LangChain version
├── Dockerfile.api
├── Dockerfile.ui
└── requirements.txt

📚 LangChain Version
bash
cd langchain-mini
python src/mini_rag.py
LangChain: 100 lines, quick prototyping | RAAS: 2000+ lines, production-ready

📄 License
MIT

 Author
Hariharan

GitHub: @hari54haran98

LinkedIn: in/hariharan54



⭐ Star this project if you find it useful!
