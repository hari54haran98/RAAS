"""
DAY 17: Optimized Prompt Generator for RAAS
Fixes document query hallucination (score 0.62 → <0.2)
"""

import os
import time
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import groq

load_dotenv()


class OptimizedGenerator:
    """LLM generator with anti-hallucination prompts."""

    def __init__(self):
        print("=" * 50)
        print("DAY 17: OPTIMIZED PROMPT GENERATOR")
        print("=" * 50)

        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found")

        self.client = groq.Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"
        print(f"✓ Groq connected")

    def generate(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Generate answer with strict anti-hallucination rules."""
        start = time.time()

        if not chunks:
            return self._not_found_response()

        # Ultra-strict prompt for document queries
        if 'document' in question.lower() or 'required' in question.lower():
            prompt = self._document_prompt(question, chunks[:2])
        else:
            prompt = self._default_prompt(question, chunks[:2])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=250
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ Error: {e}")
            answer = self._fallback(question, chunks)

        return {
            'answer': answer,
            'confidence': self._confidence(answer, chunks),
            'sources': [f"{c['doc']} p{c['page']}" for c in chunks[:2]
                        if c['doc'].lower() in answer.lower()],
            'is_not_found': 'NOT FOUND' in answer.upper(),
            'time_ms': round((time.time() - start) * 1000, 2)
        }

    def _document_prompt(self, question: str, chunks: List[Dict]) -> str:
        """Ultra-strict prompt for document queries."""
        ctx = "\n".join([f"[{c['doc']} p{c['page']}]: {c['text'][:300]}" for c in chunks])
        return f"""You are a banking document analyst. The ONLY valid answer is a bullet list of documents found EXACTLY in the context.

CONTEXT:
{ctx}

QUESTION: {question}

RULES:
- List ONLY documents mentioned in context
- If no documents listed, say "NOT FOUND"
- NEVER add words like "typically" or "usually"
- Format as bullet points only

ANSWER:"""

    def _default_prompt(self, question: str, chunks: List[Dict]) -> str:
        """Standard strict prompt."""
        ctx = "\n".join([f"[{c['doc']} p{c['page']}]: {c['text'][:300]}" for c in chunks])
        return f"""CONTEXT:
{ctx}

QUESTION: {question}

RULES:
1. Answer ONLY from context
2. Use exact numbers
3. Cite document and page
4. If missing, say "NOT FOUND"

ANSWER:"""

    def _fallback(self, question: str, chunks: List[Dict]) -> str:
        """Fallback when API fails."""
        for c in chunks[:2]:
            if 'document' in question.lower() and 'document' in c['text'].lower():
                return f"{c['doc']} mentions documents (page {c['page']})"
            if 'penalty' in question.lower() and '%' in c['text']:
                match = re.search(r'\d+\.?\d*%', c['text'])
                if match:
                    return f"{c['doc']}: {match.group()}"
        return "NOT FOUND"

    def _confidence(self, answer: str, chunks: List[Dict]) -> float:
        """Simple confidence score."""
        if 'NOT FOUND' in answer.upper():
            return 0.1
        conf = 0.5
        if any(c['doc'].lower() in answer.lower() for c in chunks):
            conf += 0.3
        if re.search(r'\d+%', answer):
            conf += 0.2
        return min(conf, 0.9)

    def _not_found_response(self) -> Dict:
        """Standard not found response."""
        return {
            'answer': 'NOT FOUND in banking documents',
            'confidence': 0.1,
            'sources': [],
            'is_not_found': True,
            'time_ms': 0
        }


# Quick test
if __name__ == "__main__":
    gen = OptimizedGenerator()

    # Test document query
    chunks = [{
        'doc': 'hdfc_home_loan_agreement',
        'page': 15,
        'text': 'Required Documents: PAN card, Aadhaar card, Income proof, Bank statements.',
        'tags': 'documents'
    }]

    res = gen.generate("What documents are required for loan?", chunks)
    print(f"\nAnswer: {res['answer']}")
    print(f"Confidence: {res['confidence']}")
    print(f"Time: {res['time_ms']}ms")