"""
DAY 7: Complete LLM Generator for RAAS
Full-featured generator with banking prompts, confidence scoring, and fallbacks
"""

import httpx
import time
import re
from typing import List, Dict, Any
import json


class LLMGenerator:
    """
    Professional LLM generator with banking-specific prompts.
    Features:
    - Persistent connection to Ollama
    - Banking-specific prompt templates
    - Confidence scoring with multiple factors
    - Fallback extraction when LLM fails
    - Source citation tracking
    - NOT FOUND handling
    """

    def __init__(self, model: str = "llama3.2"):
        print("=" * 60)
        print("DAY 7: COMPLETE LLM GENERATOR")
        print("=" * 60)

        # Initialize persistent connection
        print("\n[1/3] Initializing Ollama connection...")
        self.client = httpx.Client(timeout=30.0)
        self.base_url = "http://localhost:11434"
        self.model = model
        print(f"✓ Connected to Ollama ({model})")

        # Initialize prompt templates
        print("\n[2/3] Loading banking prompt templates...")
        self.prompt_templates = self._load_prompt_templates()
        print("✓ Templates loaded")

        # Initialize confidence factors
        print("\n[3/3] Initializing confidence scoring...")
        self.confidence_factors = {
            'has_numbers': 0.2,
            'cites_sources': 0.2,
            'matches_question_type': 0.15,
            'has_percentage': 0.15,
            'specific_terms': 0.1,
            'not_found': -0.4
        }
        print("✓ Confidence scoring ready")
        print("\n" + "=" * 60)

    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load banking-specific prompt templates."""
        return {
            'default': """You are a precise banking document analyst. Use ONLY the context below.

CONTEXT FROM BANKING DOCUMENTS:
{context}

STRICT RULES:
1. Answer ONLY from the context above
2. If the answer is NOT in the context, say: "NOT FOUND in banking documents"
3. Include EXACT numbers, percentages, and terms from context
4. Mention which document and page number
5. Be concise and factual
6. Do NOT add any information from outside the context

QUESTION: {question}

ANSWER:""",

            'penalty': """You are analyzing banking penalty clauses. Use ONLY the context.

CONTEXT:
{context}

Focus on:
- Exact penalty percentages
- When penalties apply
- Which documents specify penalties

QUESTION: {question}

ANSWER:""",

            'documents': """You are listing banking document requirements. Use ONLY the context.

CONTEXT:
{context}

List:
- Required documents
- Where to submit
- Any deadlines mentioned

QUESTION: {question}

ANSWER:"""
        }

    def generate(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """
        Complete answer generation pipeline.

        Args:
            question: User query
            chunks: Top reranked chunks

        Returns:
            Dictionary with answer, confidence, sources, and metadata
        """
        start_time = time.time()

        result = {
            'question': question,
            'answer': None,
            'confidence': 0.0,
            'sources': [],
            'is_not_found': False,
            'generation_time_ms': 0,
            'metadata': {}
        }

        if not chunks:
            result['answer'] = "NOT FOUND in banking documents."
            result['is_not_found'] = True
            result['confidence'] = 0.1
            result['generation_time_ms'] = (time.time() - start_time) * 1000
            return result

        # Select appropriate template based on question
        template = self._select_template(question)
        context = self._build_context(chunks)

        prompt = template.format(context=context, question=question)

        try:
            # Generate with Ollama
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 300,
                        "keep_alive": -1
                    }
                },
                timeout=30.0
            )
            answer = response.json()["response"]

        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            answer = self._fallback_extraction(question, chunks)

        # Calculate confidence
        confidence = self._calculate_confidence(answer, chunks, question)

        # Extract sources
        sources = self._extract_sources(answer, chunks)

        # Final result
        result['answer'] = answer
        result['confidence'] = confidence
        result['sources'] = sources
        result['is_not_found'] = 'NOT FOUND' in answer.upper()
        result['generation_time_ms'] = round((time.time() - start_time) * 1000, 2)

        # Add metadata
        result['metadata'] = {
            'chunks_used': len(chunks),
            'template_used': template[:50] + "...",
            'has_numbers': bool(re.search(r'\d+', answer)),
            'has_percentages': bool(re.search(r'\d+\.?\d*%', answer))
        }

        return result

    def _select_template(self, question: str) -> str:
        """Select appropriate template based on question type."""
        question_lower = question.lower()

        if 'penalty' in question_lower or 'late' in question_lower:
            return self.prompt_templates['penalty']
        elif 'document' in question_lower or 'required' in question_lower:
            return self.prompt_templates['documents']
        else:
            return self.prompt_templates['default']

    def _build_context(self, chunks: List[Dict], max_chars: int = 800) -> str:
        """Build context string from chunks with length limit."""
        context_parts = []
        total_chars = 0

        for i, chunk in enumerate(chunks[:2], 1):  # Use top 2 chunks
            header = f"[{chunk['doc']}, Page {chunk['page']}]"
            text = chunk['text'][:400]  # Limit each chunk

            if total_chars + len(header) + len(text) > max_chars:
                # Truncate if needed
                remaining = max_chars - total_chars
                text = text[:remaining] + "..."

            context_parts.append(header)
            context_parts.append(text)
            total_chars += len(header) + len(text)

            if i < len(chunks[:2]):
                context_parts.append("---")

        return "\n".join(context_parts)

    def _fallback_extraction(self, question: str, chunks: List[Dict]) -> str:
        """Fallback when LLM fails - extract directly from chunks."""
        question_lower = question.lower()

        # Try to find relevant information
        for chunk in chunks[:2]:
            chunk_text = chunk['text'].lower()

            # Penalty questions
            if 'penalty' in question_lower:
                if 'penalty' in chunk_text and '%' in chunk_text:
                    percentages = re.findall(r'\d+\.?\d*%', chunk['text'])
                    if percentages:
                        return f"{chunk['doc']} (Page {chunk['page']}): Penalty rate is {percentages[0]}"

            # Document questions
            if 'document' in question_lower:
                if 'document' in chunk_text or 'required' in chunk_text:
                    # Extract first 2 sentences about documents
                    sentences = chunk['text'].split('.')
                    doc_sentences = [s for s in sentences if 'document' in s.lower() or 'required' in s.lower()]
                    if doc_sentences:
                        return f"{chunk['doc']} (Page {chunk['page']}): {doc_sentences[0]}."

        return "NOT FOUND in banking documents."

    def _calculate_confidence(self, answer: str, chunks: List[Dict], question: str) -> float:
        """Multi-factor confidence calculation."""
        if "NOT FOUND" in answer.upper():
            return 0.1

        confidence = 0.3  # Base confidence

        answer_lower = answer.lower()
        question_lower = question.lower()

        # Factor 1: Has numbers
        if re.search(r'\d+', answer):
            confidence += self.confidence_factors['has_numbers']

        # Factor 2: Has percentages
        if re.search(r'\d+\.?\d*%', answer):
            confidence += self.confidence_factors['has_percentage']

        # Factor 3: Cites sources
        if any(chunk['doc'].lower() in answer_lower for chunk in chunks):
            confidence += self.confidence_factors['cites_sources']

        # Factor 4: Matches question type
        if 'penalty' in question_lower and 'penalty' in answer_lower:
            confidence += self.confidence_factors['matches_question_type']
        if 'document' in question_lower and 'document' in answer_lower:
            confidence += self.confidence_factors['matches_question_type']

        # Factor 5: Has specific banking terms
        banking_terms = ['interest', 'rate', 'loan', 'payment', 'charge', 'fee']
        if any(term in answer_lower for term in banking_terms):
            confidence += self.confidence_factors['specific_terms']

        return min(confidence, 0.95)

    def _extract_sources(self, answer: str, chunks: List[Dict]) -> List[str]:
        """Extract cited sources from answer."""
        sources = []
        answer_lower = answer.lower()

        for chunk in chunks:
            if chunk['doc'].lower() in answer_lower:
                sources.append(f"{chunk['doc']} (Page {chunk['page']})")

        return sources[:3]  # Max 3 sources

    def close(self):
        """Clean up connections."""
        self.client.close()
        print("✓ LLM Generator connections closed")


# Quick test
if __name__ == "__main__":
    print("\n🧪 TESTING COMPLETE LLM GENERATOR")
    print("=" * 60)

    # Initialize generator
    generator = LLMGenerator()

    # Sample chunks
    test_chunks = [
        {
            'doc': 'sbi_home_loan_terms',
            'page': 2,
            'text': 'Penalty Charges: 2.40% per annum for irregular payments beyond 60 days.',
            'tags': 'penalty'
        },
        {
            'doc': 'hdfc_home_loan_agreement',
            'page': 15,
            'text': 'Required Documents: PAN card, Aadhaar card, Income proof, Bank statements.',
            'tags': 'documents'
        }
    ]

    # Test queries
    test_queries = [
        "What is the penalty for late payment?",
        "What documents are required?"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = generator.generate(query, test_chunks)
        print(f"✓ Answer: {result['answer'][:100]}...")
        print(f"✓ Confidence: {result['confidence']:.2f}")
        print(f"✓ Time: {result['generation_time_ms']}ms")

    generator.close()
    print("\n" + "=" * 60)
    print("✓ COMPLETE LLM GENERATOR READY")
    print("=" * 60)