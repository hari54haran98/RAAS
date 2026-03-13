"""
DAY 16: Groq API Integration for RAAS
Professional LLM generator using Groq's ultra-fast inference
Latency: 1-2 seconds (was 15 seconds with Ollama)
"""

import os
import time
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
import groq

# Load environment variables from .env file
load_dotenv()


class GroqGenerator:
    """
    Ultra-fast LLM generator using Groq API.
    Replaces Ollama-based generator with 10x faster inference.
    """

    def __init__(self):
        print("=" * 60)
        print("DAY 16: GROQ API INTEGRATION")
        print("=" * 60)

        # Retrieve API key from environment
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env file. "
                "Please add your key to proceed."
            )

        # Initialize Groq client
        self.client = groq.Groq(api_key=self.api_key)

        # Using latest Llama 3.3 70B model (as of March 2026)
        self.model = "llama-3.3-70b-versatile"

        print(f"✓ Connected to Groq")
        print(f"✓ Model: {self.model}")
        print(f"✓ Target latency: 1-2 seconds (was 15 seconds with Ollama)")
        print("=" * 60)

    def generate(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate answer using Groq API.

        Args:
            question: User's question
            chunks: Top reranked chunks from hybrid search

        Returns:
            Dictionary containing:
                - answer: Generated text
                - confidence: Confidence score (0.1-0.9)
                - sources: List of cited documents
                - is_not_found: Boolean flag for missing information
                - generation_time_ms: Response time in milliseconds
        """
        start_time = time.time()

        # Handle empty chunks
        if not chunks:
            return {
                'answer': 'NOT FOUND in banking documents.',
                'confidence': 0.1,
                'sources': [],
                'is_not_found': True,
                'generation_time_ms': 0
            }

        # Build context from top chunks
        context = self._build_context(chunks[:2])

        # Strict banking prompt
        prompt = self._build_prompt(question, context)

        try:
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise banking document analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )

            answer = response.choices[0].message.content

        except Exception as e:
            print(f"⚠️ Groq API error: {e}")
            answer = self._fallback_answer(question, chunks)

        # Calculate metrics
        generation_time = (time.time() - start_time) * 1000
        confidence = self._calculate_confidence(answer, chunks)
        sources = self._extract_sources(answer, chunks)

        print(f"✓ Generated in {generation_time:.0f}ms (was 15,000ms with Ollama)")

        return {
            'answer': answer,
            'confidence': confidence,
            'sources': sources,
            'is_not_found': 'NOT FOUND' in answer.upper(),
            'generation_time_ms': generation_time
        }

    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"[Document {i}: {chunk['doc']}, Page {chunk['page']}]")
            context_parts.append(chunk['text'][:400])
            if i < len(chunks):
                context_parts.append("---")
        return "\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build banking-specific prompt."""
        return f"""You are a precise banking document analyst. Use ONLY the context below.

CONTEXT FROM BANKING DOCUMENTS:
{context}

STRICT RULES:
1. Answer ONLY from the context above
2. If the answer is NOT in the context, say: "NOT FOUND in banking documents"
3. Include EXACT numbers, percentages, and terms from context
4. Mention which document and page number
5. Be concise and factual

QUESTION: {question}

ANSWER:"""

    def _fallback_answer(self, question: str, chunks: List[Dict]) -> str:
        """Fallback when Groq API fails."""
        question_lower = question.lower()

        # Penalty questions
        if 'penalty' in question_lower:
            for chunk in chunks[:2]:
                if 'penalty' in chunk.get('tags', '') and '%' in chunk['text']:
                    percentages = re.findall(r'\d+\.?\d*%', chunk['text'])
                    if percentages:
                        return f"{chunk['doc']} page {chunk['page']}: Penalty rate is {percentages[0]}"

        # Document questions
        if 'document' in question_lower:
            for chunk in chunks[:2]:
                if 'documents' in chunk.get('tags', ''):
                    return f"{chunk['doc']} page {chunk['page']} mentions document requirements"

        return "NOT FOUND in banking documents"

    def _calculate_confidence(self, answer: str, chunks: List[Dict]) -> float:
        """Calculate confidence score (0.1-0.9)."""
        if "NOT FOUND" in answer.upper():
            return 0.1

        confidence = 0.5  # Base confidence

        # Contains numbers/percentages
        if re.search(r'\d+\.?\d*%?', answer):
            confidence += 0.2

        # Cites sources
        if any(chunk['doc'].lower() in answer.lower() for chunk in chunks):
            confidence += 0.2

        return min(confidence, 0.9)

    def _extract_sources(self, answer: str, chunks: List[Dict]) -> List[str]:
        """Extract cited sources from answer."""
        sources = []
        answer_lower = answer.lower()

        for chunk in chunks[:2]:
            if chunk['doc'].lower() in answer_lower:
                sources.append(f"{chunk['doc']} p{chunk['page']}")

        return sources


# Quick test
if __name__ == "__main__":
    print("\n🧪 TESTING GROQ GENERATOR")
    print("-" * 40)

    # Initialize generator
    generator = GroqGenerator()

    # Sample test chunks
    test_chunks = [
        {
            'doc': 'sbi_home_loan_terms',
            'page': 2,
            'text': 'Penalty: 2.40% per annum for irregular payments beyond 60 days.',
            'tags': 'penalty'
        }
    ]

    # Run test
    result = generator.generate("What is the penalty?", test_chunks)

    print(f"\n📝 ANSWER:")
    print(f"{result['answer']}")
    print(f"\n⏱️  Time: {result['generation_time_ms']:.0f}ms")
    print(f"📊 Confidence: {result['confidence']:.2f}")
    if result['sources']:
        print(f"📚 Sources: {', '.join(result['sources'])}")