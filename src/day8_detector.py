"""
DAY 8: Complete Hallucination Detector for RAAS
3-layer detection with comprehensive reporting
FIXED: Threshold changed from 0.2 to 0.3
"""

import re
from typing import List, Dict, Any
from collections import Counter


class HallucinationDetector:
    """
    Professional 3-layer hallucination detector.

    Layer 1: Rule-based detection
    - Guessing words (I think, probably, maybe)
    - Absolute statements (always, never, definitely)
    - NOT FOUND contradictions

    Layer 2: Source verification
    - Claims supported by source chunks
    - Phrase matching against sources
    - Document citation checking

    Layer 3: Numerical accuracy
    - Numbers present in sources
    - Percentages match exactly
    - Date and amount verification
    """

    def __init__(self):
        print("=" * 60)
        print("DAY 8: COMPLETE HALLUCINATION DETECTOR (FIXED)")
        print("=" * 60)

        # Layer 1: Rule-based patterns
        print("\n[1/3] Initializing rule-based detection...")
        self.guessing_words = [
            'i think', 'probably', 'maybe', 'likely', 'could be',
            'i believe', 'perhaps', 'might be', 'possibly', 'seems',
            'appears', 'suggests', 'indicates', 'generally', 'typically',
            'in my opinion', 'i guess', 'most likely', 'presumably'
        ]

        self.absolute_words = [
            'always', 'never', 'definitely', 'certainly', 'absolutely',
            'every', 'none', 'all', 'nobody', 'everyone', 'must always',
            'undoubtedly', 'without doubt', 'certainly', 'invariably'
        ]

        self.weasel_words = [
            'virtually', 'practically', 'essentially', 'basically',
            'quite', 'rather', 'somewhat', 'fairly', 'pretty'
        ]
        print("✓ Rule-based patterns loaded")

        # Layer 2: Source verification
        print("\n[2/3] Initializing source verification...")
        self.min_phrase_length = 15  # Minimum length to check
        self.max_phrases_to_check = 5  # Number of phrases to verify
        print("✓ Source verification ready")

        # Layer 3: Numerical accuracy
        print("\n[3/3] Initializing numerical verification...")
        self.number_patterns = {
            'percentage': r'\d+\.?\d*%',
            'integer': r'\b\d+\b(?!\.?\d)',
            'decimal': r'\b\d+\.\d+\b(?!%)',
            'currency': r'(?:Rs\.?|INR|₹)\s*\d+[\d,]*\.?\d*',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        }
        print("✓ Numerical verification ready")

        # Weights for each layer
        self.layer_weights = {
            'rule_based': 0.3,
            'source_verification': 0.4,
            'numerical': 0.3
        }

        print("\n" + "=" * 60)

    def detect(self, answer: str, chunks: List[Dict], question: str) -> Dict[str, Any]:
        """
        Complete 3-layer hallucination detection.

        Args:
            answer: Generated answer
            chunks: Source chunks used
            question: Original question

        Returns:
            Comprehensive detection report
        """
        report = {
            'has_hallucination': False,
            'hallucination_score': 0.0,
            'confidence': 0.0,
            'issues': [],
            'layers': {},
            'summary': {}
        }

        print(f"\n🔍 Analyzing answer against {len(chunks)} sources...")

        # Layer 1: Rule-based
        layer1 = self._layer1_rule_based(answer)
        report['layers']['rule_based'] = layer1
        report['issues'].extend(layer1['issues'])

        # Layer 2: Source verification
        layer2 = self._layer2_source_verification(answer, chunks)
        report['layers']['source_verification'] = layer2
        report['issues'].extend(layer2['issues'])

        # Layer 3: Numerical accuracy
        layer3 = self._layer3_numerical(answer, chunks)
        report['layers']['numerical'] = layer3
        report['issues'].extend(layer3['issues'])

        # Calculate weighted score
        report['hallucination_score'] = (
                layer1['score'] * self.layer_weights['rule_based'] +
                layer2['score'] * self.layer_weights['source_verification'] +
                layer3['score'] * self.layer_weights['numerical']
        )

        # ===== FIXED: Threshold changed from 0.2 to 0.3 =====
        report['has_hallucination'] = report['hallucination_score'] > 0.3
        report['confidence'] = 1.0 - min(report['hallucination_score'], 0.9)

        # Generate summary
        report['summary'] = self._generate_summary(report)

        # Print report
        self._print_report(report)

        return report

    def _layer1_rule_based(self, answer: str) -> Dict[str, Any]:
        """Layer 1: Rule-based detection."""
        issues = []
        score = 0.0
        answer_lower = answer.lower()

        # Check guessing words
        for word in self.guessing_words:
            if word in answer_lower:
                issues.append(f"Guessing language: '{word}'")
                score += 0.15

        # Check absolute words
        for word in self.absolute_words:
            if word in answer_lower:
                issues.append(f"Absolute statement: '{word}'")
                score += 0.2

        # Check weasel words
        for word in self.weasel_words:
            if word in answer_lower:
                issues.append(f"Weasel word: '{word}'")
                score += 0.1

        # Check for contradictions
        if "NOT FOUND" in answer.upper() and len(answer.split()) > 10:
            issues.append("NOT FOUND with additional text")
            score += 0.25

        return {
            'score': min(score, 1.0),
            'issues': issues,
            'details': {
                'guessing_words_found': [w for w in self.guessing_words if w in answer_lower],
                'absolute_words_found': [w for w in self.absolute_words if w in answer_lower]
            }
        }

    def _layer2_source_verification(self, answer: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Layer 2: Verify answer against source chunks."""
        issues = []
        score = 0.0
        answer_lower = answer.lower()

        if not chunks:
            return {'score': 0.5, 'issues': ['No source chunks'], 'details': {}}

        # Combine chunk text
        chunk_text = ' '.join([c['text'].lower() for c in chunks])

        # Extract key phrases from answer
        sentences = re.split(r'[.!?]', answer)
        phrases_to_check = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) > self.min_phrase_length:
                # Take first 50 chars of each long sentence
                phrases_to_check.append(sent[:50])

        phrases_to_check = phrases_to_check[:self.max_phrases_to_check]

        # Check each phrase
        unsupported_phrases = []
        for phrase in phrases_to_check:
            if phrase and phrase not in chunk_text:
                unsupported_phrases.append(phrase)
                score += 0.2

        if unsupported_phrases:
            issues.append(f"{len(unsupported_phrases)} phrases not in sources")

        # Check document citation
        cited_docs = []
        for chunk in chunks:
            if chunk['doc'].lower() in answer_lower:
                cited_docs.append(chunk['doc'])

        if not cited_docs and len(answer.split()) > 20:
            issues.append("No source documents cited")
            score += 0.3

        return {
            'score': min(score, 1.0),
            'issues': issues,
            'details': {
                'phrases_checked': len(phrases_to_check),
                'unsupported_phrases': unsupported_phrases[:3],
                'cited_documents': cited_docs
            }
        }

    def _layer3_numerical(self, answer: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Layer 3: Verify numerical accuracy."""
        issues = []
        score = 0.0

        # Extract all numbers from answer
        answer_numbers = []
        for pattern_name, pattern in self.number_patterns.items():
            matches = re.findall(pattern, answer)
            answer_numbers.extend(matches)

        if not answer_numbers:
            return {'score': 0.0, 'issues': [], 'details': {}}

        # Extract numbers from chunks
        chunk_text = ' '.join([c['text'] for c in chunks])
        chunk_numbers = []
        for pattern_name, pattern in self.number_patterns.items():
            matches = re.findall(pattern, chunk_text)
            chunk_numbers.extend(matches)

        # Check each number
        mismatched_numbers = []
        for num in answer_numbers:
            if num not in chunk_numbers:
                mismatched_numbers.append(num)
                score += 0.25

                # Extra penalty for percentages
                if '%' in num:
                    score += 0.1

        if mismatched_numbers:
            issues.append(f"{len(mismatched_numbers)} numbers not in sources")

        # Special check for percentages
        answer_percentages = re.findall(self.number_patterns['percentage'], answer)
        chunk_percentages = re.findall(self.number_patterns['percentage'], chunk_text)

        for pct in answer_percentages:
            if pct not in chunk_percentages:
                issues.append(f"Percentage '{pct}' not in sources")
                score += 0.3

        return {
            'score': min(score, 1.0),
            'issues': issues,
            'details': {
                'numbers_found': answer_numbers[:5],
                'mismatched': mismatched_numbers[:3],
                'percentages_checked': len(answer_percentages)
            }
        }

    def _generate_summary(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable summary."""
        return {
            'verdict': 'HALLUCINATION' if report['has_hallucination'] else 'SAFE',
            'risk_level': 'HIGH' if report['hallucination_score'] > 0.5 else
            'MEDIUM' if report['hallucination_score'] > 0.3 else 'LOW',
            'total_issues': len(report['issues']),
            'layer_scores': {
                layer: data['score'] for layer, data in report['layers'].items()
            }
        }

    def _print_report(self, report: Dict[str, Any]):
        """Print formatted detection report."""
        print("\n" + "=" * 50)
        print("HALLUCINATION DETECTION REPORT")
        print("=" * 50)

        # Overall status
        status = "⚠️ HALLUCINATION DETECTED" if report['has_hallucination'] else "✅ SAFE"
        print(f"\nStatus: {status}")
        print(f"Score: {report['hallucination_score']:.2f}/1.0")
        print(f"Confidence: {report['confidence']:.1%}")
        print(f"Risk Level: {report['summary']['risk_level']}")

        # Layer breakdown
        print(f"\nLayer Scores:")
        for layer, data in report['layers'].items():
            print(f"  • {layer}: {data['score']:.2f}")

        # Issues
        if report['issues']:
            print(f"\nIssues Found ({len(report['issues'])}):")
            for i, issue in enumerate(report['issues'][:5], 1):
                print(f"  {i}. {issue}")
            if len(report['issues']) > 5:
                print(f"  ... and {len(report['issues']) - 5} more")
        else:
            print(f"\n✓ No issues detected")

        print("\n" + "=" * 50)


# Quick test
if __name__ == "__main__":
    print("\n🧪 TESTING COMPLETE HALLUCINATION DETECTOR")
    print("=" * 60)

    detector = HallucinationDetector()

    # Test cases
    test_chunks = [{
        'doc': 'sbi_home_loan_terms',
        'page': 2,
        'text': 'Penalty: 2.40% per annum for irregular payments beyond 60 days.'
    }]

    test_cases = [
        {
            'name': 'Valid Answer',
            'answer': 'The penalty is 2.40% per annum according to sbi_home_loan_terms.',
            'question': 'What is the penalty?'
        },
        {
            'name': 'Hallucinated Answer',
            'answer': 'I think the penalty is probably 5% for all loans.',
            'question': 'What is the penalty?'
        }
    ]

    for test in test_cases:
        print(f"\n🔍 Test: {test['name']}")
        detector.detect(test['answer'], test_chunks, test['question'])

    print("\n" + "=" * 60)
    print("✓ COMPLETE HALLUCINATION DETECTOR READY")