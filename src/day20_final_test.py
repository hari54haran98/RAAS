"""
DAY 20: FINAL INTEGRATION TEST
Tests all queries with optimized components
"""

import requests
import time
from tabulate import tabulate

API_URL = "http://localhost:8000"

TEST_CASES = [
    {
        "name": "Penalty Query",
        "question": "What is the penalty for late payment?",
        "expected": ["2.40%", "penalty"],
        "min_confidence": 0.7
    },
    {
        "name": "Document Query",
        "question": "What documents are required for loan application?",
        "expected": ["PAN", "Aadhaar", "income"],
        "min_confidence": 0.6
    },
    {
        "name": "Interest Query",
        "question": "What is the interest rate for home loans?",
        "expected": ["13.22%", "interest"],
        "min_confidence": 0.7
    },
    {
        "name": "NOT FOUND Test",
        "question": "Is there COVID relief for loans?",
        "expected_not_found": True
    }
]


def test_api():
    print("=" * 60)
    print("🚀 FINAL INTEGRATION TEST - DAY 20")
    print("=" * 60)

    # Check health
    try:
        health = requests.get(f"{API_URL}/health")
        if health.status_code != 200:
            print("❌ API not healthy")
            return
        print("✅ API Healthy\n")
    except:
        print("❌ Cannot connect to API")
        return

    results = []

    for test in TEST_CASES:
        print(f"\n🔍 Testing: {test['name']}")
        print(f"   Q: {test['question']}")

        start = time.time()
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": test['question'], "top_k": 3}
        )
        elapsed = (time.time() - start) * 1000

        if response.status_code != 200:
            print(f"   ❌ API Error: {response.status_code}")
            continue

        data = response.json()

        # Check results
        passed = True
        issues = []

        if test.get('expected_not_found'):
            if 'NOT FOUND' not in data['answer'].upper():
                passed = False
                issues.append("Should say NOT FOUND")
        else:
            if 'NOT FOUND' in data['answer'].upper():
                passed = False
                issues.append("Returned NOT FOUND incorrectly")

            for term in test['expected']:
                if term.lower() not in data['answer'].lower():
                    issues.append(f"Missing '{term}'")

            if data['confidence'] < test['min_confidence']:
                issues.append(f"Low confidence: {data['confidence']:.2f}")

        print(f"   ✅ Time: {elapsed:.0f}ms")
        print(f"   Confidence: {data['confidence']:.2f}")
        print(f"   Safety: {'✅ Safe' if data['is_safe'] else '⚠️ Hallucination'}")

        if issues:
            print(f"   ⚠️ Issues: {', '.join(issues)}")
            passed = False
        else:
            print(f"   ✅ PASSED")

        results.append([
            test['name'],
            "✅ PASS" if passed else "❌ FAIL",
            f"{data['confidence']:.2f}",
            f"{elapsed:.0f}ms"
        ])

    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(tabulate(results, headers=["Test", "Status", "Conf", "Time"], tablefmt="grid"))

    passed = sum(1 for r in results if "✅" in r[1])
    total = len(results)
    print(f"\n✅ Passed: {passed}/{total} ({passed / total * 100:.1f}%)")


if __name__ == "__main__":
    test_api()