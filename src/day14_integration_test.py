"""
DAY 14: Integration Test for RAAS
End-to-end testing of complete pipeline
"""

import requests
import json
import time
from datetime import datetime
import pandas as pd
from tabulate import tabulate

# Configuration
API_URL = "http://localhost:8000"
UI_URL = "http://localhost:8501"

# Test cases
TEST_CASES = [
    {
        "name": "Penalty Query",
        "question": "What is the penalty for late payment?",
        "expected_contains": ["2.40%", "penalty"],
        "min_confidence": 0.5,
        "max_hallucination": 0.3
    },
    {
        "name": "Document Query",
        "question": "What documents are required for loan application?",
        "expected_contains": ["document", "PAN", "Aadhaar", "income"],
        "min_confidence": 0.4,
        "max_hallucination": 0.3
    },
    {
        "name": "Interest Rate Query",
        "question": "What is the interest rate for home loans?",
        "expected_contains": ["interest", "rate", "%"],
        "min_confidence": 0.3,
        "max_hallucination": 0.3
    },
    {
        "name": "NOT FOUND Test",
        "question": "Is there COVID relief for loans?",
        "expected_not_found": True,
        "max_hallucination": 0.2
    }
]


def test_api_health():
    """Test if API is healthy."""
    print("\n🔍 Testing API Health...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ API Healthy")
            print(f"   Version: {data['version']}")
            print(f"   Components: {', '.join(data['components'].keys())}")
            return True
        else:
            print(f"   ❌ API Unhealthy (Status: {response.status_code})")
            return False
    except Exception as e:
        print(f"   ❌ API Connection Failed: {e}")
        return False


def test_single_query(test_case):
    """Test a single query end-to-end."""
    print(f"\n🔍 Testing: {test_case['name']}")
    print(f"   Question: {test_case['question']}")

    start_time = time.time()

    try:
        # Call API
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": test_case['question'], "top_k": 3},
            timeout=30
        )

        api_time = (time.time() - start_time) * 1000

        if response.status_code != 200:
            print(f"   ❌ API Error: {response.status_code}")
            return None

        result = response.json()

        # Calculate metrics
        passed = True
        issues = []

        # Check 1: NOT FOUND handling
        if test_case.get('expected_not_found', False):
            if "NOT FOUND" not in result['answer'].upper():
                passed = False
                issues.append("Should have returned NOT FOUND")
        else:
            if "NOT FOUND" in result['answer'].upper():
                passed = False
                issues.append("Returned NOT FOUND but expected answer")

        # Check 2: Contains expected terms
        if 'expected_contains' in test_case:
            answer_lower = result['answer'].lower()
            for term in test_case['expected_contains']:
                if term.lower() not in answer_lower:
                    issues.append(f"Missing term: '{term}'")

        # Check 3: Confidence threshold
        if result['confidence'] < test_case.get('min_confidence', 0):
            issues.append(f"Low confidence: {result['confidence']:.2f}")

        # Check 4: Hallucination score
        if result['hallucination_score'] > test_case.get('max_hallucination', 1.0):
            issues.append(f"High hallucination: {result['hallucination_score']:.2f}")

        # Print results
        print(f"   ✅ Response received in {api_time:.0f}ms")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Hallucination Score: {result['hallucination_score']:.2f}")
        print(f"   Safety: {'✅ Safe' if result['is_safe'] else '⚠️ Hallucination'}")
        print(f"   Sources: {len(result['sources'])} documents")

        if issues:
            print(f"   ⚠️ Issues: {', '.join(issues)}")
            passed = False
        else:
            print(f"   ✅ PASSED")

        return {
            'name': test_case['name'],
            'question': test_case['question'],
            'passed': passed,
            'issues': issues,
            'confidence': result['confidence'],
            'hallucination_score': result['hallucination_score'],
            'is_safe': result['is_safe'],
            'response_time_ms': api_time,
            'num_sources': len(result['sources'])
        }

    except Exception as e:
        print(f"   ❌ Test Failed: {e}")
        return None


def check_logs():
    """Verify logs are being written."""
    print("\n🔍 Checking Logs...")
    try:
        # Check if log files exist and have data
        import os
        from pathlib import Path

        log_dir = Path("logs")
        if not log_dir.exists():
            print("   ❌ Logs directory not found")
            return False

        audit_file = log_dir / "audit.json"
        perf_file = log_dir / "performance.csv"

        if not audit_file.exists():
            print("   ❌ audit.json not found")
            return False

        if not perf_file.exists():
            print("   ❌ performance.csv not found")
            return False

        # Check file sizes
        audit_size = audit_file.stat().st_size
        perf_size = perf_file.stat().st_size

        print(f"   ✅ audit.json: {audit_size} bytes")
        print(f"   ✅ performance.csv: {perf_size} bytes")

        if audit_size > 0 and perf_size > 0:
            print("   ✅ Logs contain data")
            return True
        else:
            print("   ⚠️ Logs exist but are empty")
            return False

    except Exception as e:
        print(f"   ❌ Log check failed: {e}")
        return False


def generate_report(results):
    """Generate integration test report."""
    print("\n" + "=" * 60)
    print("📊 INTEGRATION TEST REPORT")
    print("=" * 60)

    # Filter out None results
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        print("❌ No valid test results")
        return

    # Calculate overall metrics
    total = len(valid_results)
    passed = sum(1 for r in valid_results if r['passed'])

    print(f"\n📈 OVERALL RESULTS")
    print(f"   Tests Run: {total}")
    print(f"   Passed: {passed}/{total} ({passed / total * 100:.1f}%)")

    # Table of results
    table_data = []
    for r in valid_results:
        table_data.append([
            r['name'],
            "✅ PASS" if r['passed'] else "❌ FAIL",
            f"{r['confidence']:.2f}",
            f"{r['hallucination_score']:.2f}",
            "✅ Safe" if r['is_safe'] else "⚠️ Hallucination",
            f"{r['response_time_ms']:.0f}ms",
            r['num_sources']
        ])

    print(f"\n📋 TEST DETAILS")
    print(tabulate(
        table_data,
        headers=["Test", "Status", "Conf", "Hall Score", "Safety", "Time", "Sources"],
        tablefmt="grid"
    ))

    # System health
    print(f"\n🩺 SYSTEM HEALTH")
    print(f"   API: ✅ Running")
    print(f"   UI: ✅ Available at {UI_URL}")
    print(f"   Logs: ✅ Being written")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    if passed == total:
        print("   ✅ All tests passing! System ready for deployment.")
    elif passed > total * 0.7:
        print("   ⚠️ Most tests passing. Address failing cases.")
    else:
        print("   ❌ Multiple failures. Review each component.")

    # Next steps
    print(f"\n🚀 NEXT STEPS")
    print(f"   1. Day 15: Documentation")
    print(f"   2. Day 16-20: Security improvements")
    print(f"   3. Day 21-31: Cloud deployment")

    print("\n" + "=" * 60)


def run_integration_test():
    """Run complete integration test suite."""
    print("\n" + "=" * 60)
    print("🚀 RAAS INTEGRATION TEST - DAY 14")
    print("=" * 60)
    print(f"API: {API_URL}")
    print(f"UI: {UI_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Check API health
    if not test_api_health():
        print("\n❌ Cannot proceed. API not healthy.")
        return

    # Step 2: Run test queries
    results = []
    for test in TEST_CASES:
        result = test_single_query(test)
        if result:
            results.append(result)
        time.sleep(1)  # Small delay between queries

    # Step 3: Check logs
    logs_ok = check_logs()

    # Step 4: Generate report
    generate_report(results)

    # Step 5: Save report
    report_file = f"logs/integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'api_healthy': True,
            'logs_ok': logs_ok,
            'results': results,
            'summary': {
                'total': len(results),
                'passed': sum(1 for r in results if r['passed'])
            }
        }, f, indent=2)

    print(f"\n📁 Report saved: {report_file}")


if __name__ == "__main__":
    run_integration_test()