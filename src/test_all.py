import requests
import time

API_URL = "http://localhost:8000"

test_cases = [
    {
        "name": "Penalty with 2.40%",
        "question": "What is the penalty for late payment?",
        "expected": "2.40%"
    },
    {
        "name": "Penalty with 6%",
        "question": "What is the penalty in axis_mortage_loan?",
        "expected": "6%"
    },
    {
        "name": "Documents",
        "question": "What documents are required for loan?",
        "expected": "PAN|Aadhaar"
    },
    {
        "name": "Interest Rate",
        "question": "What is the interest rate?",
        "expected": "13.22%"
    },
    {
        "name": "NOT FOUND",
        "question": "Is there COVID relief?",
        "expected": "NOT FOUND"
    }
]

print("=" * 60)
print("FINAL INTEGRATION TEST")
print("=" * 60)

for test in test_cases:
    print(f"\n🔍 Testing: {test['name']}")
    print(f"Q: {test['question']}")

    start = time.time()
    r = requests.post(f"{API_URL}/ask", json={'question': test['question'], 'top_k': 3})
    elapsed = (time.time() - start) * 1000

    if r.status_code == 200:
        data = r.json()
        print(f"A: {data['answer'][:150]}...")
        print(f"Confidence: {data['confidence']}")
        print(f"Sources: {data.get('sources', [])}")
        print(f"Time: {elapsed:.0f}ms")

        # Check if expected content is present
        if test['expected'] in data['answer'] or (
                test['expected'] == "NOT FOUND" and "NOT FOUND" in data['answer'].upper()):
            print("✅ PASS")
        else:
            print("❌ FAIL")
    else:
        print(f"❌ Error: {r.status_code}")

print("\n" + "=" * 60)