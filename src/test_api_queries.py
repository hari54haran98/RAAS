import requests

API_URL = "http://localhost:8000"

queries = [
    "What is the penalty for late payment?",
    "What is the penalty in axis_mortage_loan?",
    "What documents are required for loan?",
    "What is the interest rate?",
    "Is there COVID relief?"
]

print("=" * 60)
print("TESTING API WITH ALL QUERIES")
print("=" * 60)

for i, q in enumerate(queries, 1):
    print(f"\n🔍 Test {i}: {q}")
    print("-" * 40)

    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"question": q, "top_k": 3},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            print(f"✅ Answer: {data['answer'][:150]}...")
            print(f"📊 Confidence: {data['confidence']}")
            print(f"📚 Sources: {[s['document'] for s in data['sources']]}")
            print(f"🛡️ Safe: {data['is_safe']}")
            print(f"⏱️ Time: {data['processing_time_ms']:.0f}ms")
        else:
            print(f"❌ Error: {response.status_code}")

    except Exception as e:
        print(f"❌ Failed: {e}")

print("\n" + "=" * 60)
print("✅ API Testing Complete")
print("=" * 60)