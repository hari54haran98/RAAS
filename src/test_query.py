import requests

# Test with specific query about 2.40%
queries = [
    "What is the penalty rate in sbi_home_loan_terms?",
    "What is 2.40% penalty for?",
    "What is the penalty for late payment?"
]

for q in queries:
    print(f"\n🔍 Testing: {q}")
    r = requests.post('http://localhost:8000/ask',
                      json={'question': q, 'top_k': 3})
    if r.status_code == 200:
        data = r.json()
        print(f"Answer: {data['answer'][:150]}...")
        print(f"Confidence: {data['confidence']}")
        print(f"Sources: {data.get('sources', [])}")
    else:
        print(f"Error: {r.status_code}")