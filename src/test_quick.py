import requests

r = requests.post('http://localhost:8000/ask',
                  json={'question': 'What is the penalty?', 'top_k': 3})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Answer: {data['answer'][:100]}...")
    print(f"Confidence: {data['confidence']}")
else:
    print(f"Error: {r.text}")