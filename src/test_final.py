import requests

queries = [
    'What is the penalty for late payment?',
    'What penalty rate is in sbi_home_loan_terms?'
]

for q in queries:
    r = requests.post('http://localhost:8000/ask', json={'question': q, 'top_k': 3})
    data = r.json()
    print(f'\nQ: {q}')
    print(f'A: {data["answer"]}')
    print(f'Confidence: {data["confidence"]}')
    print(f'Sources: {data.get("sources", [])}')