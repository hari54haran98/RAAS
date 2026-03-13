import requests
import time
from day19_cache_manager import CacheManager

cache = CacheManager()
q = 'What is the penalty for late payment?'

# Test cache directly
start = time.time()
cached = cache.get_cached(q)
if cached:
    print(f"✅ Cache HIT - Time: {(time.time()-start)*1000:.0f}ms")
else:
    print("❌ Cache MISS")

# Test API
start = time.time()
r = requests.post('http://localhost:8000/ask', json={'question': q, 'top_k': 3})
print(f"✅ API response - Time: {(time.time()-start)*1000:.0f}ms")