"""
Test Ollama speed directly
"""

import httpx
import time
import sys


def test_ollama_speed():
    print("🚀 TESTING OLLAMA SPEED")
    print("=" * 50)

    # Test 1: First request (cold start)
    print("\n📊 Test 1: First request (cold start)")
    client = httpx.Client(timeout=120.0)

    start = time.time()
    response = client.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": "Say hello in one word",
            "stream": False,
            "options": {"keep_alive": -1}
        }
    )
    elapsed1 = (time.time() - start) * 1000
    print(f"   Time: {elapsed1:.1f}ms")
    print(f"   Response: {response.json()['response']}")

    # Test 2: Second request (should be fast)
    print("\n📊 Test 2: Second request (same connection)")
    start = time.time()
    response = client.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": "Say hello in one word again",
            "stream": False,
            "options": {"keep_alive": -1}
        }
    )
    elapsed2 = (time.time() - start) * 1000
    print(f"   Time: {elapsed2:.1f}ms")

    # Test 3: Third request
    print("\n📊 Test 3: Third request")
    start = time.time()
    response = client.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": "Say hello in one word again",
            "stream": False,
            "options": {"keep_alive": -1}
        }
    )
    elapsed3 = (time.time() - start) * 1000
    print(f"   Time: {elapsed3:.1f}ms")

    client.close()

    print("\n" + "=" * 50)
    print(f"RESULTS:")
    print(f"   Request 1 (cold): {elapsed1:.1f}ms")
    print(f"   Request 2: {elapsed2:.1f}ms")
    print(f"   Request 3: {elapsed3:.1f}ms")

    if elapsed2 < elapsed1 * 0.3:  # If 2nd request is 70% faster
        print("\n✅ SUCCESS: Keep-alive is working!")
        print(f"   Speed improvement: {elapsed1 / elapsed2:.1f}x faster")
    else:
        print("\n❌ FAIL: Keep-alive NOT working")
        print("   All requests taking similar time")

    return elapsed1, elapsed2, elapsed3


if __name__ == "__main__":
    test_ollama_speed()