"""
DAY 19: Cache Manager for RAAS
Redis-based caching and rate limiting
"""

import os
import time
import hashlib
import json
from typing import Dict, Any, Optional
import redis
from dotenv import load_dotenv

load_dotenv()


class CacheManager:
    """
    Manages Redis cache and rate limiting for RAAS.

    Features:
    - Cache frequent queries for instant responses (0ms)
    - Rate limiting: 100 requests/hour per IP
    - Automatic cache expiration (1 hour)
    """

    def __init__(self, host='localhost', port=6379, db=0):
        print("=" * 50)
        print("DAY 19: CACHE MANAGER")
        print("=" * 50)

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=2
            )
            # Test connection
            self.client.ping()
            self.available = True
            print(f"✓ Redis connected at {host}:{port}")
        except Exception as e:
            print(f"⚠️ Redis not available: {e}")
            print("✓ Running in fallback mode (no caching)")
            self.available = False

    def _hash_question(self, question: str) -> str:
        """Create unique hash for a question."""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()

    def get_cached(self, question: str) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        if not self.available:
            return None

        key = f"cache:{self._hash_question(question)}"
        cached = self.client.get(key)

        if cached:
            print(f"✓ Cache HIT for: '{question[:30]}...'")
            return json.loads(cached)

        print(f"⏳ Cache MISS for: '{question[:30]}...'")
        return None

    def set_cached(self, question: str, response: Dict[str, Any], ttl: int = 3600):
        """Cache response with TTL (default 1 hour)."""
        if not self.available:
            return

        key = f"cache:{self._hash_question(question)}"
        self.client.setex(
            key,
            ttl,
            json.dumps(response, default=str)
        )
        print(f"✓ Cached for 1 hour: '{question[:30]}...'")

    def check_rate_limit(self, ip: str, max_requests: int = 100, window: int = 3600) -> bool:
        """
        Check if IP has exceeded rate limit.
        Returns True if allowed, False if rate limited.
        """
        if not self.available:
            return True  # Allow if Redis not available

        key = f"rate:{ip}"

        # Increment request count
        current = self.client.incr(key)

        # Set expiry on first request
        if current == 1:
            self.client.expire(key, window)

        if current > max_requests:
            print(f"⚠️ Rate limit exceeded for {ip}: {current}/{max_requests}")
            return False

        remaining = max_requests - current
        if remaining % 10 == 0:  # Log every 10 requests
            print(f"📊 Rate limit: {remaining}/{max_requests} remaining for {ip}")

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.available:
            return {"status": "redis_not_available"}

        info = self.client.info()
        return {
            "status": "connected",
            "used_memory": info.get("used_memory_human", "unknown"),
            "total_connections": info.get("total_connections_received", 0),
            "uptime_days": info.get("uptime_in_days", 0)
        }

    def clear_cache(self, pattern: str = "cache:*"):
        """Clear cache entries (for testing)."""
        if not self.available:
            return

        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)
            print(f"✓ Cleared {len(keys)} cache entries")


# Quick test
if __name__ == "__main__":
    cache = CacheManager()

    if cache.available:
        # Test rate limiting
        test_ip = "192.168.1.1"
        for i in range(3):
            allowed = cache.check_rate_limit(test_ip, max_requests=2)
            print(f"Request {i + 1}: {'✅ Allowed' if allowed else '❌ Blocked'}")
            time.sleep(1)

        # Test caching
        test_q = "What is the penalty?"
        test_response = {
            'answer': '2.40% penalty',
            'confidence': 0.9,
            'time_ms': 850
        }

        cache.set_cached(test_q, test_response)
        cached = cache.get_cached(test_q)
        print(f"\nCached response: {cached}")

        print(f"\n📊 Cache stats: {cache.get_stats()}")
    else:
        print("\n⚠️ Install Redis locally or use Docker:")
        print("   Option 1: https://redis.io/download")
        print("   Option 2: docker run -p 6379:6379 redis")