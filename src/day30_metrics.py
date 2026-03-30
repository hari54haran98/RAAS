"""
DAY 30: Prometheus Metrics for RAAS
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response
import time

# Create metrics
REQUEST_COUNT = Counter(
    'raas_requests_total',
    'Total request count',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'raas_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

ACTIVE_REQUESTS = Gauge(
    'raas_active_requests',
    'Currently active requests'
)

HALLUCINATION_SCORE = Histogram(
    'raas_hallucination_score',
    'Hallucination detection scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

CONFIDENCE_SCORE = Histogram(
    'raas_confidence_score',
    'Answer confidence scores',
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)


class MetricsMiddleware:
    """Middleware to track request metrics"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        method = scope["method"]
        path = scope["path"]

        ACTIVE_REQUESTS.inc()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()
                latency = time.time() - start_time
                REQUEST_LATENCY.labels(method=method, endpoint=path).observe(latency)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            ACTIVE_REQUESTS.dec()


def setup_metrics(app: FastAPI):
    """Add metrics endpoint to FastAPI app"""

    # Add middleware
    app.add_middleware(MetricsMiddleware)

    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint"""
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )

    print("✅ Prometheus metrics enabled at /metrics")