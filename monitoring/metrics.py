# monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# -------------------------
# Registry-safe helpers
# -------------------------
def _iter_registry_collectors():
    """Yield collectors currently registered (safe wrapper)."""
    try:
        for c in list(REGISTRY._collector_to_names.keys()):
            yield c
    except Exception as e:
        # Fallback: registry internal changed/unexpected
        logger.debug(f"Could not iterate registry collectors: {e}")
        return

def get_collector_by_name(name: str):
    """
    Return an existing collector object whose _name == name, or None.
    NOTE: accesses internal field _name, which is pragmatic but widely used.
    """
    for collector in _iter_registry_collectors():
        try:
            if hasattr(collector, "_name") and collector._name == name:
                return collector
        except Exception:
            continue
    return None

def safe_create_counter(name: str, description: str, labelnames: Optional[Sequence[str]] = None):
    existing = get_collector_by_name(name)
    if existing:
        logger.debug(f"Reusing existing Counter: {name}")
        return existing
    if labelnames:
        return Counter(name, description, labelnames)
    return Counter(name, description)

def safe_create_histogram(name: str, description: str, labelnames: Optional[Sequence[str]] = None, **kwargs):
    existing = get_collector_by_name(name)
    if existing:
        logger.debug(f"Reusing existing Histogram: {name}")
        return existing
    if labelnames:
        return Histogram(name, description, labelnames, **kwargs)
    return Histogram(name, description, **kwargs)

def safe_create_gauge(name: str, description: str, labelnames: Optional[Sequence[str]] = None):
    existing = get_collector_by_name(name)
    if existing:
        logger.debug(f"Reusing existing Gauge: {name}")
        return existing
    if labelnames:
        return Gauge(name, description, labelnames)
    return Gauge(name, description)

# -------------------------
# Central metric definitions
# -------------------------
# HTTP / FastAPI
http_requests_total = safe_create_counter(
    "fastapi_requests_total",
    "Total number of HTTP requests handled by the FastAPI app",
    ["method", "endpoint", "status_code"]
)

http_request_duration_seconds = safe_create_histogram(
    "fastapi_request_duration_seconds",
    "Histogram of HTTP request duration in seconds",
    ["method", "endpoint"]
)

fastapi_active_requests = safe_create_gauge(
    "fastapi_active_requests",
    "Number of active in-flight requests being processed"
)

# RAG / Retrieval & AI
rag_queries_total = safe_create_counter(
    "rag_queries_total",
    "Total number of RAG (retrieve-and-generate) queries",
    ["provider", "status"]  # provider=openai/claude, status=success/error/partial
)

rag_query_success_total = safe_create_counter(
    "rag_query_success_total",
    "Count of successful RAG queries (business-centric breakdown)",
    ["success_type"]  # 'success', 'partial', 'failed'
)

rag_query_resolution_total = safe_create_counter(
    "rag_query_resolution_total",
    "How many queries resulted in a resolved / unresolved / escalated state",
    ["result"]  # 'resolved','unresolved','escalated'
)

rag_query_duration_seconds = safe_create_histogram(
    "rag_query_duration_seconds",
    "RAG query end-to-end duration in seconds",
    ["provider"]
)

rag_response_time_seconds = safe_create_histogram(
    "rag_response_time_seconds",
    "AI response time (model generation) in seconds",
    ["provider"]
)

# Vector/search & embeddings
vector_searches_total = safe_create_counter(
    "vector_searches_total",
    "Total number of vector search calls executed"
)

vector_search_duration_seconds = safe_create_histogram(
    "vector_search_duration_seconds",
    "Vector search duration in seconds"
)

embedding_generation_duration_seconds = safe_create_histogram(
    "embedding_generation_duration_seconds",
    "Embedding generation duration in seconds"
)

# Business / KPI metrics
user_satisfaction_total = safe_create_counter(
    "user_satisfaction_total",
    "Aggregate count of user satisfaction feedback events",
    ["rating"]  # 1..5 or 'helpful'/'not_helpful' as labels
)

user_satisfaction_score = safe_create_histogram(
    "user_satisfaction_score",
    "Histogram of user satisfaction ratings (1-5)",
    ["feedback_type"]
)

knowledge_coverage_total = safe_create_counter(
    "knowledge_coverage_total",
    "Number of documents/pages counted for knowledge coverage",
    ["airline", "source"]
)

retrieval_precision_score = safe_create_histogram(
    "retrieval_precision_score",
    "Precision score of retrieval (per-query)",
    ["k"]
)

answer_quality_score = safe_create_histogram(
    "answer_quality_score",
    "Human-judged or heuristic answer quality per response",
    ["quality_type"]
)

source_relevance_score = safe_create_histogram(
    "source_relevance_score",
    "Relevance score for sources returned by retrieval",
    ["source_type"]
)

# Freshness / coverage
data_freshness_days = safe_create_gauge(
    "data_freshness_days",
    "Average data age in days",
    ["airline", "source"]
)

# Cost / token usage
ai_api_token_input_total = safe_create_counter(
    "ai_api_token_input_total",
    "Total input tokens sent to AI provider",
    ["provider"]
)

ai_api_token_output_total = safe_create_counter(
    "ai_api_token_output_total",
    "Total output tokens returned by AI provider",
    ["provider"]
)

ai_api_cost_total_usd = safe_create_counter(
    "ai_api_cost_total_usd",
    "Accumulated estimated cost in USD for AI calls (sum)",
    ["provider"]
)

# System / infra
memory_usage_bytes = safe_create_gauge("memory_usage_bytes", "Process memory usage in bytes")
cpu_usage_percent = safe_create_gauge("cpu_usage_percent", "Process CPU usage percent (0-100)")

# -------------------------
# Utilities
# -------------------------
def collect_metrics_text() -> bytes:
    """
    Return the prometheus exposition text for /metrics.
    Use this in your FastAPI /metrics endpoint (return Response(..., media_type='text/plain; version=0.0.4')).
    """
    try:
        payload = generate_latest(REGISTRY)
        return payload
    except Exception as e:
        logger.error(f"Failed to generate metrics payload: {e}")
        # generate_latest can fail if registry is in bad state; return empty payload
        return b""

def metrics_summary() -> dict:
    """
    Return a small summary of registered metric names and a health indicator.
    Useful for sanity-check endpoints in admin UI.
    """
    registered_metrics = []
    try:
        for collector in _iter_registry_collectors():
            try:
                if hasattr(collector, "_name"):
                    registered_metrics.append(collector._name)
            except Exception:
                continue
        return {
            "total_registered": len(registered_metrics),
            "metrics": sorted(list(set(registered_metrics))),
            "registry_healthy": True
        }
    except Exception as e:
        logger.error(f"Metrics summary failed: {e}")
        return {
            "total_registered": 0,
            "metrics": [],
            "registry_healthy": False,
            "error": str(e)
        }

# -------------------------
# Module init logging
# -------------------------
logger.info("monitoring.metrics module imported")
try:
    logger.info(f"Registered metrics count: {len(list(_iter_registry_collectors()))}")
except Exception:
    logger.debug("Could not fetch collector count due to registry internals.")
