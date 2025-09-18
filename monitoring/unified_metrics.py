# monitoring/unified_metrics.py - UNIFIED METRICS SYSTEM
"""
Unified metrics system combining operational and business intelligence metrics.
Single source of truth for all application metrics.
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import logging
import time
from typing import Optional, Dict, List
from functools import wraps

logger = logging.getLogger(__name__)

# =============================================================================
# 1. OPERATIONAL METRICS - Temel sistem metrikleri (İlk sistemden geri getirilen)
# =============================================================================

# HTTP Request Metrics - Instrumentator ile birlikte çalışır
http_requests_total = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status_code']
)

# RAG System Core Metrics - İlk sistemdeki temel metrikler
rag_query_duration_seconds = Histogram(
    'rag_query_duration_seconds',
    'RAG query end-to-end duration in seconds',
    ['provider', 'status'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
)

vector_search_duration_seconds = Histogram(
    'vector_search_duration_seconds',
    'Vector search duration in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
)

ai_api_cost_total_usd = Counter(
    'ai_api_cost_total_usd',
    'Total AI API cost in USD (cumulative)',
    ['provider', 'model']
)

user_satisfaction_total = Counter(
    'user_satisfaction_total',
    'User satisfaction feedback count',
    ['rating']  # thumbs_up, thumbs_down
)

# =============================================================================
# 2. BUSINESS INTELLIGENCE METRICS - Enhanced metrikler 
# =============================================================================

# Accuracy & Quality Metrics
rag_accuracy_score = Histogram(
    "rag_accuracy_score",
    "RAG response accuracy score (0.0-1.0) based on source relevance and answer quality",
    ["provider", "airline", "language"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Cost Analysis Metrics
cost_per_query_usd = Histogram(
    "cost_per_query_usd", 
    "Cost per individual RAG query in USD",
    ["provider", "model"],
    buckets=[0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
)

token_usage_per_query = Histogram(
    "token_usage_per_query",
    "Token usage per query breakdown",
    ["provider", "model", "token_type"], # token_type: input/output
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
)

query_cost_efficiency = Histogram(
    "query_cost_efficiency",
    "Cost efficiency: accuracy_score / cost_usd (higher is better)",
    ["provider", "model"],
    buckets=[0, 10, 50, 100, 200, 500, 1000, 2000, 5000]
)

# Performance Breakdown Metrics
rag_component_latency_seconds = Histogram(
    "rag_component_latency_seconds",
    "RAG pipeline component latency breakdown",
    ["component", "provider"], # component: retrieval/generation/total
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# =============================================================================
# 3. SYSTEM HEALTH METRICS - Sistem durumu
# =============================================================================

active_connections_gauge = Gauge(
    'active_database_connections',
    'Number of active database connections'
)

embedding_cache_hits_total = Counter(
    'embedding_cache_hits_total',
    'Total embedding cache hits',
    ['hit_type']  # hit, miss
)

# =============================================================================
# 4. UNIFIED TRACKING FUNCTIONS - Tek yerden metrik yönetimi
# =============================================================================

def track_http_request(method: str, endpoint: str, status_code: str):
    """HTTP request tracking - Instrumentator ile koordineli çalışır"""
    try:
        http_requests_total.labels(
            method=method,
            endpoint=endpoint, 
            status_code=status_code
        ).inc()
    except Exception as e:
        logger.error(f"HTTP request tracking error: {e}")

def track_rag_query(provider: str, duration: float, status: str = "success"):
    """RAG query tracking - Hem temel hem enhanced sistemde kullanılır"""
    try:
        rag_query_duration_seconds.labels(
            provider=provider,
            status=status
        ).observe(duration)
        
        logger.debug(f"RAG query tracked: {duration:.3f}s for {provider}")
    except Exception as e:
        logger.error(f"RAG query tracking error: {e}")

def track_vector_search(duration: float):
    """Vector search performance tracking"""
    try:
        vector_search_duration_seconds.observe(duration)
        logger.debug(f"Vector search tracked: {duration:.3f}s")
    except Exception as e:
        logger.error(f"Vector search tracking error: {e}")

def track_api_cost(provider: str, model: str, cost: float):
    """API cost tracking - Cumulative total"""
    try:
        ai_api_cost_total_usd.labels(
            provider=provider,
            model=model
        ).inc(cost)
        logger.debug(f"API cost tracked: ${cost:.4f} for {provider}/{model}")
    except Exception as e:
        logger.error(f"API cost tracking error: {e}")

def track_user_feedback(rating: str):
    """User satisfaction tracking"""
    try:
        user_satisfaction_total.labels(rating=rating).inc()
        logger.debug(f"User feedback tracked: {rating}")
    except Exception as e:
        logger.error(f"User feedback tracking error: {e}")

# =============================================================================
# 5. ENHANCED TRACKING FUNCTIONS - İleri seviye analitics
# =============================================================================

def track_rag_accuracy(
    provider: str,
    airline: str, 
    language: str,
    accuracy_score: float
):
    """RAG accuracy tracking for business intelligence"""
    try:
        rag_accuracy_score.labels(
            provider=provider,
            airline=airline, 
            language=language
        ).observe(accuracy_score)
        
        logger.debug(f"RAG accuracy tracked: {accuracy_score:.3f} for {provider}")
    except Exception as e:
        logger.error(f"RAG accuracy tracking error: {e}")

def track_query_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int, 
    total_cost_usd: float,
    accuracy_score: Optional[float] = None
):
    """Comprehensive query cost and efficiency tracking"""
    try:
        # Per-query cost
        cost_per_query_usd.labels(
            provider=provider,
            model=model
        ).observe(total_cost_usd)
        
        # Token usage breakdown
        token_usage_per_query.labels(
            provider=provider,
            model=model,
            token_type="input"
        ).observe(input_tokens)
        
        token_usage_per_query.labels(
            provider=provider,
            model=model, 
            token_type="output"
        ).observe(output_tokens)
        
        # Cost efficiency calculation
        if accuracy_score and total_cost_usd > 0:
            efficiency = accuracy_score / total_cost_usd
            query_cost_efficiency.labels(
                provider=provider,
                model=model
            ).observe(efficiency)
        
        # Also track in cumulative cost (for backward compatibility)
        track_api_cost(provider, model, total_cost_usd)
        
        logger.debug(f"Query cost tracked: ${total_cost_usd:.4f} for {provider}/{model}")
    except Exception as e:
        logger.error(f"Query cost tracking error: {e}")

def track_component_latency(
    component: str,  # "retrieval", "generation", "total" 
    provider: str,
    duration_seconds: float
):
    """RAG component latency tracking for performance analysis"""
    try:
        rag_component_latency_seconds.labels(
            component=component,
            provider=provider
        ).observe(duration_seconds)
        
        logger.debug(f"{component} latency tracked: {duration_seconds:.3f}s for {provider}")
    except Exception as e:
        logger.error(f"Component latency tracking error: {e}")

# =============================================================================
# 6. BUSINESS INTELLIGENCE HELPERS - Accuracy calculation utilities
# =============================================================================

def calculate_source_precision(sources: List[Dict], relevance_threshold: float = 0.7) -> float:
    """Calculate precision of retrieved sources based on similarity scores"""
    if not sources:
        return 0.0
    
    relevant_count = sum(1 for source in sources 
                        if source.get('similarity_score', 0) >= relevance_threshold)
    return relevant_count / len(sources)

def calculate_answer_completeness(
    sources_count: int = 0,
    expected_sources: int = 3
) -> float:
    """Calculate answer completeness based on source coverage"""
    if expected_sources <= 0:
        return 0.0
    source_coverage = min(sources_count / expected_sources, 1.0)
    return source_coverage

def calculate_overall_accuracy(
    source_precision: float,
    answer_completeness: float,
    precision_weight: float = 0.6,
    completeness_weight: float = 0.4
) -> float:
    """Calculate overall RAG accuracy score"""
    accuracy = (source_precision * precision_weight) + (answer_completeness * completeness_weight)
    return min(accuracy, 1.0)

# =============================================================================
# 7. SYSTEM UTILITIES - Monitoring ve health check
# =============================================================================

def track_system_health(db_pool=None):
    """Track system health metrics"""
    try:
        if db_pool:
            active_connections_gauge.set(db_pool.get_size())
    except Exception as e:
        logger.error(f"System health tracking error: {e}")

def track_embedding_cache(hit_type: str):
    """Track embedding cache performance"""
    try:
        embedding_cache_hits_total.labels(hit_type=hit_type).inc()
    except Exception as e:
        logger.error(f"Embedding cache tracking error: {e}")

# =============================================================================
# 8. METRICS EXPORT - Prometheus endpoint
# =============================================================================

def get_metrics_text() -> str:
    """Generate Prometheus metrics text format"""
    try:
        return generate_latest().decode('utf-8')
    except Exception as e:
        logger.error(f"Metrics generation error: {e}")
        return ""

def get_metrics_summary() -> Dict:
    """Get metrics summary for debugging"""
    return {
        "operational_metrics": [
            "http_requests_total",
            "rag_query_duration_seconds", 
            "vector_search_duration_seconds",
            "ai_api_cost_total_usd",
            "user_satisfaction_total"
        ],
        "business_metrics": [
            "rag_accuracy_score",
            "cost_per_query_usd",
            "token_usage_per_query", 
            "query_cost_efficiency",
            "rag_component_latency_seconds"
        ],
        "system_metrics": [
            "active_database_connections",
            "embedding_cache_hits_total"
        ],
        "status": "unified_system_active"
    }

# =============================================================================
# 9. DECORATOR UTILITIES - Easy integration
# =============================================================================

def track_execution_time(metric_name: str, labels: Dict = None):
    """Decorator to track function execution time"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Dynamic metric tracking based on function
                if "vector_search" in func.__name__:
                    track_vector_search(duration)
                elif "rag" in func.__name__:
                    provider = labels.get("provider", "unknown") if labels else "unknown"
                    track_rag_query(provider, duration, "success")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                if "rag" in func.__name__:
                    provider = labels.get("provider", "unknown") if labels else "unknown"
                    track_rag_query(provider, duration, "error")
                raise
        return wrapper
    return decorator

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

logger.info("Unified metrics system initialized")
logger.info("Loaded metric categories: operational, business_intelligence, system_health")

# Export commonly used functions
__all__ = [
    # Core tracking functions
    'track_http_request',
    'track_rag_query', 
    'track_vector_search',
    'track_api_cost',
    'track_user_feedback',
    
    # Enhanced tracking functions
    'track_rag_accuracy',
    'track_query_cost', 
    'track_component_latency',
    
    # Calculation helpers
    'calculate_source_precision',
    'calculate_answer_completeness',
    'calculate_overall_accuracy',
    
    # System utilities
    'track_system_health',
    'get_metrics_text',
    'get_metrics_summary',
    
    # Decorators
    'track_execution_time'
]