# monitoring/enhanced_metrics.py
from prometheus_client import Counter, Histogram, Gauge
import logging
import time
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# RAG DOĞRULUĞU METRIKLERİ
rag_accuracy_score = Histogram(
    "rag_accuracy_score",
    "RAG response accuracy score (0.0-1.0) based on source relevance and answer quality",
    ["provider", "airline", "language"],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# COST PER QUERY METRIKLERİ
cost_per_query_usd = Histogram(
    "cost_per_query_usd", 
    "Cost per individual RAG query in USD",
    ["provider", "model"],
    buckets=[0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
)

token_usage_per_query = Histogram(
    "token_usage_per_query",
    "Total tokens used per query (input + output)",
    ["provider", "model", "token_type"], # token_type: input/output/total
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
)

query_cost_efficiency = Histogram(
    "query_cost_efficiency",
    "Cost efficiency: accuracy_score / cost_usd",
    ["provider", "model"],
    buckets=[0, 10, 50, 100, 200, 500, 1000, 2000, 5000]
)

# GEÇİKME METRIKLERİ - COMPONENT BREAKDOWN
rag_component_latency = Histogram(
    "rag_component_latency_seconds",
    "Latency breakdown by RAG pipeline component",
    ["component", "provider"], # component: retrieval/generation/embedding/total
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
)

def track_rag_accuracy(
    provider: str,
    airline: str, 
    language: str,
    accuracy_score: float
):
    """RAG doğruluğunu track eder"""
    try:
        rag_accuracy_score.labels(
            provider=provider,
            airline=airline, 
            language=language
        ).observe(accuracy_score)
        
        logger.debug(f"RAG accuracy tracked: {accuracy_score:.3f} for {provider}")
        
    except Exception as e:
        logger.error(f"track_rag_accuracy error: {e}")

def track_query_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int, 
    total_cost_usd: float,
    accuracy_score: Optional[float] = None
):
    """Query başına maliyet ve token kullanımını track eder"""
    try:
        # Cost per query
        cost_per_query_usd.labels(
            provider=provider,
            model=model
        ).observe(total_cost_usd)
        
        # Token usage breakdown
        total_tokens = input_tokens + output_tokens
        
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
        
        token_usage_per_query.labels(
            provider=provider,
            model=model,
            token_type="total"  
        ).observe(total_tokens)
        
        # Cost efficiency (accuracy/cost ratio)
        if accuracy_score and total_cost_usd > 0:
            efficiency = accuracy_score / total_cost_usd
            query_cost_efficiency.labels(
                provider=provider,
                model=model
            ).observe(efficiency)
        
        logger.debug(f"Query cost tracked: ${total_cost_usd:.4f} for {provider}/{model}")
        
    except Exception as e:
        logger.error(f"track_query_cost error: {e}")

def track_component_latency(
    component: str,  # "retrieval", "generation", "total" 
    provider: str,
    duration_seconds: float
):
    """RAG pipeline bileşenlerinin gecikme sürelerini track eder"""
    try:
        rag_component_latency.labels(
            component=component,
            provider=provider
        ).observe(duration_seconds)
        
        logger.debug(f"{component} latency: {duration_seconds:.3f}s for {provider}")
        
    except Exception as e:
        logger.error(f"track_component_latency error: {e}")

def calculate_source_precision(sources: List[Dict], relevance_threshold: float = 0.7) -> float:
    """Retrieved source'ların precision'ını hesaplar"""
    if not sources:
        return 0.0
    
    relevant_count = sum(1 for source in sources 
                        if source.get('similarity_score', 0) >= relevance_threshold)
    return relevant_count / len(sources)

def calculate_answer_completeness(
    sources_count: int = 0,
    expected_sources: int = 3
) -> float:
    """Cevap completeness'ini hesaplar"""
    # Basit hesaplama: source coverage
    source_coverage = min(sources_count / expected_sources, 1.0) if expected_sources > 0 else 0.0
    return source_coverage

def calculate_overall_accuracy(
    source_precision: float,
    answer_completeness: float
) -> float:
    """Genel accuracy score hesaplar"""
    # Weighted average
    accuracy = (source_precision * 0.6) + (answer_completeness * 0.4)
    return min(accuracy, 1.0)

logger.info("Enhanced RAG metrics module loaded")