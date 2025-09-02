# monitoring/business_metrics.py
"""
Business KPI helpers — use central metrics from monitoring.metrics whenever possible.

Bu modül:
- business-level metric increment/observe fonksiyonlarını sağlar
- diğer modüllerde doğrudan metric nesnelerini import etmek yerine bu yüksek seviyeli fonksiyonları kullanabilirsiniz
- duplicate metric tanımlarını önlemek için merkezi metrics modülünü tercih eder
"""

import logging
import time
from typing import Optional, Sequence, Dict

logger = logging.getLogger(__name__)

# Try to import central metric objects from monitoring.metrics
try:
    from monitoring import metrics as central_metrics
    CENTRAL_METRICS_AVAILABLE = True
    logger.info("Using central monitoring.metrics for business KPIs")
except Exception as e:
    central_metrics = None
    CENTRAL_METRICS_AVAILABLE = False
    logger.warning(f"Central monitoring.metrics not available, falling back to local metrics. ({e})")

# If central metrics missing, create safe local fallbacks
if not CENTRAL_METRICS_AVAILABLE:
    from prometheus_client import Counter, Histogram, Gauge

    def _local_counter(name, desc, labels=None):
        return Counter(name, desc, labels) if labels else Counter(name, desc)

    def _local_histogram(name, desc, labels=None, **kwargs):
        return Histogram(name, desc, labels, **kwargs) if labels else Histogram(name, desc, **kwargs)

    def _local_gauge(name, desc, labels=None):
        return Gauge(name, desc, labels) if labels else Gauge(name, desc)

    # Local fallback metric objects (names chosen to match central naming)
    rag_query_success_total = _local_counter(
        "rag_query_success_total",
        "Successful RAG queries (fallback)",
        ["success_type"]
    )

    user_satisfaction_total = _local_counter(
        "user_satisfaction_total",
        "Aggregate count of user satisfaction feedback events (fallback)",
        ["rating"]
    )

    knowledge_coverage_total = _local_counter(
        "knowledge_coverage_total",
        "Number of documents/pages counted for knowledge coverage (fallback)",
        ["airline", "source"]
    )

    retrieval_precision_score = _local_histogram(
        "retrieval_precision_score",
        "Precision score of retrieval (per-query) (fallback)",
        ["k"]
    )

    answer_quality_score = _local_histogram(
        "answer_quality_score",
        "Answer quality score histogram (fallback)",
        ["quality_type"]
    )

    source_relevance_score = _local_histogram(
        "source_relevance_score",
        "Source relevance score histogram (fallback)",
        ["source_type"]
    )

    data_freshness_days = _local_gauge(
        "data_freshness_days",
        "Average data age in days (fallback)",
        ["airline", "source"]
    )

else:
    # Map central objects (if present) into local names used by this module for convenience
    rag_query_success_total = getattr(central_metrics, "rag_query_success_total", None)
    user_satisfaction_total = getattr(central_metrics, "user_satisfaction_total", None)
    knowledge_coverage_total = getattr(central_metrics, "knowledge_coverage_total", None)
    retrieval_precision_score = getattr(central_metrics, "retrieval_precision_score", None)
    answer_quality_score = getattr(central_metrics, "answer_quality_score", None)
    source_relevance_score = getattr(central_metrics, "source_relevance_score", None)
    data_freshness_days = getattr(central_metrics, "data_freshness_days", None)

# -------------------------
# Helper functions to update metrics
# -------------------------

def track_rag_query(success_type: str = "success", provider: Optional[str] = None, duration_seconds: Optional[float] = None):
    """
    Track a RAG query result.
    - success_type: 'success', 'partial', 'failed'
    - provider: optional (openai/claude)
    - duration_seconds: optional to observe histograms in central metrics if available
    """
    try:
        # increment business counter
        if rag_query_success_total is not None:
            try:
                rag_query_success_total.labels(success_type).inc()
            except Exception:
                # if rag_query_success_total has no labels (fallback possibility), try without labels
                try:
                    rag_query_success_total.inc()
                except Exception as e:
                    logger.debug(f"Could not inc rag_query_success_total with or without labels: {e}")

        # if central rag_query_duration_seconds exists, observe
        if CENTRAL_METRICS_AVAILABLE and duration_seconds is not None:
            hist = getattr(central_metrics, "rag_query_duration_seconds", None)
            if hist:
                try:
                    if provider:
                        hist.labels(provider).observe(duration_seconds)
                    else:
                        hist.observe(duration_seconds)
                except Exception as e:
                    logger.debug(f"Failed to observe rag_query_duration_seconds: {e}")
    except Exception as e:
        logger.error(f"track_rag_query error: {e}")

def track_query_resolution(result: str = "resolved"):
    """
    Track resolution outcome (resolved/unresolved/escalated).
    This function will try central rag_query_resolution_total if present, otherwise it uses rag_query_success_total as a fallback.
    """
    try:
        resolution_counter = getattr(central_metrics, "rag_query_resolution_total", None) if CENTRAL_METRICS_AVAILABLE else None
        if resolution_counter:
            try:
                resolution_counter.labels(result).inc()
                return
            except Exception:
                logger.debug("Could not inc rag_query_resolution_total with labels, trying fallback.")
        # fallback to rag_query_success_total if present
        if rag_query_success_total is not None:
            try:
                rag_query_success_total.labels(result).inc()
            except Exception:
                try:
                    rag_query_success_total.inc()
                except Exception as e:
                    logger.debug(f"Could not inc fallback rag_query_success_total: {e}")
    except Exception as e:
        logger.error(f"track_query_resolution error: {e}")

def track_user_feedback(rating: int, feedback_type: Optional[str] = None):
    """
    Track user satisfaction / feedback.
    - rating: integer 1..5
    - feedback_type: optional, e.g. 'helpful', 'not_helpful'
    """
    try:
        # Prefer histogram if available for distribution; else increment total counter
        if user_satisfaction_total is not None:
            try:
                # try label-based inc
                user_satisfaction_total.labels(str(rating)).inc()
            except Exception:
                # no labels fallback
                try:
                    user_satisfaction_total.inc()
                except Exception as e:
                    logger.debug(f"Could not inc user_satisfaction_total: {e}")

        # central histogram
        if CENTRAL_METRICS_AVAILABLE:
            hist = getattr(central_metrics, "user_satisfaction_score", None)
            if hist:
                try:
                    label = feedback_type if feedback_type else "unknown"
                    hist.labels(label).observe(rating)
                except Exception as e:
                    logger.debug(f"Could not observe user_satisfaction_score: {e}")
    except Exception as e:
        logger.error(f"track_user_feedback error: {e}")

def update_knowledge_coverage(airline: str, source: str, delta: int = 1):
    """
    Update the coverage counter for a given airline/source.
    delta: +1 or -1 as needed.
    """
    try:
        if knowledge_coverage_total is not None:
            try:
                knowledge_coverage_total.labels(airline, source).inc(delta)
            except Exception:
                # fallback: try increment without labels
                try:
                    knowledge_coverage_total.inc(delta)
                except Exception as e:
                    logger.debug(f"Could not update knowledge_coverage_total: {e}")
    except Exception as e:
        logger.error(f"update_knowledge_coverage error: {e}")

def record_retrieval_precision(k: int, score: float):
    """
    Record retrieval precision for k (e.g. k=1,3,5).
    """
    try:
        if retrieval_precision_score is not None:
            try:
                retrieval_precision_score.labels(str(k)).observe(score)
            except Exception:
                try:
                    retrieval_precision_score.observe(score)
                except Exception as e:
                    logger.debug(f"Could not observe retrieval_precision_score: {e}")
    except Exception as e:
        logger.error(f"record_retrieval_precision error: {e}")

def record_answer_quality(quality_type: str, score: float):
    try:
        if answer_quality_score is not None:
            try:
                answer_quality_score.labels(quality_type).observe(score)
            except Exception:
                try:
                    answer_quality_score.observe(score)
                except Exception as e:
                    logger.debug(f"Could not observe answer_quality_score: {e}")
    except Exception as e:
        logger.error(f"record_answer_quality error: {e}")

def record_source_relevance(source_type: str, score: float):
    try:
        if source_relevance_score is not None:
            try:
                source_relevance_score.labels(source_type).observe(score)
            except Exception:
                try:
                    source_relevance_score.observe(score)
                except Exception as e:
                    logger.debug(f"Could not observe source_relevance_score: {e}")
    except Exception as e:
        logger.error(f"record_source_relevance error: {e}")

def set_data_freshness(airline: str, source: str, days: float):
    """
    Set the data freshness gauge for an airline/source.
    """
    try:
        if data_freshness_days is not None:
            try:
                data_freshness_days.labels(airline, source).set(days)
            except Exception:
                try:
                    data_freshness_days.set(days)
                except Exception as e:
                    logger.debug(f"Could not set data_freshness_days: {e}")
    except Exception as e:
        logger.error(f"set_data_freshness error: {e}")

# -------------------------
# Convenience: high-level tracker for a full query lifecycle
# -------------------------
def track_full_query_cycle(
    *,
    success_type: str = "success",
    provider: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    resolved: Optional[bool] = None,
    user_rating: Optional[int] = None
):
    """
    High-level wrapper to atomically update common metrics for a single user query lifecycle.
    """
    start = time.time()
    try:
        track_rag_query(success_type=success_type, provider=provider, duration_seconds=duration_seconds)
        if resolved is not None:
            track_query_resolution("resolved" if resolved else "unresolved")
        if user_rating is not None:
            track_user_feedback(user_rating)
    except Exception as e:
        logger.error(f"track_full_query_cycle error: {e}")
    finally:
        elapsed = time.time() - start
        logger.debug(f"track_full_query_cycle completed in {elapsed:.3f}s")

# -------------------------
# Module init log
# -------------------------
logger.info(f"business_metrics module initialized. Central metrics available: {CENTRAL_METRICS_AVAILABLE}")
