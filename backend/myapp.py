# myapp.py - UNIFIED METRICS EDITION
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import Response
from typing import List, Optional, Dict, Any, Tuple
import asyncpg
import os
from contextlib import asynccontextmanager
import logging
from pydantic import BaseModel, Field
from functools import lru_cache
from pydantic_settings import BaseSettings
from embedding_service import get_embedding_service
from vector_operations import EnhancedVectorOperations
from openai_service import get_openai_service
from claude_service import get_claude_service
import math
from typing import Dict, Optional
import uuid
from datetime import datetime, timedelta
import hashlib
from secrets_loader import SecretsLoader

loader = SecretsLoader()

# UNIFIED METRICS SYSTEM - Inline definitions (like original working code)
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps

# =============================================================================
# 1. OPERATIONAL METRICS - Temel sistem metrikleri 
# =============================================================================

# HTTP Request Metrics - Instrumentator ile birlikte çalışır
http_requests_total = Counter(
    'http_requests_total', 
    'Total HTTP requests', 
    ['method', 'endpoint', 'status_code']
)

# RAG System Core Metrics
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
# 4. UNIFIED TRACKING FUNCTIONS - Inline function definitions
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

# FastAPI Prometheus Instrumentator - Automatic HTTP metrics
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time
from fastapi.middleware.cors import CORSMiddleware

from fastapi import File, UploadFile
from fastapi.responses import Response
from services.aws_speech import get_aws_speech_service
import asyncio
from services.assemblyai_stt import get_assemblyai_service

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSettings(BaseSettings):
    # Database connection parameters
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(default="global_gate")
    user: str = Field(default="postgres")
    password: str = Field(
        default_factory=lambda: loader.get_secret('postgres_password', 'DB_PASSWORD') or 'postgres'
    )
    
    # Connection pool settings
    min_pool_size: int = Field(default=5, ge=1, le=20)
    max_pool_size: int = Field(default=20, ge=5, le=100)
    command_timeout: int = Field(default=60, ge=10, le=300)
    
    # SSL and advanced settings
    ssl: bool = Field(default=False, description="Enable SSL connection")
    echo: bool = Field(default=False, description="Echo SQL queries (debug)")
    
    openai_api_key: str = Field(
        default_factory=lambda: loader.get_secret('openai_api_key', 'OPENAI_API_KEY') or 'dummy'
    )
    class Config:
        env_prefix = "DB_"
        env_file = ".env"
        case_sensitive = False
    
    def get_asyncpg_params(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
        }
    
    @property
    def is_configured(self) -> bool:
        """Check if database is properly configured"""
        return all([self.database, self.user, self.password])
    
    def validate_connection_params(self) -> tuple[bool, list]:
        """Validate database connection parameters"""
        errors = []
        if not self.database:
            errors.append("Database name required")
        if not self.user:
            errors.append("Database user required") 
        if not self.password:
            errors.append("Database password required")
        
        return len(errors) == 0, errors

@lru_cache()
def get_settings() -> DatabaseSettings:
    return DatabaseSettings()

settings = get_settings()

# Global variables - AI services preloaded
db_pool = None
vector_ops = None
embedding_service = None
openai_service = None
claude_service = None
aws_speech_service = None 
assemblyai_service = None

# UNIFIED METRICS MIDDLEWARE 
class UnifiedMetricsMiddleware(BaseHTTPMiddleware):
    """Unified middleware that tracks both operational and business metrics"""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        method = request.method
        path = request.url.path
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            status_code = str(response.status_code)
            
            # Track HTTP request metrics (unified system)
            track_http_request(method, path, status_code)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            track_http_request(method, path, "500")
            logger.error(f"Request error: {method} {path} - {str(e)}")
            raise

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown procedures"""
    global db_pool, vector_ops, embedding_service, openai_service, claude_service, aws_speech_service, assemblyai_service
    
    # Startup
    logger.info("Airlines Policy API starting with Unified Metrics...")
    try:
        # 1. Database connection
        db_pool = await asyncpg.create_pool(
            **settings.get_asyncpg_params(),
            min_size=settings.min_pool_size,
            max_size=settings.max_pool_size,
            command_timeout=settings.command_timeout
        )
        logger.info("PostgreSQL connection pool created")
        
        # Test connection
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM policy")
            logger.info(f"Database contains {result} policies")
        
        # 2. Load AI services
        logger.info("Loading AI services...")
        
        # Load embedding service
        try:
            embedding_service = get_embedding_service()
            logger.info(f"Embedding service loaded (Model: {embedding_service._model_name})")
        except Exception as e:
            logger.error(f"Embedding service load failed: {e}")
            embedding_service = None
        
        # Load OpenAI service
        try:
            openai_service = get_openai_service()
            test_result = openai_service.test_connection()
            if test_result["success"]:
                logger.info(f"OpenAI service loaded (Model: {openai_service.default_model})")
            else:
                logger.warning(f"OpenAI service warning: {test_result.get('message', 'Unknown')}")
        except Exception as e:
            logger.error(f"OpenAI service load failed: {e}")
            openai_service = None
        
        # Load Claude service
        try:
            claude_service = get_claude_service()
            test_result = claude_service.test_connection()
            if test_result["success"]:
                logger.info(f"Claude service loaded (Model: {claude_service.default_model})")
            else:
                logger.warning(f"Claude service warning: {test_result.get('message', 'Unknown')}")
        except Exception as e:
            logger.error(f"Claude service load failed: {e}")
            claude_service = None
        
        # 3. Vector operations initialization
        try:
            if embedding_service:
                vector_ops = EnhancedVectorOperations(db_pool)
                logger.info("Vector operations initialized")
                
                # Auto-embed existing policies
                embedded_count = await vector_ops.embed_existing_policies()
                logger.info(f"Embedded {embedded_count} policies on startup")
            else:
                logger.warning("Vector operations skipped (no embedding service)")
        except Exception as ve:
            logger.error(f"Vector operations init failed: {ve}")
            vector_ops = None
        
        # 4. Voice service initialization
        try:
            aws_speech_service = get_aws_speech_service()
            logger.info("AWS Speech Service loaded successfully")
        except Exception as e:
            logger.error(f"AWS Speech Service load failed: {e}")
            aws_speech_service = None
        
        # 5. AssemblyAI service initialization
        try:
            assemblyai_service = get_assemblyai_service()
            logger.info("AssemblyAI service loaded successfully")
        except Exception as e:
            logger.error(f"AssemblyAI service load failed: {e}")
            assemblyai_service = None
        
        # 6. Track system health metrics
        track_system_health(db_pool)
        
        # 7. Startup summary
        services_status = {
            "database": "ready" if db_pool else "failed",
            "embedding": "ready" if embedding_service else "failed", 
            "openai": "ready" if openai_service else "failed",
            "claude": "ready" if claude_service else "failed",
            "vector_ops": "ready" if vector_ops else "failed",
            "unified_metrics": "ready",
            "aws_tts": "ready" if aws_speech_service else "failed",
            "assemblyai_stt": "ready" if assemblyai_service else "failed"
        }
        
        logger.info("Startup Summary:")
        for service, status in services_status.items():
            logger.info(f"   {service}: {status}")
        
        logger.info("Airlines Policy API ready with Unified Metrics!")
            
    except Exception as e:
        logger.error(f"Critical startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Airlines Policy API...")
    
    if db_pool:
        await db_pool.close()
        logger.info("Database connection closed")
    
    # Clean up AI services
    embedding_service = None
    openai_service = None
    claude_service = None
    vector_ops = None
    aws_speech_service = None
    assemblyai_service = None
    
    logger.info("Shutdown completed")

# FastAPI instance
app = FastAPI(
    title="Airlines Policy API - Unified Metrics",
    description="RAG-powered PostgreSQL API with Unified Metrics System",
    version="7.0.0-unified",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://localhost:5173", 
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add Unified Metrics Middleware
app.add_middleware(UnifiedMetricsMiddleware)

# Add Prometheus Instrumentator for automatic HTTP metrics
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    should_respect_env_var=True,
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app)

# SIMPLIFIED PYDANTIC MODELS 
class BaggagePolicy(BaseModel):
    id: int
    source: str
    content: str
    created_at: Optional[str] = None
    quality_score: Optional[float] = None

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[List[BaggagePolicy]] = None
    count: int = 0

class ChatRequest(BaseModel):
    question: str = Field(..., description="User question", min_length=3, max_length=2000)
    airline_preference: Optional[str] = Field(None, description="Preferred airline (e.g., 'turkish_airlines', 'pegasus')")
    max_results: int = Field(default=5, description="Max retrieved documents", le=10, ge=1)
    similarity_threshold: float = Field(default=0.3, description="Similarity threshold", ge=0.1, le=0.9)
    model: Optional[str] = Field(default=None, description="Model to use (optional)")
    language: str = Field(default="en", description="Response language (en/tr)") 

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback_type: str  # 'helpful', 'not_helpful'
    provider: str
    model: str

# DATABASE DEPENDENCY
async def get_db_connection():
    """Database connection dependency"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection not available")
    
    try:
        async with db_pool.acquire() as connection:
            yield connection
    except Exception as e:
        logger.error(f"DB connection error: {e}")
        raise HTTPException(status_code=503, detail="DB connection error")

# UTILITY FUNCTIONS
async def generate_session_id(question: str) -> str:
    """Generate unique session ID"""
    timestamp = str(int(time.time()))
    question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
    return f"session_{timestamp}_{question_hash}"

async def simple_log_query(question: str, provider: str, session_id: str):
    """Simple query logging"""
    logger.info(f"Query: {question[:50]}... | Provider: {provider} | Session: {session_id[:8]}")

async def retrieve_relevant_docs(question: str, max_results: int, 
                                 similarity_threshold: float,
                                 airline_preference: Optional[str] = None,
                                 use_category_hint: bool = True) -> Tuple[List[Dict], Dict]:
    """Document retrieval with optional category hints"""
    if not vector_ops:
        return [], {}
    
    try:
        start_time = time.time()
        
        # Category detection (if enabled)
        detected_categories = []
        if use_category_hint:
            semantic_results = await vector_ops.semantic_category_detection(question, airline_preference)
            detected_categories = [cat for cat, score in semantic_results]
        
        # Use enhanced similarity search
        docs = await vector_ops.similarity_search(
            query=question,
            airline_filter=airline_preference,
            limit=max_results,
            similarity_threshold=similarity_threshold,
            use_semantic_categories=use_category_hint
        )
        
        search_duration = time.time() - start_time
        track_vector_search(search_duration)
        
        metadata = {
            "detected_categories": detected_categories,
            "category_hint_used": use_category_hint,
            "search_duration": round(search_duration, 3)
        }
        
        return docs, metadata
        
    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        return [], {}

def calculate_retrieval_stats(retrieved_docs: List[Dict]) -> Dict[str, Any]:
    """Basic retrieval statistics"""
    if not retrieved_docs:
        return {
            "total_retrieved": 0,
            "avg_similarity": 0,
            "min_similarity": 0,
            "max_similarity": 0,
            "context_quality": "low"
        }
    
    similarities = [doc["similarity_score"] for doc in retrieved_docs]
    avg_sim = sum(similarities) / len(similarities)
    
    return {
        "total_retrieved": len(retrieved_docs),
        "avg_similarity": round(avg_sim, 3),
        "min_similarity": round(min(similarities), 3),
        "max_similarity": round(max(similarities), 3),
        "context_quality": (
            "high" if len(retrieved_docs) >= 2 and similarities[0] > 0.7 else
            "medium" if retrieved_docs and similarities[0] > 0.5 else
            "low"
        )
    }

def prepare_retrieved_docs_preview(retrieved_docs: List[Dict]) -> List[Dict]:
    """Simplified source preview"""
    return [
        {
            "airline": doc.get("airline", "Unknown"),
            "source": doc["source"], 
            "content_preview": doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"],
            "updated_date": doc.get("updated_at") or doc.get("created_at"),
            "url": doc.get("url"),
            "similarity_score": round(doc["similarity_score"], 3)
        }
        for doc in retrieved_docs
    ]

def fix_float_values(obj):
    """Fix NaN and Infinity values for JSON serialization"""
    if isinstance(obj, dict):
        return {k: fix_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_float_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    return obj

def calculate_preference_stats(retrieved_docs: List[Dict], airline_preference: Optional[str]) -> Dict[str, Any]:
    """Calculate preference-related statistics"""
    if not retrieved_docs or not airline_preference:
        return {
            "preference_applied": False,
            "boosted_results": 0,
            "preferred_airline_ratio": 0
        }
    
    boosted_count = len([doc for doc in retrieved_docs if doc.get('preference_boost', False)])
    preferred_count = len([doc for doc in retrieved_docs if doc.get('airline') == airline_preference])
    
    return {
        "preference_applied": True,
        "preferred_airline": airline_preference,
        "boosted_results": boosted_count,
        "preferred_airline_results": preferred_count,
        "preferred_airline_ratio": round(preferred_count / len(retrieved_docs), 2) if retrieved_docs else 0,
        "total_results": len(retrieved_docs)
    }

def enhance_query_simple(question: str, airline_preference: Optional[str]) -> str:
    """Simple rule-based query enhancement with airline prefix"""
    if not airline_preference:
        return question
    
    # Airline code to name mapping
    airline_names = {
        'turkish_airlines': 'Turkish Airlines',
        'pegasus': 'Pegasus Airlines',
        'sunexpress': 'SunExpress'
    }
    
    airline_name = airline_names.get(airline_preference, airline_preference)
    
    # Simple prefix addition
    enhanced_query = f"{airline_name} | {question}"
    
    # Log the enhancement
    logger.info(f"Query enhanced: '{question}' -> '{enhanced_query}'")
    
    return enhanced_query

# MAIN ENDPOINTS
@app.get("/")
async def read_root(db = Depends(get_db_connection)):
    """API main page and general information"""
    try:
        # Total policy count
        total_count = await db.fetchval("SELECT COUNT(*) FROM policy")
        
        # Available sources
        sources = await db.fetch("SELECT source, COUNT(*) as count FROM policy GROUP BY source")
        
        # AI services status
        ai_status = {
            "embedding_service": "loaded" if embedding_service else "failed",
            "openai_service": "loaded" if openai_service else "failed",
            "claude_service": "loaded" if claude_service else "failed",
            "vector_operations": "enabled" if vector_ops else "disabled"
        }
        
        return {
            "service": "Airlines Policy API - Unified Metrics",
            "version": "7.0.0-unified",
            "data_source": "PostgreSQL Database + Vector Search",
            "status": "active",
            "ai_services_status": ai_status,
            "metrics_status": {
                "unified_metrics": "active",
                "prometheus_instrumentator": "active",
                "business_intelligence": "enabled"
            },
            "unified_metrics_summary": get_metrics_summary(),
            "endpoints": {
                "all_policies": "/policies",
                "search": "/search?q=yoursearchterm",
                "vector_search": "/vector/similarity-search?q=yoursearchterm",
                "stats": "/stats",
                "health": "/health",
                "openai_chat": "/chat/openai",
                "claude_chat": "/chat/claude",
                "metrics": "/metrics",
                "documentation": "/docs"
            },
            "statistics": {
                "total_policies": total_count,
                "available_sources": [dict(row) for row in sources]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Main page loading error: {str(e)}")

@app.get("/policies", response_model=ApiResponse)
async def get_all_policies(
    limit: int = Query(50, description="Maximum number of policies to return", le=500),
    offset: int = Query(0, description="Starting point", ge=0),
    source: Optional[str] = Query(None, description="Source filter"),
    min_quality: Optional[float] = Query(None, description="Minimum quality score"),
    db = Depends(get_db_connection)
):
    """Return all policies"""
    
    try:
        # Base query
        query = "SELECT id, source, content, created_at, quality_score FROM policy WHERE 1=1"
        params = []
        param_count = 0
        
        # Filters
        if source:
            param_count += 1
            query += f" AND source = ${param_count}"
            params.append(source)
        
        if min_quality is not None:
            param_count += 1
            query += f" AND quality_score >= ${param_count}"
            params.append(min_quality)
        
        # Order and limit
        query += " ORDER BY created_at DESC"
        
        param_count += 1
        query += f" LIMIT ${param_count}"
        params.append(limit)
        
        param_count += 1
        query += f" OFFSET ${param_count}"
        params.append(offset)
        
        # Fetch data
        rows = await db.fetch(query, *params)
        
        policies = []
        for row in rows:
            policy_dict = dict(row)
            if policy_dict.get('created_at'):
                policy_dict['created_at'] = policy_dict['created_at'].isoformat()
            policies.append(BaggagePolicy(**policy_dict))
        
        return ApiResponse(
            success=True,
            message=f"Policies successfully returned",
            data=policies,
            count=len(policies)
        )
        
    except Exception as e:
        logger.error(f"Policies return error: {e}")
        raise HTTPException(status_code=500, detail=f"Policies return error: {str(e)}")

@app.get("/search")
async def search_policies(
    q: str = Query(..., description="Search query", min_length=2),
    limit: int = Query(10, description="Max results", le=50, ge=1),
    source: Optional[str] = Query(None, description="Source filter"),
    db = Depends(get_db_connection)
):
    """Simple text search in policies"""
    try:
        # Basic SQL search
        query = """
        SELECT id, source, content, created_at, quality_score 
        FROM policy 
        WHERE content ILIKE $1
        """
        params = [f"%{q}%"]
        
        if source:
            query += " AND source = $2"
            params.append(source)
            
        query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        rows = await db.fetch(query, *params)
        
        results = []
        for row in rows:
            policy_dict = dict(row)
            if policy_dict.get('created_at'):
                policy_dict['created_at'] = policy_dict['created_at'].isoformat()
            results.append(policy_dict)
        
        return {
            "success": True,
            "query": q,
            "results": results,
            "found_count": len(results),
            "search_type": "text_search"
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/vector/similarity-search")
async def vector_similarity_search(
    q: str = Query(..., description="Search query", min_length=2),
    limit: int = Query(5, description="Max results", le=20, ge=1),
    threshold: float = Query(0.3, description="Similarity threshold", ge=0.1, le=0.9),
    db = Depends(get_db_connection)
):
    """Vector similarity search with unified metrics"""
    try:
        if not vector_ops:
            raise HTTPException(status_code=503, detail="Vector search not available")
        
        start_time = time.time()
        docs = await vector_ops.similarity_search(
            query=q,
            limit=limit,
            similarity_threshold=threshold
        )
        
        # Track performance with unified metrics
        search_duration = time.time() - start_time
        track_vector_search(search_duration)
        
        return {
            "success": True,
            "query": q,
            "results": docs,
            "found_count": len(docs),
            "search_type": "vector_similarity",
            "performance": {
                "duration_seconds": round(search_duration, 3)
            }
        }
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search error: {str(e)}")

@app.get("/stats")
async def get_stats(db = Depends(get_db_connection)):
    """Simple database statistics with unified metrics info"""
    try:
        # Basic stats
        total_policies = await db.fetchval("SELECT COUNT(*) FROM policy")
        
        source_breakdown = await db.fetch("""
            SELECT source, COUNT(*) as count 
            FROM policy 
            GROUP BY source 
            ORDER BY count DESC
        """)
        
        # Recent activity
        recent_count = await db.fetchval("""
            SELECT COUNT(*) FROM policy 
            WHERE created_at > NOW() - INTERVAL '7 days'
        """)
        
        return {
            "total_policies": total_policies,
            "source_breakdown": [dict(row) for row in source_breakdown],
            "recent_additions": recent_count,
            "database_info": {
                "status": "connected",
                "pool_size": db_pool.get_size() if db_pool else 0
            },
            "ai_services": {
                "embedding": "ready" if embedding_service else "unavailable",
                "vector_search": "ready" if vector_ops else "unavailable",
                "openai": "ready" if openai_service else "unavailable", 
                "claude": "ready" if claude_service else "unavailable",
                "category_enhancement": "active"
            },
            "unified_metrics": get_metrics_summary()
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

# ENHANCED OPENAI CHAT ENDPOINT WITH UNIFIED METRICS
async def _chat_with_openai_logic(question: str,
                                  max_results: int,
                                  similarity_threshold: float,
                                  model: Optional[str],
                                  airline_preference: Optional[str] = None,
                                  language: str = "en"):
    """Enhanced OpenAI chat logic with unified metrics tracking"""
    if not openai_service:
        raise HTTPException(status_code=503, detail="OpenAI service not available")
    
    start_time = time.time()
    session_id = await generate_session_id(question)
    
    try:
        preference_log = f" | Airline: {airline_preference}" if airline_preference else ""
        await simple_log_query(question, "openai" + preference_log, session_id)

        enhanced_question = enhance_query_simple(question, airline_preference)
        
        # Step 1: Retrieve documents (with component timing)
        retrieval_start = time.time()
        retrieved_docs, retrieval_metadata = await retrieve_relevant_docs(enhanced_question, max_results, similarity_threshold, airline_preference)
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Generate response (with component timing)
        generation_start = time.time()
        openai_response = openai_service.generate_rag_response(retrieved_docs, enhanced_question, model, language)
        generation_time = time.time() - generation_start
        
        if not openai_response["success"]:
            total_duration = time.time() - start_time
            track_rag_query("openai", total_duration, "error")
            raise HTTPException(status_code=500, detail="OpenAI generation failed")
        
        # Step 3: Unified Metrics Tracking
        total_duration = time.time() - start_time
        usage = openai_response.get("usage", {})
        
        # Track operational metrics (basic)
        track_rag_query("openai", total_duration, "success")
        
        # Track component latency
        track_component_latency("retrieval", "openai", retrieval_time)
        track_component_latency("generation", "openai", generation_time)
        track_component_latency("total", "openai", total_duration)
        
        # Track business intelligence metrics
        source_precision = calculate_source_precision(retrieved_docs, 0.7)
        answer_completeness = calculate_answer_completeness(len(retrieved_docs), 3)
        accuracy_score = calculate_overall_accuracy(source_precision, answer_completeness)
        
        # Track accuracy
        track_rag_accuracy(
            provider="openai",
            airline=airline_preference or "all",
            language=language,
            accuracy_score=accuracy_score
        )
        
        # Track cost metrics
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_cost = usage.get("estimated_cost", 0.0)
        
        if total_cost > 0:
            track_query_cost(
                provider="openai",
                model=openai_response.get("model_used", model or "gpt-4o-mini"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost_usd=total_cost,
                accuracy_score=accuracy_score
            )
        
        # Enhanced response data
        response_data = {
            "success": True,
            "session_id": session_id,
            "question": question,
            "answer": openai_response["answer"],
            "model_used": openai_response["model_used"],
            "airline_preference": airline_preference,
            "sources": prepare_retrieved_docs_preview(retrieved_docs),
            "stats": calculate_retrieval_stats(retrieved_docs),
            "preference_stats": calculate_preference_stats(retrieved_docs, airline_preference),
            "performance": {
                "response_time": round(total_duration, 3),
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
                "cost": total_cost
            },
            "accuracy": {
                "overall_score": round(accuracy_score, 3),
                "source_precision": round(source_precision, 3),
                "answer_completeness": round(answer_completeness, 3)
            },
            "language": language,
            "metrics_tracked": "unified_system",
            "retrieval_metadata": retrieval_metadata
        }
        
        logger.info(f"OpenAI response generated in {total_duration:.2f}s (R:{retrieval_time:.2f}s G:{generation_time:.2f}s) Accuracy:{accuracy_score:.3f}")
        return fix_float_values(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        track_rag_query("openai", total_duration, "error")
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(status_code=500, detail="OpenAI chat error")

# ENHANCED CLAUDE CHAT ENDPOINT WITH UNIFIED METRICS
async def _chat_with_claude_logic(question: str,
                                  max_results: int,
                                  similarity_threshold: float,
                                  model: Optional[str],
                                  airline_preference: Optional[str] = None,
                                  language: str = "en"):
    """Enhanced Claude chat logic with unified metrics tracking"""
    if not claude_service:
        raise HTTPException(status_code=503, detail="Claude service not available")
    
    start_time = time.time()
    session_id = await generate_session_id(question)
    
    try:
        preference_log = f" | Airline: {airline_preference}" if airline_preference else ""
        await simple_log_query(question, "claude" + preference_log, session_id)

        enhanced_question = enhance_query_simple(question, airline_preference)

        # Step 1: Retrieve documents (with component timing)
        retrieval_start = time.time()
        retrieved_docs, retrieval_metadata = await retrieve_relevant_docs(enhanced_question, max_results, similarity_threshold, airline_preference)
        retrieval_time = time.time() - retrieval_start
        
        # Step 2: Generate response (with component timing)
        generation_start = time.time()
        claude_response = claude_service.generate_rag_response(retrieved_docs, enhanced_question, model, language)
        generation_time = time.time() - generation_start
        
        if not claude_response["success"]:
            total_duration = time.time() - start_time
            track_rag_query("claude", total_duration, "error")
            raise HTTPException(status_code=500, detail="Claude generation failed")
        
        # Step 3: Unified Metrics Tracking
        total_duration = time.time() - start_time
        usage = claude_response.get("usage", {})
        
        # Track operational metrics (basic)
        track_rag_query("claude", total_duration, "success")
        
        # Track component latency
        track_component_latency("retrieval", "claude", retrieval_time)
        track_component_latency("generation", "claude", generation_time)
        track_component_latency("total", "claude", total_duration)
        
        # Track business intelligence metrics
        source_precision = calculate_source_precision(retrieved_docs, 0.7)
        answer_completeness = calculate_answer_completeness(len(retrieved_docs), 3)
        accuracy_score = calculate_overall_accuracy(source_precision, answer_completeness)
        
        # Track accuracy
        track_rag_accuracy(
            provider="claude",
            airline=airline_preference or "all",
            language=language,
            accuracy_score=accuracy_score
        )
        
        # Track cost metrics
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_cost = usage.get("estimated_cost", 0.0)
        
        if total_cost > 0:
            track_query_cost(
                provider="claude",
                model=claude_response.get("model_used", model or "claude-3-5-haiku-20241022"),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_cost_usd=total_cost,
                accuracy_score=accuracy_score
            )
        
        # Enhanced response data
        response_data = {
            "success": True,
            "session_id": session_id,
            "question": question,
            "answer": claude_response["answer"],
            "model_used": claude_response["model_used"],
            "airline_preference": airline_preference,
            "sources": prepare_retrieved_docs_preview(retrieved_docs),
            "stats": calculate_retrieval_stats(retrieved_docs),
            "preference_stats": calculate_preference_stats(retrieved_docs, airline_preference),
            "performance": {
                "response_time": round(total_duration, 3),
                "retrieval_time": round(retrieval_time, 3),
                "generation_time": round(generation_time, 3),
                "cost": total_cost
            },
            "accuracy": {
                "overall_score": round(accuracy_score, 3),
                "source_precision": round(source_precision, 3),
                "answer_completeness": round(answer_completeness, 3)
            },
            "language": language,
            "metrics_tracked": "unified_system",
            "retrieval_metadata": retrieval_metadata
        }
        
        logger.info(f"Claude response generated in {total_duration:.2f}s (R:{retrieval_time:.2f}s G:{generation_time:.2f}s) Accuracy:{accuracy_score:.3f}")
        return fix_float_values(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        total_duration = time.time() - start_time
        track_rag_query("claude", total_duration, "error")
        logger.error(f"Claude error: {e}")
        raise HTTPException(status_code=500, detail="Claude chat error")

# Chat Endpoints - All HTTP methods supported
@app.post("/chat/openai")
async def openai_chat_post(
    chat_request: Optional[ChatRequest] = None,
    question: Optional[str] = Query(None, description="User question", min_length=3),
    airline_preference: Optional[str] = Query(None, description="Preferred airline"),
    max_results: Optional[int] = Query(None, description="Max retrieved documents", le=10, ge=1),
    similarity_threshold: Optional[float] = Query(None, description="Similarity threshold", ge=0.1, le=0.9),
    model: Optional[str] = Query(None, description="Model to use (optional)"),
    language: Optional[str] = Query("en", description="Response language (en/tr)")
):
    """RAG Chat with OpenAI (POST method) - Unified Metrics"""
    
    if chat_request:
        return await _chat_with_openai_logic(
            chat_request.question,
            chat_request.max_results,
            chat_request.similarity_threshold,
            chat_request.model,
            chat_request.airline_preference,
            chat_request.language
        )
    elif question:
        return await _chat_with_openai_logic(
            question,
            max_results or 3,
            similarity_threshold or 0.3,
            model,
            airline_preference,
            language or "en"
        )
    else:
        raise HTTPException(
            status_code=422, 
            detail="Either provide JSON body with 'question' field or 'question' query parameter"
        )

@app.get("/chat/openai") 
async def openai_chat_get(
    question: str = Query(..., description="User question", min_length=3),
    airline_preference: Optional[str] = Query(None, description="Preferred airline"),
    max_results: int = Query(5, description="Max retrieved documents", le=10, ge=1),
    similarity_threshold: float = Query(0.3, description="Similarity threshold", ge=0.1, le=0.9),
    model: Optional[str] = Query(None, description="Model to use (optional)"),
    language: str = Query("en", description="Response language (en/tr)")
):
    """RAG Chat with OpenAI (GET method) - Unified Metrics"""
    return await _chat_with_openai_logic(question, max_results, similarity_threshold, model, airline_preference, language)

@app.post("/chat/claude")
async def claude_chat_post(
    chat_request: Optional[ChatRequest] = None,
    question: Optional[str] = Query(None, description="User question", min_length=3),
    airline_preference: Optional[str] = Query(None, description="Preferred airline"),
    max_results: Optional[int] = Query(None, description="Max retrieved documents", le=10, ge=1),
    similarity_threshold: Optional[float] = Query(None, description="Similarity threshold", ge=0.1, le=0.9),
    model: Optional[str] = Query(None, description="Model to use (optional)"),
    language: str = Query("en", description="Response language (en/tr)")
):
    """RAG Chat with Claude (POST method) - Unified Metrics"""
    
    if chat_request:
        return await _chat_with_claude_logic(
            chat_request.question,
            chat_request.max_results,
            chat_request.similarity_threshold,
            chat_request.model,
            chat_request.airline_preference,
            chat_request.language
        )
    elif question:
        return await _chat_with_claude_logic(
            question,
            max_results or 3,
            similarity_threshold or 0.3,
            model,
            airline_preference,
            language or "en"
        )
    else:
        raise HTTPException(
            status_code=422, 
            detail="Either provide JSON body with 'question' field or 'question' query parameter"
        )

@app.get("/chat/claude")
async def claude_chat_get(
    question: str = Query(..., description="User question", min_length=3),
    airline_preference: Optional[str] = Query(None, description="Preferred airline"),
    max_results: int = Query(5, description="Max retrieved documents", le=10, ge=1),
    similarity_threshold: float = Query(0.3, description="Similarity threshold", ge=0.1, le=0.9),
    model: Optional[str] = Query(None, description="Model to use (optional)"),
    language: str = Query("en", description="Response language (en/tr)")
):
    """RAG Chat with Claude (GET method) - Unified Metrics"""
    return await _chat_with_claude_logic(question, max_results, similarity_threshold, model, airline_preference, language)

# FEEDBACK ENDPOINT WITH UNIFIED METRICS
@app.post("/feedback")
async def collect_user_feedback(feedback: FeedbackRequest):
    """User feedback collection with unified metrics tracking"""
    try:
        rating = "thumbs_up" if feedback.feedback_type == "helpful" else "thumbs_down"
        track_user_feedback(rating)
        
        logger.info(f"Feedback: {feedback.feedback_type} for {feedback.provider}")
        
        return {
            "success": True,
            "message": "Feedback recorded in unified metrics system"
        }
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Feedback error")

# SPEECH ENDPOINTS (unchanged)
@app.post("/speech/synthesize")
async def text_to_speech(
    text: str,
    language: str = "tr-TR"
):
    """AWS Polly TTS"""
    try:
        if not aws_speech_service:
            raise HTTPException(status_code=503, detail="AWS Speech Service not available")
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Empty text not allowed")
        
        result = aws_speech_service.text_to_audio(text, language)
        
        if result["success"]:
            return Response(
                content=result["audio_data"],
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "attachment; filename=speech.mp3"
                }
            )
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")

@app.post("/speech/transcribe")
async def speech_to_text_realtime(
    audio_file: UploadFile = File(...),
    language: str = Query("tr", description="Language code (tr, en, etc.)")
):
    """AssemblyAI STT"""
    try:
        if not assemblyai_service:
            raise HTTPException(status_code=503, detail="AssemblyAI service not available")
        
        # Validate file
        if not audio_file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file size (limit to 25MB)
        MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB
        audio_bytes = await audio_file.read()
        
        if len(audio_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large (max 25MB)")
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Log the request
        logger.info(f"STT request: {audio_file.filename} ({len(audio_bytes)} bytes, lang: {language})")
        
        # Transcribe using AssemblyAI
        result = await assemblyai_service.transcribe_audio_file(
            audio_bytes=audio_bytes,
            language=language,
            filename=audio_file.filename
        )
        
        if result["success"]:
            response_data = {
                "success": True,
                "transcript": result["transcript"],
                "language": language,
                "filename": audio_file.filename,
                "file_size_bytes": len(audio_bytes),
                "details": result["details"],
                "service": "AssemblyAI",
                "timestamp": time.time()
            }
            
            logger.info(f"STT success: '{result['transcript'][:50]}...' ({result['details'].get('words_count', 0)} words)")
            return response_data
        else:
            logger.error(f"STT failed: {result['error']}")
            raise HTTPException(
                status_code=500, 
                detail=f"Transcription failed: {result['error']}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"STT error: {str(e)}")

# METRICS ENDPOINT - Unified Prometheus export
@app.get("/metrics")
async def get_metrics():
    """Unified metrics endpoint for Prometheus scraping"""
    metrics_text = get_metrics_text()
    return Response(
        content=metrics_text,
        media_type=CONTENT_TYPE_LATEST
    )

# HEALTH ENDPOINTS (updated with unified metrics info)
@app.get("/health")
async def health_check(db = Depends(get_db_connection)):
    """Health check endpoint with unified metrics status"""
    try:
        total_count = await db.fetchval("SELECT COUNT(*) FROM policy")
        
        ai_status = {
            "embedding_service": embedding_service is not None,
            "openai_service": openai_service is not None,
            "claude_service": claude_service is not None,
            "vector_operations": vector_ops is not None
        }
        
        return {
            "status": "healthy",
            "models_ready": all(ai_status.values()),
            "database": {"connected": True, "policies_count": total_count},
            "ai_services": ai_status,
            "speech_services": {
                "tts": aws_speech_service is not None,
                "stt": assemblyai_service is not None
            },
            "metrics": {
                "unified_system": "active",
                "prometheus_instrumentator": "active",
                "business_intelligence": "enabled"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.get("/speech/health")
async def speech_health_check():
    """Speech Services health check - TTS + STT"""
    try:
        if not aws_speech_service:
            return {
                "status": "unhealthy", 
                "error": "TTS Service not initialized",
                "services": {
                    "polly": "unavailable",
                    "assemblyai": "not_loaded"
                }
            }
        
        # Polly test (TTS)
        try:
            voices = aws_speech_service.polly_client.describe_voices(LanguageCode='tr-TR')
            polly_status = "ready"
            polly_info = f"{len(voices.get('Voices', []))} Turkish voices available"
        except Exception as e:
            polly_status = "failed"
            polly_info = str(e)
        
        assemblyai_status = (
                        "ready" 
                        if loader.get_secret('assemblyai_api_key', 'ASSEMBLYAI_API_KEY') 
                        else "not_configured"
                    )

        
        # Overall status
        overall_status = "healthy" if polly_status == "ready" and assemblyai_status == "ready" else "partial"
        
        return {
            "status": overall_status,
            "service": "Speech Services (TTS + STT)",
            "services": {
                "polly": {
                    "status": polly_status,
                    "info": polly_info,
                    "purpose": "Text-to-Speech"
                },
                "assemblyai": {
                    "status": assemblyai_status,
                    "info": "Real-time Speech-to-Text",
                    "purpose": "Speech-to-Text"
                }
            },
            "features": {
                "tts_enabled": polly_status == "ready",
                "stt_enabled": assemblyai_status == "ready",
                "supported_languages": ["tr-TR", "en-US", "en-GB"]
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/speech/assemblyai/info")
async def get_assemblyai_info():
    """Get AssemblyAI service information"""
    try:
        if not assemblyai_service:
            return {
                "status": "unavailable",
                "error": "AssemblyAI service not initialized"
            }
        
        info = assemblyai_service.get_service_info()
        return info
        
    except Exception as e:
        logger.error(f"AssemblyAI info error: {e}")
        return {
            "status": "error", 
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "myapp:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )