# Required imports
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import Response
from typing import List, Optional, Dict, Any
import asyncpg
import os
from contextlib import asynccontextmanager
import logging
from pydantic import BaseModel, Field
from functools import lru_cache
from pydantic_settings import BaseSettings
from embedding_service import get_embedding_service
from vector_operations import VectorOperations
from openai_service import get_openai_service
from claude_service import get_claude_service
import math
from typing import Dict, Optional
import uuid
from datetime import datetime, timedelta
import hashlib

# Basic Prometheus instrumentation
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest
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

# SIMPLIFIED METRICS SYSTEM - Only 5 core metrics
try:
    # Try to use centralized metrics if available
    from monitoring.metrics import (
        http_requests_total,
        rag_query_duration_seconds,
        vector_search_duration_seconds,
        ai_api_cost_total_usd,
        user_satisfaction_total
    )
    
    # Core metrics
    REQUEST_COUNT = http_requests_total
    RAG_DURATION = rag_query_duration_seconds  
    VECTOR_DURATION = vector_search_duration_seconds
    API_COST = ai_api_cost_total_usd
    USER_SATISFACTION = user_satisfaction_total
    
    METRICS_AVAILABLE = True
    print("Using centralized metrics")
    
except ImportError:
    # Fallback: create basic metrics
    REQUEST_COUNT = Counter(
        'http_requests_total', 
        'Total HTTP requests', 
        ['method', 'endpoint', 'status_code']
    )
    
    RAG_DURATION = Histogram(
        'rag_query_duration_seconds',
        'RAG query duration',
        ['provider', 'status'],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    )
    
    VECTOR_DURATION = Histogram(
        'vector_search_duration_seconds',
        'Vector search duration',
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    )
    
    API_COST = Counter(
        'ai_api_cost_total_usd',
        'Total AI API cost in USD',
        ['provider', 'model']
    )
    
    USER_SATISFACTION = Counter(
        'user_satisfaction_total',
        'User satisfaction feedback',
        ['rating']  # thumbs_up, thumbs_down
    )
    
    METRICS_AVAILABLE = True
    print("Using fallback metrics")

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSettings(BaseSettings):
    # Database connection parameters
    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    database: str = Field(..., description="Database name (required)")
    user: str = Field(..., description="Database user (required)")
    password: str = Field(..., description="Database password (required)")
    
    # Connection pool settings
    min_pool_size: int = Field(default=5, ge=1, le=20)
    max_pool_size: int = Field(default=20, ge=5, le=100)
    command_timeout: int = Field(default=60, ge=10, le=300)
    
    # SSL and advanced settings
    ssl: bool = Field(default=False, description="Enable SSL connection")
    echo: bool = Field(default=False, description="Echo SQL queries (debug)")
    
    openai_api_key: str = Field(default_factory=lambda: os.getenv('OPENAI_API_KEY', 'dummy'))
    
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

# SIMPLIFIED HELPER FUNCTIONS
def track_rag_query(provider: str, duration: float, status: str = "success"):
    """Simple RAG query tracking"""
    RAG_DURATION.labels(provider=provider, status=status).observe(duration)

def track_vector_search(duration: float):
    """Simple vector search tracking"""  
    VECTOR_DURATION.observe(duration)

def track_api_cost(provider: str, model: str, cost: float):
    """Simple API cost tracking"""
    API_COST.labels(provider=provider, model=model).inc(cost)

def track_user_feedback(rating: str):
    """Simple user feedback tracking"""
    USER_SATISFACTION.labels(rating=rating).inc()

def track_http_request(method: str, endpoint: str, status_code: str):
    """Simple HTTP request tracking"""
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()

class SimplifiedMetricsMiddleware(BaseHTTPMiddleware):
    """Simplified FastAPI Middleware for basic metrics"""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        method = request.method
        path = request.url.path
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            status_code = str(response.status_code)
            
            # Track basic metrics
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
    logger.info("Airlines Policy API starting...")
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
                vector_ops = VectorOperations(db_pool)
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
        
        # 6. Startup summary
        services_status = {
            "database": "ready" if db_pool else "failed",
            "embedding": "ready" if embedding_service else "failed",
            "openai": "ready" if openai_service else "failed",
            "claude": "ready" if claude_service else "failed",
            "vector_ops": "ready" if vector_ops else "failed",
            "metrics": "ready" if METRICS_AVAILABLE else "failed",
            "aws_tts": "ready" if aws_speech_service else "failed",
            "assemblyai_stt": "ready" if assemblyai_service else "failed"
        }
        
        logger.info("Startup Summary:")
        for service, status in services_status.items():
            logger.info(f"   {service}: {status}")
        
        logger.info("Airlines Policy API ready!")
            
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
    
    logger.info("Shutdown completed")

# FastAPI instance
app = FastAPI(
    title="Airlines Policy API - Simplified",
    description="RAG-powered PostgreSQL API with Clean Metrics",
    version="5.0.0",
    lifespan=lifespan
)

# Add simplified middleware
app.add_middleware(SimplifiedMetricsMiddleware)

app.add_middleware(
    CORSMiddleware,  # ❌ Bu eksik - ZORUNLU
    allow_origins=[
        "http://localhost:8501",
        "http://localhost:5173", 
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add Prometheus instrumentation
instrumentator = Instrumentator()
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
    max_results: int = Field(default=3, description="Max retrieved documents", le=10, ge=1)
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
    """Simple query logging instead of complex database tracking"""
    logger.info(f"Query: {question[:50]}... | Provider: {provider} | Session: {session_id[:8]}")

async def retrieve_relevant_docs(question: str, max_results: int, 
                                 similarity_threshold: float,
                                 airline_preference: Optional[str] = None) -> List[Dict]:
    """Document retrieval with simplified metrics"""
    if not vector_ops:
        return []
    
    try:
        start_time = time.time()
        
        # FIX: Parameter adını airline_filter olarak değiştir
        docs = await vector_ops.similarity_search(
            query=question,
            airline_filter=airline_preference,  # airline_preference -> airline_filter
            limit=max_results,
            similarity_threshold=similarity_threshold
        )
        
        # Simple metrics tracking
        search_duration = time.time() - start_time
        track_vector_search(search_duration)
        
        preference_info = f" (preference: {airline_preference})" if airline_preference else ""
        logger.info(f"Retrieved {len(docs)} documents in {search_duration:.2f}s{preference_info}")
        return docs
        
    except Exception as e:
        logger.error(f"Document retrieval error: {e}")
        return []

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
    """Retrieved docs preview"""
    return [
        {
            "source": doc["source"],
            "airline": doc.get("airline", "Unknown"),  # Havayolu bilgisi ekle
            "content_preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
            "similarity_score": round(doc["similarity_score"], 3),
            "source_display": f"{doc['source']} - {doc.get('airline', 'Unknown Airline')}"  # Display için
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
            "service": "Airlines Policy API - Simplified",
            "version": "5.0.0 (Clean Architecture)",
            "data_source": "PostgreSQL Database + Vector Search",
            "status": "active",
            "ai_services_status": ai_status,
            "metrics_status": {
                "simplified_metrics": METRICS_AVAILABLE,
                "core_metrics_only": True
            },
            "core_metrics": [
                "http_requests_total",
                "rag_query_duration_seconds", 
                "vector_search_duration_seconds",
                "ai_api_cost_total_usd",
                "user_satisfaction_total"
            ],
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
    """Vector similarity search"""
    try:
        if not vector_ops:
            raise HTTPException(status_code=503, detail="Vector search not available")
        
        start_time = time.time()
        docs = await vector_ops.similarity_search(
            query=q,
            limit=limit,
            similarity_threshold=threshold
        )
        
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
    """Simple database statistics"""
    try:
        # Basic stats only
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
                "claude": "ready" if claude_service else "unavailable"
            }
        }
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")
    
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

# SIMPLIFIED AI CHAT ENDPOINTS
async def _chat_with_openai_logic(question: str,
                                  max_results: int,
                                  similarity_threshold: float,
                                  model: Optional[str],
                                  airline_preference: Optional[str] = None,
                                  language: str = "en"):
    """Simplified OpenAI chat logic"""
    if not openai_service:
        raise HTTPException(status_code=503, detail="OpenAI service not available")
    
    start_time = time.time()
    session_id = await generate_session_id(question)
    
    try:
        preference_log = f" | Airline: {airline_preference}" if airline_preference else ""
        # Simple logging instead of complex database storage
        await simple_log_query(question, "openai" + preference_log, session_id)

        enhanced_question = enhance_query_simple(question, airline_preference)
        
        # Step 1: Retrieve documents
        retrieved_docs = await retrieve_relevant_docs(enhanced_question, max_results, similarity_threshold,airline_preference)
        
        # Step 2: Generate response
        openai_response = openai_service.generate_rag_response(retrieved_docs, enhanced_question, model, language)
        
        if not openai_response["success"]:
            duration = time.time() - start_time
            track_rag_query("openai", duration, "error")
            raise HTTPException(status_code=500, detail="OpenAI generation failed")
        
        # Step 3: Track metrics
        duration = time.time() - start_time
        track_rag_query("openai", duration, "success")
        
        # Track cost
        usage = openai_response.get("usage", {})
        if "estimated_cost" in usage:
            track_api_cost("openai", openai_response.get("model_used", model or "unknown"), usage["estimated_cost"])
        
        # Simple response
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
                "response_time": round(duration, 3),
                "cost": usage.get("estimated_cost", 0.0)
            },
            "language": language
        }
        
        logger.info(f"OpenAI response generated in {duration:.2f}s (preference: {airline_preference or 'none'})")
        return fix_float_values(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        track_rag_query("openai", duration, "error")
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(status_code=500, detail="OpenAI chat error")

async def _chat_with_claude_logic(question: str,
                                  max_results: int,
                                  similarity_threshold: float,
                                  model: Optional[str],
                                  airline_preference: Optional[str] = None,
                                  language: str = "en"):
    """Simplified Claude chat logic"""
    if not claude_service:
        raise HTTPException(status_code=503, detail="Claude service not available")
    
    start_time = time.time()
    session_id = await generate_session_id(question)
    
    try:
        # Simple logging instead of complex database storage
        preference_log = f" | Airline: {airline_preference}" if airline_preference else ""
        await simple_log_query(question, "claude" + preference_log, session_id)

        enhanced_question = enhance_query_simple(question, airline_preference)

        # Step 1: Retrieve documents
        retrieved_docs = await retrieve_relevant_docs(
            enhanced_question,
            max_results, 
            similarity_threshold,
            airline_preference
        )
        
        # Step 2: Generate response
        claude_response = claude_service.generate_rag_response(retrieved_docs, enhanced_question, model, language)
        
        if not claude_response["success"]:
            duration = time.time() - start_time
            track_rag_query("claude", duration, "error")
            raise HTTPException(status_code=500, detail="Claude generation failed")
        
        # Step 3: Track metrics
        duration = time.time() - start_time
        track_rag_query("claude", duration, "success")
        
        # Track cost
        usage = claude_response.get("usage", {})
        if "estimated_cost" in usage:
            track_api_cost("claude", claude_response.get("model_used", model or "unknown"), usage["estimated_cost"])
        
        # Simple response
        response_data = {
            "success": True,
            "session_id": session_id,
            "question": question,
            "answer": claude_response["answer"],
            "model_used": claude_response["model_used"],
            "airline_preference": airline_preference,  # NEW: Include preference in response
            "sources": prepare_retrieved_docs_preview(retrieved_docs),
            "stats": calculate_retrieval_stats(retrieved_docs),
            "preference_stats": calculate_preference_stats(retrieved_docs, airline_preference),
            "performance": {
                "response_time": round(duration, 3),
                "cost": usage.get("estimated_cost", 0.0)
            },
            "language": language
        }
        
        logger.info(f"Claude response generated in {duration:.2f}s (preference: {airline_preference or 'none'})")
        return fix_float_values(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        track_rag_query("claude", duration, "error")
        logger.error(f"Claude error: {e}")
        raise HTTPException(status_code=500, detail="Claude chat error")

# Chat Endpoints
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
    """RAG Chat with OpenAI (POST method)"""
    
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

@app.post("/speech/synthesize")
async def text_to_speech(
    text: str,
    language: str = "tr-TR"
):
    """AWS Polly ile TTS"""
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

@app.get("/chat/openai") 
async def openai_chat_get(
    question: str = Query(..., description="User question", min_length=3),
    airline_preference: Optional[str] = Query(None, description="Preferred airline"),
    max_results: int = Query(3, description="Max retrieved documents", le=10, ge=1),
    similarity_threshold: float = Query(0.3, description="Similarity threshold", ge=0.1, le=0.9),
    model: Optional[str] = Query(None, description="Model to use (optional)"),
    language: str = Query("en", description="Response language (en/tr)")
):
    """RAG Chat with OpenAI (GET method)"""
    return await _chat_with_openai_logic(question, max_results, similarity_threshold, model, airline_preference,language)

@app.post("/chat/claude")
async def claude_chat_post(
    chat_request: Optional[ChatRequest] = None,
    question: Optional[str] = Query(None, description="User question", min_length=3),
    airline_preference: Optional[str] = Query(None, description="Preferred airline"),
    max_results: Optional[int] = Query(None, description="Max retrieved documents", le=10, ge=1),
    similarity_threshold: Optional[float] = Query(None, description="Similarity threshold", ge=0.1, le=0.9),
    model: Optional[str] = Query(None, description="Model to use (optional)"),
    language: str = Query("en", description="Response language (en/tr)")  # BU PARAMETRE EKLENDI
):
    """RAG Chat with Claude (POST method)"""
    
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
    max_results: int = Query(3, description="Max retrieved documents", le=10, ge=1),
    similarity_threshold: float = Query(0.3, description="Similarity threshold", ge=0.1, le=0.9),
    model: Optional[str] = Query(None, description="Model to use (optional)"),
    language: str = Query("en", description="Response language (en/tr)")
):
    """RAG Chat with Claude (GET method)"""
    return await _chat_with_claude_logic(question, max_results, similarity_threshold, model, airline_preference,language)

# SIMPLIFIED FEEDBACK ENDPOINT
@app.post("/feedback")
async def collect_user_feedback(feedback: FeedbackRequest):
    """Simplified user feedback"""
    try:
        rating = "thumbs_up" if feedback.feedback_type == "helpful" else "thumbs_down"
        track_user_feedback(rating)
        
        # Simple logging instead of complex database storage
        logger.info(f"Feedback: {feedback.feedback_type} for {feedback.provider}")
        
        return {
            "success": True,
            "message": "Feedback recorded"
        }
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Feedback error")

# METRICS ENDPOINT
@app.get("/metrics")
async def get_metrics():
    """Simple Prometheus metrics endpoint"""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.post("/speech/transcribe")
async def speech_to_text_realtime(
    audio_file: UploadFile = File(...),
    language: str = Query("tr", description="Language code (tr, en, etc.)")
):
    """AssemblyAI Real-time STT"""
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
            # Success response
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
            # Error response
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

# AssemblyAI service info endpoint'i de ekle:
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

@app.get("/health")
async def health_check(db = Depends(get_db_connection)):
    """Health check endpoint for frontend"""
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
        
        # AssemblyAI test (STT) - YENİ
        assemblyai_status = "ready" if os.getenv("ASSEMBLYAI_API_KEY") else "not_configured"
        
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

@app.get("/speech/debug")
async def debug_speech():
    """Speech Services debug - S3 kısımları kaldırıldı"""
    try:
        debug_info = {
            "environment_variables": {
                "AWS_REGION": os.getenv("AWS_REGION", "NOT_SET"),
                "AWS_ACCESS_KEY_ID": "SET" if os.getenv("AWS_ACCESS_KEY_ID") else "NOT_SET",
                "AWS_SECRET_ACCESS_KEY": "SET" if os.getenv("AWS_SECRET_ACCESS_KEY") else "NOT_SET",
                "ASSEMBLYAI_API_KEY": "SET" if os.getenv("ASSEMBLYAI_API_KEY") else "NOT_SET"  # YENİ
            }
        }
        
        if aws_speech_service:
            # Sadece Polly test
            try:
                voices = aws_speech_service.polly_client.describe_voices(LanguageCode='tr-TR')
                debug_info["polly_test"] = {
                    "status": "success",
                    "voices_count": len(voices.get('Voices', [])),
                    "sample_voices": [v.get('Name') for v in voices.get('Voices', [])[:3]]
                }
            except Exception as e:
                debug_info["polly_test"] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return debug_info
        
    except Exception as e:
        return {"error": f"Debug failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "myapp:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )