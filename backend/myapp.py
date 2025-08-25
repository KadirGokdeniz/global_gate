# Gerekli importlar
from fastapi import FastAPI, HTTPException, Query, Depends
from typing import List, Optional
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

import math

# Logging ayarları
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

@lru_cache()
def get_settings() -> DatabaseSettings:
    return DatabaseSettings()

settings = get_settings()

# Global connection pool
db_pool = None
# ===== New RAG GLOBAL VARIABLE =====
vector_ops = None

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlatma ve kapatma işlemleri"""
    global db_pool, vector_ops 
    
    # Startup
    logger.info("🚀 Turkish Airlines API is starting...")
    try:
        db_pool = await asyncpg.create_pool(
            **settings.get_asyncpg_params(),
            min_size=settings.min_pool_size,
            max_size=settings.max_pool_size,
            command_timeout=settings.command_timeout
        )
        logger.info("✅ PostgreSQL connection pool was created.")
        
        # Test bağlantısı
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM baggage_policies")
            logger.info(f"📊 In the database, {result} policy are available.")
        
        # ===== New RAG INITIALIZATION =====
        try:
            vector_ops = VectorOperations(db_pool)
            logger.info("✅ Vector operations initialized")
            
            # Auto-embed existing policies
            embedded_count = await vector_ops.embed_existing_policies()
            logger.info(f"🧠 Embedded {embedded_count} policies on startup")
            
        except Exception as ve:
            logger.error(f"⚠️ Vector operations init warning: {ve}")
            # Don't fail startup if vector ops fail
            
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        raise
    
    yield
    
    # Shutdown
    if db_pool:
        await db_pool.close()
        logger.info("🔒 Database connection is closed")

# FastAPI instance
app = FastAPI(
    title="Airlines Policy API",
    description="RAG-powered PostgreSQL API",
    version="4.0.0", # dummy
    lifespan=lifespan
)

# Pydantic Models
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

class SearchResponse(BaseModel):
    success: bool
    query: str
    results: List[BaggagePolicy]
    found_count: int
    total_in_db: int

class StatsResponse(BaseModel):
    total_policies: int
    source_breakdown: dict
    database_info: dict

# Database Dependency
async def get_db_connection():
    """Database conenction dependency"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database coonection is not available.")
    
    try:
        async with db_pool.acquire() as connection:
            yield connection
    except Exception as e:
        logger.error(f"DB connection error: {e}")
        raise HTTPException(status_code=503, detail="DB connection error")

# Main Page
@app.get("/")
async def read_root(db = Depends(get_db_connection)):
    """API main page and general information"""
    try:
        # Toplam policy sayısı
        total_count = await db.fetchval("SELECT COUNT(*) FROM baggage_policies")
        
        # Mevcut kaynaklar
        sources = await db.fetch("SELECT source, COUNT(*) as count FROM baggage_policies GROUP BY source")
        
        return {
            "service": "Airlines Policy API",
            "version": "4.0.0",
            "data_source": "PostgreSQL Database + Vector Search",
            "status": "active",
            "rag_features": "enabled",
            "endpoints": {
                "all_policies": "/policies",
                "search": "/search?q=yoursearchterm",
                "vector_search": "/vector/similarity-search?q=yoursearchterm",
                "by_source": "/policies/{source_name}",
                "stats": "/stats",
                "vector_stats": "/vector/stats",
                "health": "/health",
                "documentation": "/docs",
                "openai_rag_chat": "/chat/openai"
            },
            "statistics": {
                "total_policies": total_count,
                "available_sources": [dict(row) for row in sources]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Main page loading error: {str(e)}")

# Tüm Policies
@app.get("/policies", response_model=ApiResponse)
async def get_policies(
    limit: int = Query(50, description="Maximum can be returned policy number", le=500),
    offset: int = Query(0, description="Starting point", ge=0),
    source: Optional[str] = Query(None, description="Source filter"),
    min_quality: Optional[float] = Query(None, description="Minimum quality score"),
    db = Depends(get_db_connection)
):
    """Retun all policies"""
    
    try:
        # Base query
        query = "SELECT id, source, content, created_at, quality_score FROM baggage_policies WHERE 1=1"
        params = []
        param_count = 0
        
        # Filter
        if source:
            param_count += 1
            query += f" AND source = ${param_count}"
            params.append(source)
        
        if min_quality is not None:
            param_count += 1
            query += f" AND quality_score >= ${param_count}"
            params.append(min_quality)
        
        # Rate and limit
        query += " ORDER BY created_at DESC"
        
        param_count += 1
        query += f" LIMIT ${param_count}"
        params.append(limit)
        
        param_count += 1
        query += f" OFFSET ${param_count}"
        params.append(offset)
        
        # Veriyi çek
        rows = await db.fetch(query, *params)
        
        policies = []
        for row in rows:
            policy_dict = dict(row)
            if policy_dict.get('created_at'):
                policy_dict['created_at'] = policy_dict['created_at'].isoformat()
            policies.append(BaggagePolicy(**policy_dict))
        
        return ApiResponse(
            success=True,
            message=f"Policies successfully returned (PostgreSQL)",
            data=policies,
            count=len(policies)
        )
        
    except Exception as e:
        logger.error(f"Policies return error: {e}")
        raise HTTPException(status_code=500, detail=f"Policies return error: {str(e)}")

# Search
@app.get("/search", response_model=SearchResponse)
async def search_policies(
    q: str = Query(..., description="Search term", min_length=2),
    limit: int = Query(10, description="Maximum result number", le=100),
    source: Optional[str] = Query(None, description="Source Filter"),
    db = Depends(get_db_connection)
):
    """Advanced text search (PostgreSQL ILIKE)"""
    
    try:
        # Total record number
        total_query = "SELECT COUNT(*) FROM baggage_policies"
        total_count = await db.fetchval(total_query)
        
        # Arama query'si
        search_query = """
        SELECT id, source, content, created_at, quality_score 
        FROM baggage_policies 
        WHERE content ILIKE $1
        """
        params = [f"%{q}%"]
        
        # Kaynak filtresi
        if source:
            search_query += " AND source = $2"
            params.append(source)
        
        # Sıralama ve limit
        search_query += " ORDER BY quality_score DESC NULLS LAST, created_at DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        # Arama yap
        rows = await db.fetch(search_query, *params)
        
        results = []
        for row in rows:
            policy_dict = dict(row)
            if policy_dict.get('created_at'):
                policy_dict['created_at'] = policy_dict['created_at'].isoformat()
            results.append(BaggagePolicy(**policy_dict))
        
        return SearchResponse(
            success=True,
            query=q,
            results=results,
            found_count=len(results),
            total_in_db=total_count
        )
        
    except Exception as e:
        logger.error(f"Arama hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Arama hatası: {str(e)}")

# Source'a Göre Getir
@app.get("/policies/{source}", response_model=ApiResponse)
async def get_policies_by_source(
    source: str,
    limit: int = Query(100, description="Maksimum policy sayısı"),
    db = Depends(get_db_connection)
):
    """Belirli kaynağa göre policies getir"""
    
    try:
        # Önce kaynak var mı kontrol et
        source_check = await db.fetchval(
            "SELECT COUNT(*) FROM baggage_policies WHERE source = $1", 
            source
        )
        
        if source_check == 0:
            # Mevcut kaynakları göster
            available = await db.fetch("SELECT DISTINCT source FROM baggage_policies ORDER BY source")
            available_sources = [row['source'] for row in available]
            
            raise HTTPException(
                status_code=404,
                detail=f"Kaynak '{source}' bulunamadı. Mevcut kaynaklar: {available_sources}"
            )
        
        # Veriyi getir
        query = """
        SELECT id, source, content, created_at, quality_score 
        FROM baggage_policies 
        WHERE source = $1 
        ORDER BY quality_score DESC NULLS LAST, created_at DESC 
        LIMIT $2
        """
        
        rows = await db.fetch(query, source, limit)
        
        policies = []
        for row in rows:
            policy_dict = dict(row)
            if policy_dict.get('created_at'):
                policy_dict['created_at'] = policy_dict['created_at'].isoformat()
            policies.append(BaggagePolicy(**policy_dict))
        
        return ApiResponse(
            success=True,
            message=f"'{source}' kaynağından {len(policies)} policy getirildi",
            data=policies,
            count=len(policies)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Source policies getirme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Veri getirme hatası: {str(e)}")

# Kaynakları Listele
@app.get("/sources")
async def get_available_sources(db = Depends(get_db_connection)):
    """Mevcut veri kaynaklarını listele"""
    
    try:
        query = """
        SELECT 
            source,
            COUNT(*) as policy_count,
            AVG(quality_score) as avg_quality,
            MIN(created_at) as first_added,
            MAX(created_at) as last_added
        FROM baggage_policies 
        GROUP BY source 
        ORDER BY source
        """
        
        rows = await db.fetch(query)
        
        source_info = {}
        for row in rows:
            row_dict = dict(row)
            source = row_dict['source']
            
            # Tarihleri string'e çevir
            if row_dict.get('first_added'):
                row_dict['first_added'] = row_dict['first_added'].isoformat()
            if row_dict.get('last_added'):
                row_dict['last_added'] = row_dict['last_added'].isoformat()
            
            source_info[source] = {
                "policy_count": row_dict['policy_count'],
                "avg_quality": float(row_dict['avg_quality']) if row_dict['avg_quality'] else None,
                "first_added": row_dict['first_added'],
                "last_added": row_dict['last_added']
            }
        
        total_policies = await db.fetchval("SELECT COUNT(*) FROM baggage_policies")
        
        return {
            "available_sources": source_info,
            "total_sources": len(source_info),
            "total_policies": total_policies
        }
        
    except Exception as e:
        logger.error(f"Sources listeleme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Kaynak listeleme hatası: {str(e)}")

# İstatistikler
@app.get("/stats", response_model=StatsResponse)
async def get_stats(db = Depends(get_db_connection)):
    """Detaylı istatistikler"""
    
    try:
        # Toplam policy sayısı
        total_count = await db.fetchval("SELECT COUNT(*) FROM baggage_policies")
        
        # Kaynak bazlı istatistikler
        source_stats_query = """
        SELECT 
            source,
            COUNT(*) as count,
            AVG(quality_score) as avg_quality,
            MIN(quality_score) as min_quality,
            MAX(quality_score) as max_quality
        FROM baggage_policies 
        GROUP BY source 
        ORDER BY count DESC
        """
        
        source_rows = await db.fetch(source_stats_query)
        
        source_breakdown = {}
        for row in source_rows:
            source_breakdown[row['source']] = {
                "count": row['count'],
                "avg_quality": float(row['avg_quality']) if row['avg_quality'] else None,
                "min_quality": float(row['min_quality']) if row['min_quality'] else None,
                "max_quality": float(row['max_quality']) if row['max_quality'] else None
            }
        
        # Database bilgileri
        db_info_query = """
        SELECT 
            MIN(created_at) as oldest_record,
            MAX(created_at) as newest_record,
            COUNT(DISTINCT source) as unique_sources
        FROM baggage_policies
        """
        
        db_info = await db.fetchrow(db_info_query)
        
        database_info = {
            "oldest_record": db_info['oldest_record'].isoformat() if db_info['oldest_record'] else None,
            "newest_record": db_info['newest_record'].isoformat() if db_info['newest_record'] else None,
            "unique_sources": db_info['unique_sources'],
            "connection_pool": "Active"
        }
        
        return StatsResponse(
            total_policies=total_count,
            source_breakdown=source_breakdown,
            database_info=database_info
        )
        
    except Exception as e:
        logger.error(f"Stats hatası: {e}")
        raise HTTPException(status_code=500, detail=f"İstatistik hatası: {str(e)}")

# Health Check
# Health Check - UPDATE EXISTING FUNCTION
@app.get("/health")
async def health_check():
    """API ve veritabanı health check - Enhanced with OpenAI status"""
    
    try:
        if not db_pool:
            return {
                "status": "unhealthy",
                "database": "disconnected",
                "error": "Connection pool not available"
            }
        
        # Database connectivity test
        async with db_pool.acquire() as conn:
            test_result = await conn.fetchval("SELECT 1")
            policy_count = await conn.fetchval("SELECT COUNT(*) FROM baggage_policies")
        
        # ===== EXISTING RAG STATUS CHECKS =====
        vector_status = "available" if vector_ops else "unavailable"
        embedding_service_status = "available"
        try:
            embedding_service = get_embedding_service()
            embedding_service_status = "available"
        except:
            embedding_service_status = "unavailable"
        
        # ===== NEW: OpenAI STATUS CHECK =====
        openai_status = "unavailable"
        openai_model = "unknown"
        try:
            openai_service = get_openai_service()
            connection_test = openai_service.test_connection()
            if connection_test["success"]:
                openai_status = "available"
                openai_model = openai_service.default_model
            else:
                openai_status = f"error: {connection_test.get('message', 'unknown')}"
        except Exception as e:
            openai_status = f"error: {str(e)}"
        
        return {
            "status": "healthy",
            "database": "connected",
            "rag_features": {
                "vector_operations": vector_status,
                "embedding_service": embedding_service_status,
                "openai_service": openai_status,  # NEW
                "openai_model": openai_model      # NEW
            },
            "connection_pool": {
                "size": db_pool.get_size(),
                "max_size": db_pool.get_max_size(),
                "min_size": db_pool.get_min_size()
            },
            "data": {
                "total_policies": policy_count
            },
            "endpoints": {
                "traditional_search": "/search",
                "vector_search": "/vector/similarity-search",
                "openai_rag_chat": "/chat/openai",  # NEW
                "docs": "/docs"
            },
            "timestamp": "auto-generated"
        }
        
    except Exception as e:
        logger.error(f"Health check hatası: {e}")
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e)
        }

# Yeni Policy Ekleme (Bonus)
@app.post("/policies", response_model=dict)
async def add_policy(
    source: str,
    content: str,
    quality_score: Optional[float] = None,
    db = Depends(get_db_connection)
):
    """Yeni baggage policy ekle"""
    
    try:
        query = """
        INSERT INTO baggage_policies (source, content, quality_score, created_at)
        VALUES ($1, $2, $3, NOW())
        RETURNING id, created_at
        """
        
        result = await db.fetchrow(query, source, content, quality_score)
        
        return {
            "success": True,
            "message": "Policy başarıyla eklendi",
            "policy_id": result['id'],
            "created_at": result['created_at'].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Policy ekleme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Ekleme hatası: {str(e)}")
    
@app.get("/config/database")
async def get_database_config():
    """Database configuration görüntüle"""
    return {
        "host": settings.host,
        "port": settings.port,
        "database": settings.database,
        "user": settings.user,
        "pool_config": {
            "min_size": settings.min_pool_size,
            "max_size": settings.max_pool_size
        }
    }

# ===============================================
# ===== YENİ RAG VECTOR ENDPOINTS =====
# ===============================================

@app.get("/vector/similarity-search")
async def vector_similarity_search(
    q: str = Query(..., description="Search query", min_length=2),
    limit: int = Query(5, description="Maximum results", le=20),
    threshold: float = Query(0.3, description="Similarity threshold", ge=0.0, le=1.0),
    source: Optional[str] = Query(None, description="Source filter")
):
    """Vector similarity search endpoint"""
    
    if not vector_ops:
        raise HTTPException(status_code=503, detail="Vector operations not initialized")
    
    try:
        results = await vector_ops.similarity_search(
            query=q,
            limit=limit,
            similarity_threshold=threshold,
            source_filter=source
        )
        
        return {
            "success": True,
            "query": q,
            "method": "vector_similarity",
            "results": results,
            "count": len(results),
            "parameters": {
                "similarity_threshold": threshold,
                "source_filter": source
            }
        }
        
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search error: {str(e)}")

@app.post("/vector/embed-policies")
async def embed_existing_policies():
    """Mevcut policy'leri embed et (manual trigger)"""
    
    if not vector_ops:
        raise HTTPException(status_code=503, detail="Vector operations not initialized")
    
    try:
        embedded_count = await vector_ops.embed_existing_policies()
        
        return {
            "success": True,
            "message": f"Successfully embedded {embedded_count} policies",
            "embedded_count": embedded_count
        }
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")

@app.get("/vector/stats")
async def get_vector_stats():
    """Vector/embedding istatistikleri"""
    
    if not vector_ops:
        raise HTTPException(status_code=503, detail="Vector operations not initialized")
    
    try:
        stats = await vector_ops.get_embedding_stats()
        
        # Embedding service info
        embedding_service = get_embedding_service()
        
        return {
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_dimension": embedding_service.embedding_dimension,
            "database_stats": stats,
            "vector_search": "enabled"
        }
        
    except Exception as e:
        logger.error(f"Vector stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Vector stats error: {str(e)}")

@app.post("/chat/openai")
async def chat_with_openai_rag(
    question: str = Query(..., description="User question", min_length=3, max_length=500),
    max_results: int = Query(2, description="Max retrieved documents", le=5, ge=1),
    similarity_threshold: float = Query(0.5, description="Similarity threshold", ge=0.3, le=0.9),
    model: str = Query("gpt-3.5-turbo", description="OpenAI model to use")
):
    """RAG Chat with OpenAI - Production ready endpoint"""
    
    if not vector_ops:
        raise HTTPException(status_code=503, detail="Vector operations not available")
    
    try:
        # Step 1: Retrieve relevant documents using existing vector search
        logger.info(f"🔍 OpenAI RAG Query: '{question}' with model: {model}")
        
        retrieved_docs = await vector_ops.similarity_search(
            query=question,
            limit=max_results,
            similarity_threshold=similarity_threshold
        )
        
        logger.info(f"📊 Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Generate response using OpenAI
        openai_service = get_openai_service()
        
        # Handle both empty and populated context cases
        openai_response = openai_service.generate_rag_response(
            retrieved_docs, question, model
        )
        
        if not openai_response["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"OpenAI generation failed: {openai_response.get('error', 'Unknown error')}"
            )
        
        # Step 3: Prepare detailed response
        response_data = {
            "success": True,
            "question": question,
            "answer": openai_response["answer"],
            "model_used": openai_response["model_used"],
            "context_used": openai_response["context_used"],
            "retrieved_docs": [
                {
                    "source": doc["source"],
                    "content_preview": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                    "similarity_score": round(doc["similarity_score"], 3)
                }
                for doc in retrieved_docs
            ],
            "retrieval_stats": {
                "total_retrieved": len(retrieved_docs),
                "avg_similarity": round(
                    sum(doc["similarity_score"] for doc in retrieved_docs) / len(retrieved_docs), 3
                ) if retrieved_docs else 0,
                "min_similarity": round(min(doc["similarity_score"] for doc in retrieved_docs), 3) if retrieved_docs else 0,
                "max_similarity": round(max(doc["similarity_score"] for doc in retrieved_docs), 3) if retrieved_docs else 0,
                "context_quality": "high" if len(retrieved_docs) >= 2 and retrieved_docs[0]["similarity_score"] > 0.5 else "medium" if retrieved_docs else "low"
            },
            "usage_stats": openai_response.get("usage", {}),
            "timestamp": "auto-generated"
        }
        
        logger.info(f"✅ OpenAI RAG response generated successfully. Cost: ${openai_response.get('usage', {}).get('estimated_cost', 0):.4f}")
        
        def fix_float_values(obj):
            """Fix NaN and Infinity values for JSON"""
            if isinstance(obj, dict):
                return {k: fix_float_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [fix_float_values(item) for item in obj]
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return 0.0
                return obj
            return obj

        # Fix the response data
        response_data = fix_float_values(response_data)

        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OpenAI RAG chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "myapp:app",  # Import string olarak geç
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Development için auto-reload
        log_level="info"
    )