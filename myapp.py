# myapp.py - Temizlenmiş ve Düzeltilmiş Versiyon

from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncpg
import time
from contextlib import asynccontextmanager
import logging
from config import get_settings
from smart_embedding_service_fixed import SmartEmbeddingService

# OpenAI import'u - sadece gerektiğinde kullanılacak
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Settings
settings = get_settings()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
db_pool = None
embedding_service = SmartEmbeddingService(
    openai_api_key=settings.openai_api_key  # Optional, fallback if unavailable
)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, embedding_service
    
    logger.info("🚀 Turkish Airlines RAG API başlatılıyor...")
    try:
        # Database pool
        db_pool = await asyncpg.create_pool(
            **settings.get_asyncpg_params(),
            min_size=settings.db_min_pool_size,
            max_size=settings.db_max_pool_size,
            command_timeout=settings.db_command_timeout
        )
        logger.info("✅ PostgreSQL bağlantı havuzu oluşturuldu")
        
        # Initialize Smart embedding service with OpenAI support
        embedding_service = SmartEmbeddingService(
                openai_api_key=settings.openai_api_key
            )
        await embedding_service.initialize()
        logger.info("✅ Simple embedding service başlatıldı")
        
        # Database status check
        async with db_pool.acquire() as conn:
            total = await conn.fetchval("SELECT COUNT(*) FROM baggage_policies")
            embedded = await conn.fetchval("SELECT COUNT(*) FROM baggage_policies WHERE embedding IS NOT NULL")
            logger.info(f"📊 Database: {total} policies, {embedded} embedded")
            
    except Exception as e:
        logger.error(f"❌ Başlatma hatası: {e}")
        raise
    
    yield
    
    # Cleanup
    if embedding_service:
        await embedding_service.cleanup()
    if db_pool:
        await db_pool.close()
    logger.info("🔒 Servisler kapatıldı")

app = FastAPI(
    title="Turkish Airlines RAG API",
    description="AI-powered baggage policy search",
    version="4.0.0",
    lifespan=lifespan
)

# Pydantic Models
class BaggagePolicy(BaseModel):
    id: int
    source: str
    content: str
    created_at: Optional[str] = None
    quality_score: Optional[float] = None

class SemanticSearchResponse(BaseModel):
    success: bool
    query: str
    results: List[dict]
    found_count: int
    processing_time_ms: float

# Database Dependency
async def get_db_connection():
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database connection unavailable")
    
    try:
        async with db_pool.acquire() as connection:
            yield connection
    except Exception as e:
        logger.error(f"DB connection error: {e}")
        raise HTTPException(status_code=503, detail="Database connection error")

# Mock RAG Service
class MockRAGService:
    """Simple mock RAG service without OpenAI dependency"""
    
    def __init__(self):
        self.responses = {
            "laptop": "Laptops are allowed in carry-on baggage up to 8kg. Lithium batteries should stay in carry-on only.",
            "baggage": "Carry-on baggage limits: 8kg weight, 55x40x23cm dimensions for most airlines.",
            "liquid": "Liquids must be in 100ml containers or less, in a clear plastic bag.",
            "battery": "Lithium batteries must be in carry-on luggage. Power banks limited to 100Wh.",
            "electronics": "Electronic devices including laptops and tablets are allowed in carry-on baggage.",
            "weight": "Typical carry-on weight limit is 8kg for most airlines."
        }
    
    def get_answer(self, query: str, context_docs: List[Dict]) -> Dict:
        """Generate answer based on query keywords"""
        query_lower = query.lower()
        
        # Find best matching response
        answer = "I can help with baggage policies. Ask about laptops, liquids, weight limits, or electronics."
        
        for keyword, response in self.responses.items():
            if keyword in query_lower:
                answer = response
                break
        
        # Add source information if available
        if context_docs:
            sources = list(set([doc.get('source', 'Unknown') for doc in context_docs]))
            answer += f" (Based on {', '.join(sources)} policies)"
        
        return {
            "answer": answer,
            "sources": context_docs[:3],
            "confidence_score": 0.85,
            "provider": "Mock RAG Service"
        }

# CORE ENDPOINTS

@app.get("/")
async def root(db = Depends(get_db_connection)):
    """API status and statistics"""
    try:
        total = await db.fetchval("SELECT COUNT(*) FROM baggage_policies")
        embedded = await db.fetchval("SELECT COUNT(*) FROM baggage_policies WHERE embedding IS NOT NULL")
        
        return {
            "service": "Turkish Airlines RAG API",
            "version": "4.0.0",
            "status": "active",
            "statistics": {
                "total_policies": total,
                "embedded_policies": embedded
            },
            "endpoints": {
                "semantic_search": "/search/semantic?q=laptop",
                "traditional_search": "/search?q=laptop",
                "mock_rag": "/test/rag-mock?q=laptop",
                "real_rag": "/test/rag?q=laptop",
                "health": "/health"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint - optimized"""
    try:
        if not db_pool:
            return {"status": "unhealthy", "error": "No database connection"}
        
        # Sadece connection test, COUNT query'si yok
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
        return {
            "status": "healthy",
            "database": "connected",
            "embedding_service": "active" if embedding_service else "inactive"
            # total_policies satırını kaldır - bu COUNT query yavaş
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/search", response_model=List[BaggagePolicy])
async def traditional_search(
    q: str = Query(..., min_length=2),
    limit: int = Query(10, le=50),
    db = Depends(get_db_connection)
):
    """Traditional keyword search"""
    try:
        query = """
        SELECT id, source, content, created_at, quality_score 
        FROM baggage_policies 
        WHERE content ILIKE $1
        ORDER BY quality_score DESC NULLS LAST 
        LIMIT $2
        """
        
        rows = await db.fetch(query, f"%{q}%", limit)
        
        results = []
        for row in rows:
            policy = dict(row)
            if policy.get('created_at'):
                policy['created_at'] = policy['created_at'].isoformat()
            results.append(BaggagePolicy(**policy))
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/semantic", response_model=SemanticSearchResponse)
async def semantic_search(
    q: str = Query(..., min_length=2),
    limit: int = Query(5, le=20),
    threshold: float = Query(0.3, ge=0.0, le=1.0),
    db = Depends(get_db_connection)
):
    """Vector similarity search with pgvector"""
    start_time = time.time()
    
    try:
        if not embedding_service:
            raise HTTPException(status_code=503, detail="Embedding service not available")
        
        # Generate query embedding
        query_embedding = await embedding_service.create_embedding(q)
        
        # Convert to vector format string
        vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # pgvector similarity search using cosine distance
        similarity_query = """
        SELECT 
            id, source, content, quality_score, created_at,
            1 - (embedding <=> $1::vector) as similarity_score
        FROM baggage_policies 
        WHERE embedding IS NOT NULL
        AND 1 - (embedding <=> $1::vector) >= $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
        """
        
        rows = await db.fetch(similarity_query, vector_str, threshold, limit)
        
        # Format results
        results = [
            {
                "id": row['id'],
                "source": row['source'],
                "content": row['content'],
                "similarity_score": round(float(row['similarity_score']), 3),
                "quality_score": float(row['quality_score']) if row['quality_score'] else None
            }
            for row in rows
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return SemanticSearchResponse(
            success=True,
            query=q,
            results=results,
            found_count=len(results),
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# RAG ENDPOINTS

@app.get("/test/rag-mock")
async def mock_rag(
    q: str = Query(..., min_length=2),
    db = Depends(get_db_connection)
):
    """Mock RAG with pgvector similarity search"""
    try:
        context_docs = []
        
        if embedding_service:
            try:
                query_embedding = await embedding_service.create_embedding(q)
                vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
                
                # pgvector similarity search
                similarity_query = """
                SELECT 
                    id, source, content,
                    1 - (embedding <=> $1::vector) as similarity_score
                FROM baggage_policies 
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> $1::vector) >= 0.3
                ORDER BY embedding <=> $1::vector
                LIMIT 3
                """
                
                rows = await db.fetch(similarity_query, vector_str)
                
                context_docs = [
                    {
                        "id": row['id'],
                        "source": row['source'],
                        "content": row['content'][:200] + "...",
                        "similarity_score": round(float(row['similarity_score']), 3)
                    }
                    for row in rows
                ]
                    
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        
        # Generate mock answer
        mock_service = MockRAGService()
        result = mock_service.get_answer(q, context_docs)
        
        return {
            "success": True,
            "query": q,
            **result
        }
        
    except Exception as e:
        logger.error(f"Mock RAG error: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": q
        }

@app.get("/test/rag")
async def real_rag(
    q: str = Query(..., min_length=2),
    db = Depends(get_db_connection)
):
    """Real RAG with OpenAI GPT (if API key available)"""
    try:
        # Check OpenAI availability
        if not OPENAI_AVAILABLE or not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
            return {
                "success": False,
                "error": "OpenAI not configured",
                "suggestion": f"Try mock version: /test/rag-mock?q={q}",
                "query": q
            }
        
        # Get relevant documents using pgvector
        if not embedding_service:
            raise HTTPException(status_code=503, detail="Embedding service not available")
        
        query_embedding = await embedding_service.create_embedding(q)
        vector_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        similarity_query = """
        SELECT 
            id, source, content,
            1 - (embedding <=> $1::vector) as similarity_score
        FROM baggage_policies 
        WHERE embedding IS NOT NULL
        AND 1 - (embedding <=> $1::vector) >= 0.3
        ORDER BY embedding <=> $1::vector
        LIMIT 3
        """
        
        rows = await db.fetch(similarity_query, vector_str)
        
        if not rows:
            return {
                "success": False,
                "error": "No relevant documents found",
                "suggestion": "Try generating embeddings first",
                "query": q
            }
        
        # Format context for OpenAI
        context = "\n".join([f"Source: {row['source']}\n{row['content']}" for row in rows])
        
        # Call OpenAI API
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful travel assistant. Answer questions based on provided airline policies. Be specific and helpful."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer based on the context above:"
                }
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content
        
        sources = [
            {
                "id": row['id'],
                "source": row['source'],
                "excerpt": row['content'][:200] + "...",
                "similarity_score": round(float(row['similarity_score']), 3)
            }
            for row in rows
        ]
        
        return {
            "success": True,
            "query": q,
            "answer": answer,
            "sources": sources,
            "provider": "OpenAI GPT-3.5-turbo"
        }
        
    except Exception as e:
        logger.error(f"Real RAG error: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": q
        }

# UTILITY ENDPOINTS

@app.get("/test/embedding")
async def test_embedding():
    """Test embedding service"""
    try:
        if not embedding_service:
            return {"status": "error", "message": "Embedding service not available"}
        
        test_text = "Can I bring my laptop on the plane?"
        start_time = time.time()
        embedding = await embedding_service.create_embedding(test_text)
        elapsed = (time.time() - start_time) * 1000
        
        return {
            "status": "success",
            "test_text": test_text,
            "dimension": len(embedding),
            "processing_time_ms": round(elapsed, 2),
            "sample_values": embedding[:3]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/embedding/stats")
async def embedding_stats():
    """Get embedding service statistics"""
    if embedding_service:
        return embedding_service.get_stats()
    return {"error": "Embedding service not available"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("myapp:app", host="0.0.0.0", port=8000, reload=True)