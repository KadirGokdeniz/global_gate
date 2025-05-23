# Gerekli importlar
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel
from typing import List, Optional
import asyncpg
import os
from contextlib import asynccontextmanager
import logging
from pydantic import BaseModel, Field
from functools import lru_cache
from pydantic_settings import BaseSettings

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
    
    # 👈 EKLENDİ: SSL and advanced settings
    ssl: bool = Field(default=False, description="Enable SSL connection")
    echo: bool = Field(default=False, description="Echo SQL queries (debug)")

    openai_api_key: str
    
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

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlatma ve kapatma işlemleri"""
    global db_pool
    
    # Startup
    logger.info("🚀 Turkish Airlines API başlatılıyor...")
    try:
        db_pool = await asyncpg.create_pool(
            **settings.get_asyncpg_params(),
            min_size=settings.min_pool_size,
            max_size=settings.max_pool_size,
            command_timeout=settings.command_timeout
        )
        logger.info("✅ PostgreSQL bağlantı havuzu oluşturuldu")
        
        # Test bağlantısı
        async with db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM baggage_policies")
            logger.info(f"📊 Veritabanında {result} policy bulundu")
            
    except Exception as e:
        logger.error(f"❌ Veritabanı bağlantı hatası: {e}")
        raise
    
    yield
    
    # Shutdown
    if db_pool:
        await db_pool.close()
        logger.info("🔒 Veritabanı bağlantıları kapatıldı")

# FastAPI instance
app = FastAPI(
    title="Turkish Airlines Baggage Policy API",
    description="Turkish Airlines baggage kuralları - PostgreSQL API",
    version="3.0.0",
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
    """Veritabanı bağlantısı dependency"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Veritabanı bağlantısı mevcut değil")
    
    try:
        async with db_pool.acquire() as connection:
            yield connection
    except Exception as e:
        logger.error(f"DB bağlantı hatası: {e}")
        raise HTTPException(status_code=503, detail="Veritabanı bağlantı hatası")

# Ana Sayfa
@app.get("/")
async def read_root(db = Depends(get_db_connection)):
    """API ana sayfa ve genel bilgiler"""
    try:
        # Toplam policy sayısı
        total_count = await db.fetchval("SELECT COUNT(*) FROM baggage_policies")
        
        # Mevcut kaynaklar
        sources = await db.fetch("SELECT source, COUNT(*) as count FROM baggage_policies GROUP BY source")
        
        return {
            "service": "Turkish Airlines Baggage Policy API",
            "version": "3.0.0",
            "data_source": "PostgreSQL Database",
            "status": "active",
            "endpoints": {
                "all_policies": "/policies",
                "search": "/search?q=yoursearchterm",
                "by_source": "/policies/{source_name}",
                "stats": "/stats",
                "health": "/health",
                "documentation": "/docs"
            },
            "statistics": {
                "total_policies": total_count,
                "available_sources": [dict(row) for row in sources]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ana sayfa yükleme hatası: {str(e)}")

# Tüm Policies
@app.get("/policies", response_model=ApiResponse)
async def get_policies(
    limit: int = Query(50, description="Maksimum döndürülecek policy sayısı", le=500),
    offset: int = Query(0, description="Başlangıç pozisyonu", ge=0),
    source: Optional[str] = Query(None, description="Kaynak filtresi"),
    min_quality: Optional[float] = Query(None, description="Minimum kalite skoru"),
    db = Depends(get_db_connection)
):
    """Tüm baggage policies'leri döndür"""
    
    try:
        # Base query
        query = "SELECT id, source, content, created_at, quality_score FROM baggage_policies WHERE 1=1"
        params = []
        param_count = 0
        
        # Filtreleme
        if source:
            param_count += 1
            query += f" AND source = ${param_count}"
            params.append(source)
        
        if min_quality is not None:
            param_count += 1
            query += f" AND quality_score >= ${param_count}"
            params.append(min_quality)
        
        # Sıralama ve limit
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
            message=f"Policies başarıyla getirildi (PostgreSQL)",
            data=policies,
            count=len(policies)
        )
        
    except Exception as e:
        logger.error(f"Policies getirme hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Veri getirme hatası: {str(e)}")

# Arama
@app.get("/search", response_model=SearchResponse)
async def search_policies(
    q: str = Query(..., description="Arama terimi", min_length=2),
    limit: int = Query(10, description="Maksimum sonuç sayısı", le=100),
    source: Optional[str] = Query(None, description="Kaynak filtresi"),
    db = Depends(get_db_connection)
):
    """Gelişmiş metin araması (PostgreSQL ILIKE)"""
    
    try:
        # Toplam kayıt sayısı
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
@app.get("/health")
async def health_check():
    """API ve veritabanı health check"""
    
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
        
        return {
            "status": "healthy",
            "database": "connected",
            "connection_pool": {
                "size": db_pool.get_size(),
                "max_size": db_pool.get_max_size(),
                "min_size": db_pool.get_min_size()
            },
            "data": {
                "total_policies": policy_count
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "myapp:app",  # Import string olarak geç
        host="0.0.0.0", 
        port=8000,
        reload=True,  # Development için auto-reload
        log_level="info"
    )