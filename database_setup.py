# database_setup.py (yeni dosya oluştur)
import asyncio
import asyncpg
from config import get_settings

async def setup_pgvector():
    """pgvector extension'ını kur ve test et"""
    settings = get_settings()
    
    try:
        conn = await asyncpg.connect(**settings.get_asyncpg_params())
        
        # Extension'ı kur
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✅ pgvector extension kuruldu")
        
        # Test et
        result = await conn.fetchval(
            "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"
        )
        
        if result > 0:
            print("✅ pgvector aktif ve çalışıyor")
        else:
            print("❌ pgvector kurulumu başarısız")
            
        await conn.close()
        
    except Exception as e:
        print(f"❌ Setup hatası: {e}")

if __name__ == "__main__":
    asyncio.run(setup_pgvector())