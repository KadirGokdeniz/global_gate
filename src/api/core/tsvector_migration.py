"""
Migration: BM25/full-text search desteği için content_tsv kolonu ve index.

Startup'ta otomatik çalışır, idempotent (tekrar çalışsa sorun çıkarmaz).
Schema değişikliği gerektirir ama script kendisi halleder.

Neden 'simple' config?
  - Belgeler karışık dilli (TR + EN)
  - 'turkish' veya 'english' stemmer seçmek bir dili bozuyor
  - 'simple' sadece lowercase + punctuation, dil bağımsız çalışır

Çalışma süresi:
  - İlk çalıştırma: ~30s (2722 satır için tsvector üretimi + GIN index)
  - Sonraki çalıştırmalar: <1s (her şey var, atlanır)
"""
import logging
import asyncpg

logger = logging.getLogger(__name__)


async def ensure_tsvector_support(db_pool) -> dict:
    """
    content_tsv kolonu ve GIN index'inin var olduğunu garanti et.
    
    Returns:
        dict: {
            "column_added": bool,   # Bu çalışmada kolon eklendi mi
            "index_added": bool,    # Bu çalışmada index eklendi mi
            "row_count": int,       # tsvector dolu satır sayısı
            "skipped": bool         # Zaten varmış, iş yapılmadı mı
        }
    """
    result = {
        "column_added": False,
        "index_added": False,
        "row_count": 0,
        "skipped": False,
    }
    
    async with db_pool.acquire() as conn:
        # ─────────────────────────────────────────────────────────────
        # 1. content_tsv kolonu var mı?
        # ─────────────────────────────────────────────────────────────
        column_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'policy'
                  AND column_name = 'content_tsv'
            )
        """)
        
        if not column_exists:
            logger.info("🔨 Migration: content_tsv kolonu ekleniyor...")
            # GENERATED ALWAYS AS STORED = content değiştikçe otomatik güncellenir
            # simple config = dil-bağımsız (karışık dil için en güvenli)
            await conn.execute("""
                ALTER TABLE policy 
                ADD COLUMN content_tsv tsvector 
                GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED
            """)
            result["column_added"] = True
            logger.info("✅ content_tsv kolonu eklendi")
        else:
            logger.debug("content_tsv kolonu zaten var")
        
        # ─────────────────────────────────────────────────────────────
        # 2. GIN index var mı?
        # ─────────────────────────────────────────────────────────────
        index_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM pg_indexes
                WHERE tablename = 'policy'
                  AND indexname = 'idx_policy_content_tsv'
            )
        """)
        
        if not index_exists:
            logger.info("🔨 Migration: GIN index oluşturuluyor (30-60sn sürebilir)...")
            # CONCURRENTLY kullanmıyoruz çünkü bir transaction içinde DDL var,
            # locking önemli değil - startup sırasında kimse query atmıyor
            await conn.execute("""
                CREATE INDEX idx_policy_content_tsv 
                ON policy USING GIN(content_tsv)
            """)
            result["index_added"] = True
            logger.info("✅ GIN index oluşturuldu")
        else:
            logger.debug("GIN index zaten var")
        
        # ─────────────────────────────────────────────────────────────
        # 3. Durum raporu
        # ─────────────────────────────────────────────────────────────
        row_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM policy 
            WHERE content_tsv IS NOT NULL
        """)
        result["row_count"] = row_count
        
        total_count = await conn.fetchval("SELECT COUNT(*) FROM policy")
        
        if result["column_added"] or result["index_added"]:
            logger.info(
                f"✅ tsvector migration tamamlandı: "
                f"{row_count}/{total_count} satır indexlendi"
            )
        else:
            result["skipped"] = True
            logger.info(
                f"✅ tsvector hazır (migration atlandı): "
                f"{row_count}/{total_count} satır"
            )
    
    return result


async def check_tsvector_health(db_pool) -> dict:
    """
    Full-text search sağlık kontrolü - debug için.
    
    Returns:
        dict: tsvector istatistikleri
    """
    async with db_pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM policy")
        indexed = await conn.fetchval(
            "SELECT COUNT(*) FROM policy WHERE content_tsv IS NOT NULL"
        )
        
        # Örnek query - "PETC" gibi özel bir token arayalım
        sample = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM policy 
            WHERE content_tsv @@ to_tsquery('simple', 'PETC')
        """)
        
        return {
            "total_rows": total,
            "indexed_rows": indexed,
            "coverage_pct": round(indexed / total * 100, 1) if total else 0,
            "sample_petc_matches": sample,
        }