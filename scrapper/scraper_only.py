#!/usr/bin/env python3
"""
scraper_only.py - Sadece veri toplama ve embedding
Container başlatılır, işini yapar, temiz bir şekilde kapanır
"""

import os
import sys
import asyncio
import logging
from web_scrapper import (
    setup_database, 
    clear_old_data, 
    scrape_all_turkish_airlines,
    get_database_stats
)
from vector_operations import VectorOperations
import asyncpg

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'db'),
    'database': os.getenv('DB_DATABASE', 'global_gate'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'qeqe')
}

def print_banner():
    """Startup banner"""
    print("=" * 70)
    print("🚀 TURKISH AIRLINES DATA SCRAPER - STANDALONE")
    print("=" * 70)
    print("📋 Mission: Scrape → Process → Embed → Exit")
    print("🎯 Target: Turkish Airlines baggage policies")
    print("🧠 AI: Multilingual embedding generation")
    print("=" * 70)

async def wait_for_database(max_retries=30, delay=2):
    """Database'in hazır olmasını bekle"""
    print("⏳ Waiting for database to be ready...")
    
    for attempt in range(max_retries):
        try:
            conn = await asyncpg.connect(**DB_CONFIG)
            await conn.fetchval("SELECT 1")
            await conn.close()
            print(f"✅ Database ready after {attempt + 1} attempts")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"🔄 Attempt {attempt + 1}/{max_retries} failed, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                print(f"❌ Database connection failed after {max_retries} attempts: {e}")
                return False
    
    return False

async def run_embedding_process():
    """Embedding sürecini çalıştır"""
    print("\n🧠 EMBEDDING PROCESS STARTING")
    print("-" * 50)
    
    pool = None
    try:
        # Connection pool oluştur
        pool = await asyncpg.create_pool(
            **DB_CONFIG,
            min_size=1,
            max_size=3,
            command_timeout=60
        )
        
        print("🔗 Database pool created")
        
        # Vector operations initialize
        vector_ops = VectorOperations(pool)
        
        # Embedding işlemini başlat
        embedded_count = await vector_ops.embed_existing_policies()
        
        if embedded_count > 0:
            print(f"🎉 Successfully embedded {embedded_count} policies!")
        else:
            print("ℹ️ All policies already had embeddings")
        
        # Final stats
        stats = await vector_ops.get_embedding_stats()
        print(f"\n📊 FINAL EMBEDDING STATS:")
        print(f"  Total policies: {stats.get('total_policies', 0)}")
        print(f"  Embedded: {stats.get('embedded_policies', 0)}")
        print(f"  Coverage: {stats.get('embedding_coverage_percent', 0)}%")
        
        return embedded_count
        
    except Exception as e:
        print(f"❌ Embedding process failed: {e}")
        return 0
    finally:
        if pool:
            await pool.close()
            print("🔌 Database pool closed")

def main():
    """Ana scraper fonksiyonu"""
    
    print_banner()
    
    # Exit codes
    EXIT_SUCCESS = 0
    EXIT_DB_ERROR = 1
    EXIT_SCRAPING_ERROR = 2
    EXIT_EMBEDDING_ERROR = 3
    
    try:
        # Step 1: Database hazırlığını bekle
        print("\n📊 STEP 1: DATABASE PREPARATION")
        print("-" * 50)
        
        # Async wait
        db_ready = asyncio.run(wait_for_database())
        if not db_ready:
            print("❌ Database preparation failed")
            sys.exit(EXIT_DB_ERROR)
        
        # Database setup
        if not setup_database():
            print("❌ Database setup failed")
            sys.exit(EXIT_DB_ERROR)
        
        # Check existing data
        existing_stats = get_database_stats()
        total_existing = existing_stats.get('total_policies', 0)
        
        if total_existing > 0:
            print(f"🗃️ Found {total_existing} existing policies")
            print("🧹 Clearing old data for fresh scraping...")
            clear_old_data()
        
        print("✅ Database prepared successfully")
        
        # Step 2: Web scraping
        print("\n🕷️ STEP 2: WEB SCRAPING")
        print("-" * 50)
        
        scraped_count = scrape_all_turkish_airlines()
        
        if scraped_count == 0:
            print("❌ No data was scraped")
            sys.exit(EXIT_SCRAPING_ERROR)
        
        print(f"✅ Successfully scraped {scraped_count} policies")
        
        # Step 3: Embedding generation
        print("\n🧠 STEP 3: AI EMBEDDING GENERATION")
        print("-" * 50)
        
        embedded_count = asyncio.run(run_embedding_process())
        
        if embedded_count == 0:
            print("⚠️ No new embeddings generated (might be already embedded)")
        else:
            print(f"✅ Successfully generated {embedded_count} embeddings")
        
        # Step 4: Final verification
        print("\n🔍 STEP 4: FINAL VERIFICATION")
        print("-" * 50)
        
        final_stats = get_database_stats()
        
        print(f"📊 SCRAPER MISSION COMPLETE:")
        print(f"  📝 Total policies: {final_stats.get('total_policies', 0)}")
        print(f"  📂 Sources: {final_stats.get('total_sources', 0)}")
        print(f"  ⭐ Avg quality: {final_stats.get('avg_quality_score', 0):.2f}")
        
        if final_stats.get('source_breakdown'):
            print(f"\n📋 SOURCE BREAKDOWN:")
            for source, info in final_stats['source_breakdown'].items():
                print(f"  - {source}: {info['count']} policies (quality: {info['avg_quality']})")
        
        print("\n🎯 SCRAPER CONTAINER READY TO EXIT")
        print("✅ FastAPI can now start with prepared data")
        print("=" * 70)
        
        sys.exit(EXIT_SUCCESS)
        
    except KeyboardInterrupt:
        print("\n⚠️ Scraper interrupted by user")
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        print(f"\n❌ Unexpected error in scraper: {e}")
        sys.exit(EXIT_EMBEDDING_ERROR)

if __name__ == "__main__":
    main()