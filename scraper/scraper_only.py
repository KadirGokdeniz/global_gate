#!/usr/bin/env python3
"""
Updated scraper_only.py - Multi-airline data collection and embedding
Container başlatılır, tüm airline'ları scrape eder, embed eder, temiz bir şekilde kapanır
"""

import os
import sys
import asyncio
import logging
from web_scraper import (
    setup_database, 
    clear_old_data, 
    get_database_stats,
    scrape_all_airlines,
    scrape_specific_airline,
    get_supported_airlines,
    get_detailed_stats,
    print_airline_summary
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
    """Enhanced startup banner"""
    print("=" * 80)
    print("🚀 MULTI-AIRLINE DATA SCRAPER - CONTAINER STARTUP")
    print("=" * 80)
    print("📋 Mission: Scrape → Process → Embed → Exit")
    print("🎯 Targets: Turkish Airlines + Pegasus Airlines")
    print("🧠 AI: Multilingual embedding generation")
    print("📊 Features: Multi-airline support, enhanced deduplication")
    print("=" * 80)

def get_scraping_strategy():
    """Determine which airlines to scrape based on environment"""
    
    # Environment variable kontrolü
    airline_selection = os.getenv('SCRAPE_AIRLINES', 'all').lower()
    
    if airline_selection == 'thy_only':
        return ['turkish_airlines'], "Turkish Airlines Only"
    elif airline_selection == 'pegasus_only':
        return ['pegasus'], "Pegasus Only"
    elif airline_selection == 'all':
        return get_supported_airlines(), "All Airlines"
    else:
        # Custom selection
        airlines = airline_selection.split(',')
        airlines = [a.strip() for a in airlines if a.strip()]
        
        # Validate
        supported = get_supported_airlines()
        valid_airlines = [a for a in airlines if a in supported]
        
        if valid_airlines:
            return valid_airlines, f"Custom: {', '.join(valid_airlines)}"
        else:
            logger.warning(f"⚠️ Invalid airline selection: {airline_selection}")
            logger.info(f"✅ Falling back to all airlines")
            return get_supported_airlines(), "All Airlines (fallback)"

async def wait_for_database(max_retries=30, delay=2):
    """Database'in hazır olmasını bekle - geliştirilmiş"""
    print("⏳ Waiting for database to be ready...")
    
    for attempt in range(max_retries):
        try:
            conn = await asyncpg.connect(**DB_CONFIG)
            
            # Database ve schema kontrolü
            await conn.fetchval("SELECT 1")
            
            # Airline column kontrolü (yeni schema)
            result = await conn.fetch("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'baggage_policies' AND column_name = 'airline'
            """)
            
            await conn.close()
            
            if result:
                print(f"✅ Database ready with multi-airline schema after {attempt + 1} attempts")
                return True
            else:
                print(f"⚠️ Database ready but schema needs update (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    print("❌ Schema update required - check init.sql")
                    return False
                
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"🔄 Attempt {attempt + 1}/{max_retries} failed, retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                print(f"❌ Database connection failed after {max_retries} attempts: {e}")
                return False
    
    return False

async def run_embedding_process():
    """Enhanced embedding sürecini çalıştır"""
    print("\n🧠 MULTI-AIRLINE EMBEDDING PROCESS STARTING")
    print("-" * 60)
    
    pool = None
    try:
        # Connection pool oluştur
        pool = await asyncpg.create_pool(
            **DB_CONFIG,
            min_size=2,
            max_size=5,
            command_timeout=120  # Longer timeout for embedding
        )
        
        print("🔗 Database pool created for embedding")
        
        # Vector operations initialize
        vector_ops = VectorOperations(pool)
        
        # Check existing embeddings
        stats_before = await vector_ops.get_embedding_stats()
        print(f"📊 Pre-embedding stats: {stats_before}")
        
        # Embedding işlemini başlat
        embedded_count = await vector_ops.embed_existing_policies(batch_size=32)
        
        if embedded_count > 0:
            print(f"🎉 Successfully embedded {embedded_count} new policies!")
        else:
            print("ℹ️ All policies already had embeddings")
        
        # Final embedding stats
        stats_after = await vector_ops.get_embedding_stats()
        print(f"\n📊 FINAL EMBEDDING STATS:")
        print(f"  Total policies: {stats_after.get('total_policies', 0)}")
        print(f"  Embedded: {stats_after.get('embedded_policies', 0)}")
        print(f"  Missing: {stats_after.get('missing_embeddings', 0)}")
        print(f"  Coverage: {stats_after.get('embedding_coverage_percent', 0)}%")
        
        return embedded_count
        
    except Exception as e:
        print(f"❌ Embedding process failed: {e}")
        return 0
    finally:
        if pool:
            await pool.close()
            print("🔌 Embedding database pool closed")

def main():
    """Enhanced ana scraper fonksiyonu"""
    
    print_banner()
    
    # Exit codes
    EXIT_SUCCESS = 0
    EXIT_DB_ERROR = 1
    EXIT_SCRAPING_ERROR = 2
    EXIT_EMBEDDING_ERROR = 3
    EXIT_SCHEMA_ERROR = 4
    
    try:
        # Step 0: Determine scraping strategy
        print("\n🎯 STEP 0: SCRAPING STRATEGY")
        print("-" * 60)
        
        selected_airlines, strategy_description = get_scraping_strategy()
        print(f"📋 Strategy: {strategy_description}")
        print(f"✈️ Airlines to scrape: {', '.join(selected_airlines)}")
        
        print_airline_summary()
        
        # Step 1: Database hazırlığını bekle
        print("\n📊 STEP 1: DATABASE PREPARATION")
        print("-" * 60)
        
        # Async wait
        db_ready = asyncio.run(wait_for_database())
        if not db_ready:
            print("❌ Database preparation failed")
            sys.exit(EXIT_DB_ERROR)
        
        # Database setup
        if not setup_database():
            print("❌ Database setup failed")
            sys.exit(EXIT_SCHEMA_ERROR)
        
        # Check existing data
        existing_stats = get_database_stats()
        total_existing = existing_stats.get('total_policies', 0)
        
        if total_existing > 0:
            print(f"🗃️ Found {total_existing} existing policies")
            print(get_detailed_stats())
            print("🧹 Clearing old data for fresh scraping...")
            clear_old_data()
        
        print("✅ Database prepared successfully")
        
        # Step 2: Multi-airline web scraping
        print(f"\n🕷️ STEP 2: MULTI-AIRLINE WEB SCRAPING")
        print("-" * 60)
        
        scraping_results = {}
        total_scraped = 0
        
        for airline_id in selected_airlines:
            print(f"\n🚀 Scraping {airline_id.upper()}...")
            
            try:
                count = scrape_specific_airline(airline_id)
                scraping_results[airline_id] = count
                total_scraped += count
                
                if count > 0:
                    print(f"✅ {airline_id}: {count} policies scraped")
                else:
                    print(f"⚠️ {airline_id}: No data scraped")
                    
            except Exception as e:
                print(f"❌ {airline_id} scraping error: {e}")
                scraping_results[airline_id] = 0
        
        print(f"\n📊 SCRAPING SUMMARY:")
        print(f"  Total policies: {total_scraped}")
        print(f"  Results: {scraping_results}")
        
        if total_scraped == 0:
            print("❌ No data was scraped from any airline")
            sys.exit(EXIT_SCRAPING_ERROR)
        
        print(f"✅ Successfully scraped {total_scraped} policies from {len([k for k, v in scraping_results.items() if v > 0])} airlines")
        
        # Step 3: AI Embedding generation
        print(f"\n🧠 STEP 3: AI EMBEDDING GENERATION")
        print("-" * 60)
        
        embedded_count = asyncio.run(run_embedding_process())
        
        if embedded_count == 0:
            print("⚠️ No new embeddings generated (might be already embedded)")
        else:
            print(f"✅ Successfully generated {embedded_count} embeddings")
        
        # Step 4: Final verification and reporting
        print(f"\n🔍 STEP 4: FINAL VERIFICATION")
        print("-" * 60)
        
        final_stats = get_database_stats()
        
        print(f"📊 CONTAINER MISSION COMPLETE:")
        print(f"  📝 Total policies: {final_stats.get('total_policies', 0)}")
        print(f"  ✈️ Airlines: {final_stats.get('total_airlines', 0)}")
        print(f"  📂 Sources: {final_stats.get('total_sources', 0)}")
        print(f"  ⭐ Avg quality: {final_stats.get('avg_quality_score', 0):.2f}")
        
        # Detailed breakdown
        print(get_detailed_stats())
        
        print(f"\n🎯 CONTAINER READY TO EXIT")
        print("✅ FastAPI can now start with multi-airline data")
        print("🚀 RAG system ready for Turkish Airlines + Pegasus queries")
        print("=" * 80)
        
        sys.exit(EXIT_SUCCESS)
        
    except KeyboardInterrupt:
        print("\n⚠️ Container interrupted by user")
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        print(f"\n❌ Unexpected error in multi-airline scraper: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(EXIT_EMBEDDING_ERROR)

if __name__ == "__main__":
    main()