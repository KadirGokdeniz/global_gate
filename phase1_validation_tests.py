#!/usr/bin/env python3
"""
Phase 1 RAG Foundation Validation Tests
Yeni implement edilen özellikleri test eder
"""

import requests
import json
import time
import asyncio
import asyncpg
import os

BASE_URL = "http://localhost:8000"

# Test configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'database': os.getenv('DB_DATABASE', 'global_gate'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'qeqe')
}

def test_api_health():
    """1. API health check"""
    print("🔍 1. API Health Check...")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("  ✅ API is running")
            print(f"  📊 Database status: {data.get('database', 'unknown')}")
            return True
        else:
            print(f"  ❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"  ❌ API connection error: {e}")
        return False

def test_vector_stats():
    """2. Vector stats endpoint test"""
    print("\n🔍 2. Vector Stats Endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/vector/stats", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("  ✅ Vector stats endpoint working")
            print(f"  🧠 Embedding model: {data.get('embedding_model', 'unknown')}")
            print(f"  📏 Embedding dimension: {data.get('embedding_dimension', 'unknown')}")
            
            db_stats = data.get('database_stats', {})
            total = db_stats.get('total_policies', 0)
            embedded = db_stats.get('embedded_policies', 0)
            coverage = db_stats.get('embedding_coverage_percent', 0)
            
            print(f"  📊 Policy coverage: {embedded}/{total} ({coverage}%)")
            
            if coverage >= 100:
                print("  🎉 All policies have embeddings!")
            elif coverage >= 50:
                print("  ✅ Good embedding coverage")
            else:
                print("  ⚠️  Low embedding coverage - may need to run embed-policies")
            
            return True, data
        else:
            print(f"  ❌ Vector stats failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False, None
    except Exception as e:
        print(f"  ❌ Vector stats error: {e}")
        return False, None

def test_embed_policies():
    """3. Embed policies test (trigger manual embedding)"""
    print("\n🔍 3. Embedding Policies...")
    
    try:
        print("  🔄 Triggering policy embedding...")
        response = requests.post(f"{BASE_URL}/vector/embed-policies", timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            embedded_count = data.get('embedded_count', 0)
            
            if embedded_count > 0:
                print(f"  ✅ Successfully embedded {embedded_count} new policies")
            else:
                print("  ✅ All policies already embedded")
            
            return True
        else:
            print(f"  ❌ Embed policies failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"  ❌ Embed policies error: {e}")
        return False

def test_vector_similarity_search():
    """4. Vector similarity search test"""
    print("\n🔍 4. Vector Similarity Search...")
    
    test_queries = [
        "baggage weight limit",
        "sports equipment policy", 
        "excess baggage fees",
        "Turkish Airlines carry-on rules",
        "pets travel policy"
    ]
    
    all_passed = True
    
    for i, query in enumerate(test_queries):
        print(f"\n  [{i+1}] Testing query: '{query}'")
        
        try:
            params = {
                'q': query,
                'limit': 3,
                'threshold': 0.2
            }
            
            response = requests.get(f"{BASE_URL}/vector/similarity-search", 
                                  params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                print(f"    ✅ Found {len(results)} results")
                
                # Show top result
                if results:
                    top_result = results[0]
                    similarity = top_result.get('similarity_score', 0)
                    content_preview = top_result.get('content', '')[:60] + "..."
                    
                    print(f"    🥇 Best match (similarity: {similarity:.3f})")
                    print(f"       Source: {top_result.get('source', 'unknown')}")
                    print(f"       Content: {content_preview}")
                    
                    if similarity >= 0.5:
                        print("    🎯 High relevance match!")
                    elif similarity >= 0.3:
                        print("    ✅ Good match")
                    else:
                        print("    ⚠️  Low similarity score")
                else:
                    print("    ⚠️  No results found")
                    all_passed = False
                    
            else:
                print(f"    ❌ Search failed: {response.status_code}")
                all_passed = False
                
        except Exception as e:
            print(f"    ❌ Search error: {e}")
            all_passed = False
    
    return all_passed

def test_comparison_with_regular_search():
    """5. Vector vs Regular search comparison"""
    print("\n🔍 5. Vector vs Regular Search Comparison...")
    
    test_query = "baggage weight limit"
    
    try:
        # Regular text search
        print("  📝 Regular text search...")
        regular_response = requests.get(f"{BASE_URL}/search", 
                                      params={'q': test_query, 'limit': 3},
                                      timeout=15)
        
        # Vector search  
        print("  🧠 Vector similarity search...")
        vector_response = requests.get(f"{BASE_URL}/vector/similarity-search",
                                     params={'q': test_query, 'limit': 3, 'threshold': 0.2},
                                     timeout=15)
        
        if regular_response.status_code == 200 and vector_response.status_code == 200:
            regular_data = regular_response.json()
            vector_data = vector_response.json()
            
            regular_count = len(regular_data.get('results', []))
            vector_count = len(vector_data.get('results', []))
            
            print(f"  📊 Regular search: {regular_count} results")
            print(f"  📊 Vector search: {vector_count} results")
            
            if vector_count > 0:
                print("  ✅ Vector search is working!")
                
                # Show semantic understanding
                if vector_count > 0:
                    top_vector = vector_data['results'][0]
                    similarity = top_vector.get('similarity_score', 0)
                    print(f"  🎯 Top vector result similarity: {similarity:.3f}")
                
                return True
            else:
                print("  ⚠️  Vector search returned no results")
                return False
        else:
            print("  ❌ Comparison test failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Comparison test error: {e}")
        return False

async def test_database_vector_column():
    """6. Database vector column validation"""
    print("\n🔍 6. Database Vector Column Check...")
    
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Check if vector column exists
        column_check = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_name = 'baggage_policies' 
                AND column_name = 'embedding'
            )
        """)
        
        if column_check:
            print("  ✅ Vector column exists in database")
            
            # Check vector index
            index_check = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE tablename = 'baggage_policies' 
                    AND indexname LIKE '%embedding%'
                )
            """)
            
            if index_check:
                print("  ✅ Vector index exists")
            else:
                print("  ⚠️  Vector index not found")
            
            # Check embedding data
            embedding_sample = await conn.fetchrow("""
                SELECT id, array_length(embedding, 1) as embedding_dim
                FROM baggage_policies 
                WHERE embedding IS NOT NULL 
                LIMIT 1
            """)
            
            if embedding_sample:
                dim = embedding_sample['embedding_dim']
                print(f"  ✅ Embeddings found with dimension: {dim}")
                
                if dim == 384:
                    print("  🎯 Correct embedding dimension!")
                else:
                    print(f"  ⚠️  Unexpected embedding dimension: {dim}")
            else:
                print("  ⚠️  No embeddings found in database")
            
        else:
            print("  ❌ Vector column not found in database!")
            return False
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Database check error: {e}")
        return False

def main():
    """Ana test suite"""
    print("=" * 70)
    print("🚀 PHASE 1 RAG FOUNDATION - VALIDATION TESTS")
    print("=" * 70)
    
    test_results = []
    
    # Test 1: API Health
    test_results.append(("API Health", test_api_health()))
    
    # Test 2: Vector Stats
    stats_success, stats_data = test_vector_stats()
    test_results.append(("Vector Stats", stats_success))
    
    if stats_success and stats_data:
        coverage = stats_data.get('database_stats', {}).get('embedding_coverage_percent', 0)
        if coverage < 100:
            # Test 3: Embed Policies (if needed)
            test_results.append(("Embed Policies", test_embed_policies()))
        else:
            print("\n✅ All policies already embedded, skipping embed test")
    
    # Test 4: Vector Search
    test_results.append(("Vector Similarity Search", test_vector_similarity_search()))
    
    # Test 5: Comparison
    test_results.append(("Vector vs Regular Search", test_comparison_with_regular_search()))
    
    # Test 6: Database Check
    db_result = asyncio.run(test_database_vector_column())
    test_results.append(("Database Vector Column", db_result))
    
    # Final Report
    print("\n" + "=" * 70)
    print("📊 FINAL TEST REPORT")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if result:
            passed += 1
    
    print(f"\nSCORE: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 CONGRATULATIONS!")
        print("✅ Phase 1 RAG Foundation is successfully implemented!")
        print("🚀 Ready for Phase 2: RAG Pipeline Development")
    elif passed >= total * 0.8:
        print("\n🎯 MOSTLY SUCCESSFUL!")
        print("✅ RAG Foundation is working with minor issues")
        print("🔧 Fix failing tests before proceeding to Phase 2")
    else:
        print("\n⚠️  ISSUES DETECTED")
        print("❌ Several tests failed - investigate and fix before proceeding")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        exit(1)