#!/usr/bin/env python3
"""
debug_system.py - Complete system health check and debugging
Docker sistemini ve Pegasus verilerini kontrol eder
"""

import requests
import json
import sys
import time
from datetime import datetime

# API Base URL
API_BASE = "http://localhost:8000"

def print_header(title):
    print("\n" + "="*60)
    print(f"🔍 {title}")
    print("="*60)

def check_docker_services():
    """Docker servislerinin durumunu kontrol et"""
    print_header("DOCKER SERVICES STATUS")
    
    import subprocess
    
    try:
        # Docker compose durumu
        result = subprocess.run(['docker-compose', 'ps'], 
                              capture_output=True, text=True)
        print("📋 Docker Compose Services:")
        print(result.stdout)
        
        # Specific service checks
        services = ['db', 'api', 'frontend']
        for service in services:
            result = subprocess.run(['docker-compose', 'logs', '--tail=10', service], 
                                  capture_output=True, text=True)
            print(f"\n📜 {service.upper()} LOGS (last 10 lines):")
            print(result.stdout[-500:])  # Last 500 chars
            
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        print("💡 Make sure Docker and docker-compose are running")

def check_api_health():
    """API health check"""
    print_header("API HEALTH CHECK")
    
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is healthy!")
            print(f"📊 Status: {health_data.get('status', 'unknown')}")
            print(f"🗄️ Database: {health_data.get('database', 'unknown')}")
            
            # RAG features check
            rag_features = health_data.get('rag_features', {})
            print(f"\n🧠 RAG Features:")
            for feature, status in rag_features.items():
                emoji = "✅" if status == "available" else "❌"
                print(f"  {emoji} {feature}: {status}")
            
            # Connection pool info
            pool_info = health_data.get('connection_pool', {})
            print(f"\n🔗 Database Pool:")
            print(f"  Size: {pool_info.get('size', 'unknown')}")
            print(f"  Max: {pool_info.get('max_size', 'unknown')}")
            
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API - is the service running?")
        print("💡 Try: docker-compose up api")
        return False
    except Exception as e:
        print(f"❌ API health check error: {e}")
        return False

def check_database_content():
    """Database içeriğini kontrol et"""
    print_header("DATABASE CONTENT CHECK")
    
    try:
        # Basic stats
        response = requests.get(f"{API_BASE}/stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            
            print(f"📊 Total Policies: {stats.get('total_policies', 0)}")
            
            # Source breakdown
            source_breakdown = stats.get('source_breakdown', {})
            print(f"\n📂 Source Breakdown:")
            for source, info in source_breakdown.items():
                print(f"  📄 {source}: {info.get('count', 0)} policies")
                print(f"      Quality: {info.get('avg_quality', 0):.2f}")
            
            # Database info
            db_info = stats.get('database_info', {})
            print(f"\n🗄️ Database Info:")
            print(f"  Oldest: {db_info.get('oldest_record', 'unknown')}")
            print(f"  Newest: {db_info.get('newest_record', 'unknown')}")
            print(f"  Sources: {db_info.get('unique_sources', 0)}")
            
            return stats
        else:
            print(f"❌ Database stats failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Database check error: {e}")
        return None

def check_pegasus_data():
    """Pegasus verilerini özel olarak kontrol et"""
    print_header("PEGASUS DATA SPECIFIC CHECK")
    
    try:
        # Pegasus policies'leri direkt iste
        response = requests.get(f"{API_BASE}/policies", 
                               params={'limit': 100}, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            policies = data.get('data', [])
            
            # Airline breakdown
            airline_count = {}
            for policy in policies:
                source = policy.get('source', 'unknown')
                if 'pegasus' in source.lower():
                    airline_count['pegasus'] = airline_count.get('pegasus', 0) + 1
                elif 'turkish' in source.lower():
                    airline_count['turkish'] = airline_count.get('turkish', 0) + 1
                else:
                    airline_count['other'] = airline_count.get('other', 0) + 1
            
            print(f"📊 Airline Data Distribution:")
            for airline, count in airline_count.items():
                print(f"  ✈️ {airline.title()}: {count} policies")
            
            # Pegasus specific sources
            pegasus_sources = {}
            for policy in policies:
                source = policy.get('source', 'unknown')
                if 'pegasus' in source.lower():
                    pegasus_sources[source] = pegasus_sources.get(source, 0) + 1
            
            if pegasus_sources:
                print(f"\n🔥 Pegasus Sources Found:")
                for source, count in pegasus_sources.items():
                    print(f"  📄 {source}: {count} policies")
                    
                # Show sample Pegasus content
                pegasus_policy = next((p for p in policies if 'pegasus' in p.get('source', '').lower()), None)
                if pegasus_policy:
                    print(f"\n📋 Sample Pegasus Content:")
                    content = pegasus_policy.get('content', '')
                    print(f"  Source: {pegasus_policy.get('source', 'unknown')}")
                    print(f"  Quality: {pegasus_policy.get('quality_score', 0):.2f}")
                    print(f"  Content: {content[:200]}...")
                    
                return True
            else:
                print("❌ NO PEGASUS DATA FOUND!")
                print("💡 This is likely the problem - Pegasus scraping failed")
                return False
        else:
            print(f"❌ Policies fetch failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Pegasus data check error: {e}")
        return False

def check_vector_operations():
    """Vector/embedding durumunu kontrol et"""
    print_header("VECTOR OPERATIONS CHECK")
    
    try:
        response = requests.get(f"{API_BASE}/vector/stats", timeout=10)
        
        if response.status_code == 200:
            stats = response.json()
            
            print(f"🧠 Embedding Model: {stats.get('embedding_model', 'unknown')}")
            print(f"📐 Dimension: {stats.get('embedding_dimension', 'unknown')}")
            
            db_stats = stats.get('database_stats', {})
            print(f"\n📊 Embedding Stats:")
            print(f"  Total Policies: {db_stats.get('total_policies', 0)}")
            print(f"  Embedded: {db_stats.get('embedded_policies', 0)}")
            print(f"  Missing: {db_stats.get('missing_embeddings', 0)}")
            print(f"  Coverage: {db_stats.get('embedding_coverage_percent', 0)}%")
            
            if db_stats.get('missing_embeddings', 0) > 0:
                print("⚠️ Some policies are not embedded!")
                print("💡 Try running embedding process again")
                
            return True
        else:
            print(f"❌ Vector stats failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Vector operations check error: {e}")
        return False

def test_pegasus_search():
    """Pegasus ile ilgili arama testi"""
    print_header("PEGASUS SEARCH TEST")
    
    # Test queries specifically for Pegasus
    test_queries = [
        "Pegasus baggage allowance",
        "Pegasus excess baggage fees", 
        "Pegasus carry-on restrictions",
        "What is Pegasus baggage policy?"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        
        try:
            # Vector search test
            response = requests.get(f"{API_BASE}/vector/similarity-search", 
                                   params={'q': query, 'limit': 3}, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])
                
                print(f"  📊 Vector search: {len(results)} results")
                
                for i, result in enumerate(results[:2], 1):
                    source = result.get('source', 'unknown')
                    similarity = result.get('similarity_score', 0)
                    content = result.get('content', '')[:100]
                    
                    is_pegasus = 'pegasus' in source.lower()
                    emoji = "🔥" if is_pegasus else "🇹🇷"
                    
                    print(f"    {emoji} Result {i}: {source} (sim: {similarity:.2f})")
                    print(f"       {content}...")
                    
            else:
                print(f"  ❌ Vector search failed: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Search test error: {e}")

def test_openai_rag():
    """OpenAI RAG sistemini test et"""
    print_header("OPENAI RAG TEST")
    
    test_question = "What are Pegasus Airlines baggage fees for domestic flights?"
    
    print(f"🤖 Testing RAG with: '{test_question}'")
    
    try:
        response = requests.post(f"{API_BASE}/chat/openai", 
                               params={
                                   'question': test_question,
                                   'max_results': 3,
                                   'similarity_threshold': 0.4,
                                   'model': 'gpt-3.5-turbo'
                               }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                print("✅ RAG response generated successfully!")
                
                answer = data.get('answer', '')
                print(f"\n🤖 Answer: {answer[:300]}...")
                
                retrieved_docs = data.get('retrieved_docs', [])
                print(f"\n📚 Retrieved {len(retrieved_docs)} documents:")
                
                pegasus_count = 0
                for doc in retrieved_docs:
                    source = doc.get('source', 'unknown')
                    similarity = doc.get('similarity_score', 0)
                    
                    is_pegasus = 'pegasus' in source.lower()
                    emoji = "🔥" if is_pegasus else "🇹🇷"
                    
                    if is_pegasus:
                        pegasus_count += 1
                    
                    print(f"  {emoji} {source} (similarity: {similarity:.2f})")
                
                if pegasus_count == 0:
                    print("⚠️ No Pegasus documents retrieved for Pegasus-specific question!")
                    print("💡 This suggests Pegasus data is missing or not properly embedded")
                else:
                    print(f"✅ Found {pegasus_count} Pegasus documents in results")
                
                # Check usage stats
                usage = data.get('usage_stats', {})
                if usage:
                    print(f"\n💰 OpenAI Usage:")
                    print(f"  Tokens: {usage.get('total_tokens', 0)}")
                    print(f"  Cost: ${usage.get('estimated_cost', 0):.4f}")
                    
            else:
                print(f"❌ RAG failed: {data.get('error', 'unknown error')}")
                
        else:
            print(f"❌ RAG request failed: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ RAG test error: {e}")

def check_streamlit_connection():
    """Streamlit bağlantısını kontrol et"""
    print_header("STREAMLIT CONNECTION CHECK")
    
    try:
        # Streamlit health check
        response = requests.get("http://localhost:8501/_stcore/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ Streamlit is running")
            
            # Check if Streamlit can reach API
            print("\n🔗 Testing Streamlit -> API connection...")
            
            # This simulates what Streamlit does
            api_test = requests.get(f"{API_BASE}/health", timeout=5)
            if api_test.status_code == 200:
                print("✅ Streamlit can reach API")
            else:
                print("❌ Streamlit cannot reach API")
                print("💡 Check if API container is running")
            
        else:
            print(f"❌ Streamlit not healthy: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Streamlit")
        print("💡 Try: docker-compose up frontend")
    except Exception as e:
        print(f"❌ Streamlit check error: {e}")

def run_complete_diagnosis():
    """Tam sistem teşhisi"""
    print("\n🏥 COMPLETE SYSTEM DIAGNOSIS")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all checks
    checks = [
        ("Docker Services", check_docker_services),
        ("API Health", check_api_health),
        ("Database Content", check_database_content),
        ("Pegasus Data", check_pegasus_data),
        ("Vector Operations", check_vector_operations),
        ("Pegasus Search", test_pegasus_search),
        ("OpenAI RAG", test_openai_rag),
        ("Streamlit Connection", check_streamlit_connection)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n⏳ Running {check_name}...")
        try:
            result = check_func()
            results[check_name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            results[check_name] = f"💥 ERROR: {str(e)}"
            print(f"💥 {check_name} crashed: {e}")
    
    # Final summary
    print_header("DIAGNOSIS SUMMARY")
    
    for check_name, result in results.items():
        print(f"{result} {check_name}")
    
    # Specific recommendations
    failed_checks = [name for name, result in results.items() if "FAIL" in result or "ERROR" in result]
    
    if failed_checks:
        print(f"\n🔧 RECOMMENDED ACTIONS:")
        
        if "Docker Services" in failed_checks:
            print("  1. Check Docker: docker-compose ps")
            print("  2. Restart services: docker-compose restart")
        
        if "Pegasus Data" in failed_checks:
            print("  3. Re-run scraper: docker-compose up scraper")
            print("  4. Check scraper logs: docker-compose logs scraper")
        
        if "Vector Operations" in failed_checks:
            print("  5. Re-embed data: POST /vector/embed-policies")
            print("  6. Check embedding logs")
        
        if "OpenAI RAG" in failed_checks:
            print("  7. Check OpenAI API key: OPENAI_API_KEY")
            print("  8. Verify model access: gpt-3.5-turbo")
            
    else:
        print("\n🎉 ALL CHECKS PASSED!")
        print("✅ System appears to be working correctly")
        print("💡 If Streamlit still has issues, try refreshing the page")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "quick":
            check_api_health()
            check_pegasus_data()
        elif command == "pegasus":
            check_pegasus_data()
            test_pegasus_search()
        elif command == "docker":
            check_docker_services()
        elif command == "rag":
            test_openai_rag()
        else:
            print("Usage: python debug_system.py [quick|pegasus|docker|rag]")
    else:
        run_complete_diagnosis()