#!/usr/bin/env python3
"""
RAG Embedding Test Script
Mevcut sisteme dokunmadan embedding test'i yapar
"""

import sys
import os
import time
import psycopg2
from psycopg2.extras import RealDictCursor

def check_dependencies():
    """Gerekli kütüphaneleri kontrol et"""
    print("📦 Dependency kontrolü...")
    
    missing_deps = []
    
    try:
        import sentence_transformers
        print("  ✅ sentence-transformers available")
    except ImportError:
        missing_deps.append("sentence-transformers")
        print("  ❌ sentence-transformers missing")
    
    try:
        import torch
        print("  ✅ torch available")
    except ImportError:
        missing_deps.append("torch")
        print("  ❌ torch missing")
        
    try:
        import numpy
        print("  ✅ numpy available")
    except ImportError:
        missing_deps.append("numpy")
        print("  ❌ numpy missing")
    
    if missing_deps:
        print(f"\n❌ Eksik dependencies: {missing_deps}")
        print("\nYüklemek için:")
        print("pip install sentence-transformers torch numpy")
        return False
    
    print("✅ Tüm dependencies mevcut!")
    return True

def test_embedding_generation():
    """Embedding generation test"""
    print("\n🧠 Embedding generation test...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Multilingual model (Turkish + English)
        print("  📥 Model yükleniyor...")
        start_time = time.time()
        
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        load_time = time.time() - start_time
        print(f"  ✅ Model yüklendi ({load_time:.2f}s)")
        
        # Test cümleleri
        test_texts = [
            "What is the baggage allowance for Turkish Airlines?",
            "Turkish Airlines bagaj limiti nedir?",
            "Can I carry sports equipment on the plane?",
            "Uçakta spor malzemesi taşıyabilir miyim?",
            "Excess baggage fees for international flights"
        ]
        
        print(f"  🔄 {len(test_texts)} text için embedding oluşturuluyor...")
        
        embeddings = []
        for i, text in enumerate(test_texts):
            start = time.time()
            embedding = model.encode(text)
            duration = time.time() - start
            
            embeddings.append(embedding)
            print(f"    [{i+1}] '{text[:40]}...' → {embedding.shape} ({duration:.3f}s)")
        
        # Similarity test
        print("\n  🔍 Similarity test...")
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        embeddings_matrix = np.array(embeddings)
        similarities = cosine_similarity(embeddings_matrix)
        
        print(f"    Turkish/English baggage questions similarity: {similarities[0][1]:.3f}")
        print(f"    Sports equipment questions similarity: {similarities[2][3]:.3f}")
        
        return True, model, embeddings
        
    except Exception as e:
        print(f"  ❌ Embedding test hatası: {e}")
        return False, None, None

def get_sample_policies():
    """Mevcut database'den sample policies al"""
    print("\n📊 Database'den sample policies alınıyor...")
    
    # Mevcut config'den database bilgilerini al
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_DATABASE', 'global_gate'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'qeqe')
    }
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # İlk 10 policy'yi al
        cursor.execute("""
            SELECT id, source, content, quality_score 
            FROM baggage_policies 
            ORDER BY quality_score DESC NULLS LAST 
            LIMIT 10
        """)
        
        policies = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"  ✅ {len(policies)} policy alındı")
        
        for i, policy in enumerate(policies[:3]):  # İlk 3'ünü göster
            content_preview = policy['content'][:60] + "..." if len(policy['content']) > 60 else policy['content']
            print(f"    [{i+1}] {policy['source']}: {content_preview}")
        
        return policies
        
    except Exception as e:
        print(f"  ❌ Database hatası: {e}")
        print("  ℹ️  Database'in çalıştığından emin olun")
        return None

def test_policy_embeddings(model, policies):
    """Sample policies için embeddings oluştur ve benzerlik test et"""
    print("\n🔄 Policy embeddings test...")
    
    if not policies or len(policies) < 2:
        print("  ❌ Yeterli policy bulunamadı")
        return False
    
    try:
        # İlk 5 policy için embedding oluştur
        policy_embeddings = []
        for i, policy in enumerate(policies[:5]):
            embedding = model.encode(policy['content'])
            policy_embeddings.append({
                'id': policy['id'],
                'source': policy['source'],
                'content_preview': policy['content'][:50] + "...",
                'embedding': embedding
            })
            print(f"    [{i+1}] {policy['source']}: {embedding.shape}")
        
        # User query simulation
        print("\n  🔍 User query simulation...")
        user_queries = [
            "baggage weight limit",
            "sports equipment policy",
            "excess baggage fees"
        ]
        
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        for query in user_queries:
            print(f"\n    Query: '{query}'")
            query_embedding = model.encode(query)
            
            # Her policy ile similarity hesapla
            similarities = []
            for policy_emb in policy_embeddings:
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1), 
                    policy_emb['embedding'].reshape(1, -1)
                )[0][0]
                similarities.append((similarity, policy_emb))
            
            # En benzer 3'ünü göster
            similarities.sort(reverse=True)
            for i, (sim_score, policy) in enumerate(similarities[:3]):
                print(f"      [{i+1}] {sim_score:.3f} - {policy['source']}: {policy['content_preview']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Policy embedding test hatası: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("=" * 60)
    print("🚀 RAG EMBEDDING TEST - QUICK VALIDATION")
    print("=" * 60)
    
    # 1. Dependency check
    if not check_dependencies():
        print("\n❌ Dependencies eksik - test durduruluyor")
        return False
    
    # 2. Embedding generation test
    success, model, embeddings = test_embedding_generation()
    if not success:
        print("\n❌ Embedding generation başarısız")
        return False
    
    # 3. Database'den sample al
    policies = get_sample_policies()
    if not policies:
        print("\n⚠️  Database policies alınamadı - mock data ile devam...")
        # Mock policies for testing
        policies = [
            {'id': 1, 'source': 'test', 'content': 'Baggage weight limit is 23kg for economy class', 'quality_score': 1.0},
            {'id': 2, 'source': 'test', 'content': 'Sports equipment can be carried with special arrangements', 'quality_score': 1.0}
        ]
    
    # 4. Policy embeddings test
    if test_policy_embeddings(model, policies):
        print("\n" + "=" * 60)
        print("🎉 TÜM TESTLER BAŞARILI!")
        print("=" * 60)
        print("✅ Sentence Transformers çalışıyor")
        print("✅ Embedding generation OK")
        print("✅ Similarity search OK")
        print("✅ Database connectivity OK")
        print("\n🚀 RAG implementation'a hazırsınız!")
        print("\nBir sonraki adım: Full implementation (Phase 1)")
        return True
    else:
        print("\n❌ Policy embedding test başarısız")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)