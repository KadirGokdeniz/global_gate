# STEP 1: Optimized Embedding Service
# Değişiklikler: Model persistence, better caching, batch optimization

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from functools import lru_cache
import time
from collections import OrderedDict
import threading
import hashlib
import os

logger = logging.getLogger(__name__)

class ImprovedLRUCache:
    """Geliştirilmiş LRU Cache - mevcut cache'den daha verimli"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()  # ReentrantLock for thread safety
        
        # Performance tracking
        self.total_requests = 0
        self.avg_access_time = 0.0
    
    def get(self, key: str) -> Optional[np.ndarray]:
        start_time = time.time()
        
        with self._lock:
            self.total_requests += 1
            
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                
                # Update access time tracking
                access_time = time.time() - start_time
                self.avg_access_time = (
                    (self.avg_access_time * (self.total_requests - 1) + access_time) / 
                    self.total_requests
                )
                
                return self.cache[key].copy()  # Return copy to prevent modification
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: np.ndarray):
        with self._lock:
            if key in self.cache:
                # Update existing
                self.cache[key] = value.copy()
                self.cache.move_to_end(key)
            else:
                # Add new, remove oldest if necessary
                if len(self.cache) >= self.max_size:
                    # Remove 10% of oldest items for better performance
                    cleanup_count = max(1, self.max_size // 10)
                    for _ in range(cleanup_count):
                        if self.cache:
                            self.cache.popitem(last=False)
                
                self.cache[key] = value.copy()
    
    def get_stats(self) -> Dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2),
            "avg_access_time_ms": round(self.avg_access_time * 1000, 3),
            "total_requests": self.total_requests
        }

class OptimizedEmbeddingService:
    """İyileştirilmiş embedding service - step by step optimization"""
    
    def __init__(self, cache_size: int = 2000):
        # Model konfigürasyonu
        self.model = None
        self._model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self._embedding_dim = 384
        self._model_loaded = False
        self._load_lock = threading.Lock()
        
        # İyileştirilmiş cache
        self.cache = ImprovedLRUCache(cache_size)
        
        # Preprocessing optimization
        self._preprocessing_cache = {}
        self._stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
            'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about'
        }
        
        logger.info(f"🧠 Optimized Embedding Service initialized with cache size: {cache_size}")
    
    def _load_model_once(self):
        """Model'i sadece bir kez yükle ve memory'de tut"""
        
        if self._model_loaded:
            return self.model
        
        with self._load_lock:
            # Double-check locking pattern
            if self._model_loaded:
                return self.model
            
            logger.info(f"🔄 Loading model: {self._model_name} (one-time operation)")
            start_time = time.time()
            
            try:
                self.model = SentenceTransformer(
                    self._model_name,
                    device='cpu',  # CPU for stability
                    cache_folder='/app/model_cache'
                )
                
                # Model warm-up
                warmup_text = "Turkish Airlines baggage policy"
                _ = self.model.encode(warmup_text, convert_to_numpy=True)
                
                self._model_loaded = True
                load_time = time.time() - start_time
                
                logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Model loading failed: {e}")
                raise
        
        return self.model
    
    def _optimize_cache_key(self, text: str) -> str:
        """Optimize edilmiş cache key generation"""
        
        # Basit normalization
        normalized = text.lower().strip()
        normalized = ' '.join(normalized.split())  # Multiple spaces -> single space
        
        # Cache key strategy
        if len(normalized) > 100:
            # Long text: use hash
            return f"hash_{hashlib.md5(normalized.encode()).hexdigest()[:16]}"
        else:
            # Short text: use direct (more cache hits)
            return f"text_{normalized[:100]}"
    
    def _preprocess_text_fast(self, text: str) -> str:
        """Hızlı text preprocessing"""
        
        if not text:
            return ""
        
        # Check cache first
        if text in self._preprocessing_cache:
            return self._preprocessing_cache[text]
        
        # Simple preprocessing
        cleaned = text.strip()
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace
        
        # Truncate if too long (prevent memory issues)
        if len(cleaned) > 1000:
            cleaned = cleaned[:1000] + "..."
        
        # Cache result (limit cache size)
        if len(self._preprocessing_cache) < 500:
            self._preprocessing_cache[text] = cleaned
        
        return cleaned
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Ana embedding generation - optimized version"""
        
        # Step 1: Preprocess
        cleaned_text = self._preprocess_text_fast(text)
        if not cleaned_text:
            return np.zeros(self._embedding_dim)
        
        # Step 2: Check cache
        cache_key = self._optimize_cache_key(cleaned_text)
        cached_embedding = self.cache.get(cache_key)
        
        if cached_embedding is not None:
            logger.debug(f"💨 Cache hit for: {text[:30]}...")
            return cached_embedding
        
        # Step 3: Generate new embedding
        try:
            model = self._load_model_once()
            
            embedding = model.encode(
                cleaned_text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Step 4: Cache result
            self.cache.put(cache_key, embedding)
            
            logger.debug(f"🔄 Generated new embedding for: {text[:30]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return np.zeros(self._embedding_dim)
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Optimized batch processing"""
        
        if not texts:
            return []
        
        logger.info(f"🔄 Processing batch of {len(texts)} texts")
        start_time = time.time()
        
        # Step 1: Preprocess all texts and check cache
        results = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cleaned = self._preprocess_text_fast(text)
            if not cleaned:
                results.append((i, np.zeros(self._embedding_dim)))
                continue
            
            cache_key = self._optimize_cache_key(cleaned)
            cached = self.cache.get(cache_key)
            
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_indices.append(i)
                uncached_texts.append(cleaned)
        
        cache_hit_rate = (len(results) / len(texts)) * 100
        logger.info(f"📊 Cache hit rate: {cache_hit_rate:.1f}% ({len(results)}/{len(texts)})")
        
        # Step 2: Process uncached texts
        if uncached_texts:
            try:
                model = self._load_model_once()
                
                # Generate embeddings in batches
                new_embeddings = model.encode(
                    uncached_texts,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=len(uncached_texts) > 50,
                    convert_to_numpy=True
                )
                
                # Add to results and cache
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    results.append((idx, embedding))
                    
                    # Cache new embedding
                    cleaned_text = uncached_texts[uncached_indices.index(idx)]
                    cache_key = self._optimize_cache_key(cleaned_text)
                    self.cache.put(cache_key, embedding)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Fallback: add zero embeddings
                for idx in uncached_indices:
                    results.append((idx, np.zeros(self._embedding_dim)))
        
        # Step 3: Sort by original order and return
        results.sort(key=lambda x: x[0])
        final_embeddings = [embedding for _, embedding in results]
        
        processing_time = time.time() - start_time
        logger.info(f"🚀 Batch processing completed: {len(texts)} texts in {processing_time:.2f}s")
        
        return final_embeddings
    
    def get_performance_stats(self) -> Dict:
        """Performance istatistikleri"""
        
        cache_stats = self.cache.get_stats()
        
        return {
            "model_loaded": self._model_loaded,
            "model_name": self._model_name,
            "embedding_dimension": self._embedding_dim,
            "cache_performance": cache_stats,
            "preprocessing_cache_size": len(self._preprocessing_cache),
            "total_cache_requests": cache_stats["total_requests"]
        }
    
    def clear_cache(self):
        """Cache'i temizle (debugging için)"""
        self.cache = ImprovedLRUCache(self.cache.max_size)
        self._preprocessing_cache.clear()
        logger.info("🗑️ Cache cleared")

# Backward compatibility için global instance
@lru_cache()
def get_optimized_embedding_service() -> OptimizedEmbeddingService:
    """Optimized embedding service instance"""
    return OptimizedEmbeddingService(cache_size=2000)

# Eski interface için wrapper
def get_embedding_service():
    """Backward compatibility wrapper"""
    return get_optimized_embedding_service()