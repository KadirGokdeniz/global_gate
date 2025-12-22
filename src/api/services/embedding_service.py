# STEP 1: Fixed Embedding Service - TRUE PRELOADING

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
    """Upgraded LRU Cache"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()
        
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
                
                return self.cache[key].copy()
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: np.ndarray):
        with self._lock:
            if key in self.cache:
                self.cache[key] = value.copy()
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
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
    """Embedding service - TRUE PRELOADING FIXED"""
    
    def __init__(self, cache_size: int = 2000, preload: bool = True):
        # Model konfigürasyonu
        self.model = None
        self._model_name = 'Alibaba-NLP/gte-multilingual-base'    # public property
        self.embedding_dimension = 768  # public property
        self._model_loaded = False
        self._load_lock = threading.Lock()
        
        # Cache
        self.cache = ImprovedLRUCache(cache_size)
        
        # Preprocessing optimization
        self._preprocessing_cache = {}
        
        # FIXED: TRUE PRELOADING - Load model immediately if requested
        if preload:
            self._preload_model_now()
        
        logger.info(f"Embedding Service initialized (preload: {preload})")
    
    def _preload_model_now(self):
        """IMMEDIATE model loading"""
        logger.info(f"🔥 PRELOADING MODEL: {self._model_name}")
        start_time = time.time()
        
        try:
            # Ensure model cache directory exists
            model_cache_dir = '/app/model_cache'
            os.makedirs(model_cache_dir, exist_ok=True)
            
            # Load model immediately
            self.model = SentenceTransformer(
                self._model_name,
                device='cpu',
                cache_folder=model_cache_dir,
                trust_remote_code=True,
                revision="main"
            )
            
            # Model warm-up with real embedding
            warmup_texts = [
                "Turkish Airlines baggage policy",
                "Pegasus Airlines pet travel",
                "International flight requirements"
            ]
            
            logger.info("⚡ Warming up model...")
            for text in warmup_texts:
                _ = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            
            self._model_loaded = True
            load_time = time.time() - start_time
            
            logger.info(f"✅ Model preloaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Model preloading FAILED: {e}")
            self._model_loaded = False
            self.model = None
            raise
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded (fallback if preloading failed)"""
        if not self._model_loaded or self.model is None:
            logger.warning("⚠️ Model not preloaded, loading now...")
            self._preload_model_now()
        return self.model
    
    def _optimize_cache_key(self, text: str) -> str:
        """Optimized cache key generation"""
        normalized = text.lower().strip()
        normalized = ' '.join(normalized.split())
        
        if len(normalized) > 100:
            return f"hash_{hashlib.md5(normalized.encode()).hexdigest()[:16]}"
        else:
            return f"text_{normalized[:100]}"
    
    def _preprocess_text_fast(self, text: str) -> str:
        """Hızlı text preprocessing"""
        if not text:
            return ""
        
        if text in self._preprocessing_cache:
            return self._preprocessing_cache[text]
        
        cleaned = text.strip()
        cleaned = ' '.join(cleaned.split())
        
        if len(cleaned) > 1000:
            cleaned = cleaned[:1000] + "..."
        
        if len(self._preprocessing_cache) < 500:
            self._preprocessing_cache[text] = cleaned
        
        return cleaned
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding - uses PRELOADED model"""
        
        # Step 1: Preprocess
        cleaned_text = self._preprocess_text_fast(text)
        if not cleaned_text:
            return np.zeros(self.embedding_dimension)
        
        # Step 2: Check cache
        cache_key = self._optimize_cache_key(cleaned_text)
        cached_embedding = self.cache.get(cache_key)
        
        if cached_embedding is not None:
            return cached_embedding
        
        # Step 3: Generate using PRELOADED model
        try:
            model = self._ensure_model_loaded()  # Should already be loaded
            
            embedding = model.encode(
                cleaned_text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Step 4: Cache result
            self.cache.put(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return np.zeros(self.embedding_dimension)
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Batch processing with PRELOADED model"""
        
        if not texts:
            return []
        
        logger.info(f"📄 Processing batch of {len(texts)} texts")
        start_time = time.time()
        
        # Step 1: Check cache for all texts
        results = []
        uncached_indices = []
        uncached_texts = []
        
        for i, text in enumerate(texts):
            cleaned = self._preprocess_text_fast(text)
            if not cleaned:
                results.append((i, np.zeros(self.embedding_dimension)))
                continue
            
            cache_key = self._optimize_cache_key(cleaned)
            cached = self.cache.get(cache_key)
            
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_indices.append(i)
                uncached_texts.append(cleaned)
        
        cache_hit_rate = (len(results) / len(texts)) * 100
        logger.info(f"📊 Cache hit rate: {cache_hit_rate:.1f}%")
        
        # Step 2: Process uncached with PRELOADED model
        if uncached_texts:
            try:
                model = self._ensure_model_loaded()  # Should already be loaded
                
                new_embeddings = model.encode(
                    uncached_texts,
                    normalize_embeddings=True,
                    batch_size=batch_size,
                    show_progress_bar=len(uncached_texts) > 50,
                    convert_to_numpy=True
                )
                
                # Add to results and cache
                for i, (idx, embedding) in enumerate(zip(uncached_indices, new_embeddings)):
                    results.append((idx, embedding))
                    
                    cleaned_text = uncached_texts[i]  # ← Doğru
                    cache_key = self._optimize_cache_key(cleaned_text)
                    self.cache.put(cache_key, embedding)
                
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                for idx in uncached_indices:
                    results.append((idx, np.zeros(self.embedding_dimension)))
        
        # Step 3: Sort and return
        results.sort(key=lambda x: x[0])
        final_embeddings = [embedding for _, embedding in results]
        
        processing_time = time.time() - start_time
        logger.info(f"🚀 Batch completed: {len(texts)} texts in {processing_time:.2f}s")
        
        return final_embeddings
    
    def get_performance_stats(self) -> Dict:
        """Performance stats"""
        cache_stats = self.cache.get_stats()
        
        return {
            "model_loaded": self._model_loaded,
            "model_name": self._model_name,
            "embedding_dimension": self.embedding_dimension,
            "cache_performance": cache_stats,
            "preprocessing_cache_size": len(self._preprocessing_cache),
            "total_cache_requests": cache_stats["total_requests"]
        }
    
    def is_ready(self) -> bool:
        """Check if service is ready for use"""
        return self._model_loaded and self.model is not None

# Global instance with TRUE PRELOADING
@lru_cache()
def get_embedding_service() -> OptimizedEmbeddingService:
    """Get embedding service instance - PRELOADED"""
    return OptimizedEmbeddingService(cache_size=2000, preload=True)