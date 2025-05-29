import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import logging
from functools import lru_cache
import asyncio
import time
from collections import OrderedDict
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class LRUCache:
    """Custom LRU Cache implementation for embeddings"""
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
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
                # Add new
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
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
            "hit_rate": round(hit_rate, 2)
        }
    
    def clear(self):
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

class BatchProcessor:
    """Batch processing for single requests"""
    
    def __init__(self, batch_size: int = 32, wait_time: float = 0.01):
        self.batch_size = batch_size
        self.wait_time = wait_time  # 10ms wait time
        self.queue = []
        self.futures = {}
        self._lock = threading.Lock()
        self._processing = False
        
    def add_request(self, text: str, future):
        with self._lock:
            self.queue.append((text, future))
            if not self._processing and len(self.queue) >= self.batch_size:
                self._process_batch()
            elif not self._processing:
                # Start timer for partial batch
                threading.Timer(self.wait_time, self._process_batch).start()
    
    def _process_batch(self):
        with self._lock:
            if self._processing or not self.queue:
                return
            self._processing = True
            batch = self.queue[:]
            self.queue.clear()
        
        try:
            # Extract texts and futures
            texts = [item[0] for item in batch]
            futures = [item[1] for item in batch]
            
            # This will be called by the embedding service
            # For now, just mark as ready to be processed
            for i, (text, future) in enumerate(batch):
                if not future.done():
                    future.set_result((text, i))  # Return text and index for batch processing
                    
        finally:
            with self._lock:
                self._processing = False

class OptimizedEmbeddingService:
    """High-performance embedding service with advanced optimizations"""
    
    def __init__(self, cache_size: int = 1000, enable_batch_processing: bool = True):
        # Model configuration
        self.model = None
        self._model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self._embedding_dim = 384
        
        # Advanced LRU cache
        self.cache = LRUCache(cache_size)
        
        # Batch processing
        self.enable_batch_processing = enable_batch_processing
        self.batch_processor = BatchProcessor() if enable_batch_processing else None
        
        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Preprocessing cache to avoid duplicate work
        self._preprocess_cache = LRUCache(cache_size // 2)
        
    def _load_model(self):
        """Model'i thread-safe lazy loading ile yükle"""
        if self.model is None:
            logger.info(f"🧠 Loading embedding model: {self._model_name}")
            
            # Local CPU optimizations (model instance'a özel)
            import torch
            
            self.model = SentenceTransformer(
                self._model_name,
                device='cpu',
                cache_folder='/app/model_cache'
            )
            
            # Model-specific thread setting
            if hasattr(self.model[0], 'auto_model'):
                # Set threads only for this model's tokenizer/encoder
                import os
                os.environ['OMP_NUM_THREADS'] = '2'
                os.environ['MKL_NUM_THREADS'] = '2'
            
            logger.info("✅ Embedding model loaded successfully")
        return self.model

    def _get_cache_key(self, text: str) -> str:
        """Simple and fast cache key generation"""
        # Basit normalization - MD5 hash yerine direct string
        normalized = text.lower().strip()
        normalized = ' '.join(normalized.split())
        
        # Long text için truncate + hash, short text için direct
        if len(normalized) > 200:
            return f"long_{hash(normalized)}"  # Python'ın built-in hash'i MD5'den hızlı
        return f"short_{normalized}"

    def _preprocess_text_cached(self, text: str) -> str:
        """Cached preprocessing to avoid duplicate work"""
        cache_key = f"prep_{text[:100]}"  # First 100 chars as key
        
        cached = self._preprocess_cache.get(cache_key)
        if cached is not None:
            return str(cached)  # Cache'den string olarak dön
        
        # Actual preprocessing
        if not text:
            result = ""
        else:
            text = text.strip()
            text = ' '.join(text.split())
            if len(text) > 1000:
                text = text[:1000] + "..."
            result = text
        
        # Cache result as string
        self._preprocess_cache.put(cache_key, np.array([result], dtype=object))
        return result

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding with advanced optimizations"""
        
        # Step 1: Single preprocessing
        cleaned_text = self._preprocess_text_cached(text)
        
        # Step 2: Check cache with preprocessed text
        cache_key = self._get_cache_key(cleaned_text)
        cached_embedding = self.cache.get(cache_key)
        if cached_embedding is not None:
            logger.debug(f"💨 Cache hit for: {text[:30]}...")
            return cached_embedding
        
        # Step 3: Batch processing for single requests (if enabled)
        if self.enable_batch_processing:
            return self._process_with_batching(cleaned_text, cache_key)
        
        # Step 4: Direct processing
        return self._generate_single_embedding(cleaned_text, cache_key)
    
    def _process_with_batching(self, cleaned_text: str, cache_key: str) -> np.ndarray:
        """Process single request through batch system"""
        future = threading.Event()
        result_container = {}
        
        def callback():
            embedding = self._generate_single_embedding(cleaned_text, cache_key)
            result_container['embedding'] = embedding
            future.set()
        
        # Add to batch queue or process immediately
        self.executor.submit(callback)
        future.wait()  # Wait for completion
        
        return result_container['embedding']
    
    def _generate_single_embedding(self, cleaned_text: str, cache_key: str) -> np.ndarray:
        """Generate single embedding without batching"""
        model = self._load_model()
        
        # Generate embedding
        embedding = model.encode(
            cleaned_text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Add to cache
        self.cache.put(cache_key, embedding)
        
        logger.debug(f"🔄 Generated new embedding for: {cleaned_text[:30]}...")
        return embedding

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """Optimized batch processing with cache integration"""
        
        if not texts:
            return []
        
        # Step 1: Preprocess all texts once
        cleaned_texts = [self._preprocess_text_cached(text) for text in texts]
        
        # Step 2: Check cache for all texts
        cache_keys = [self._get_cache_key(text) for text in cleaned_texts]
        results = []
        uncached_indices = []
        uncached_texts = []
        
        for i, (text, cache_key) in enumerate(zip(cleaned_texts, cache_keys)):
            cached = self.cache.get(cache_key)
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        logger.info(f"📊 Cache stats: {len(results)}/{len(texts)} hits ({len(results)/len(texts)*100:.1f}%)")
        
        # Step 3: Process uncached texts in batches
        if uncached_texts:
            model = self._load_model()
            
            # Process in optimized batches
            uncached_embeddings = model.encode(
                uncached_texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=len(uncached_texts) > 50,
                convert_to_numpy=True
            )
            
            # Add to cache and results
            for idx, embedding in zip(uncached_indices, uncached_embeddings):
                cache_key = cache_keys[idx]
                self.cache.put(cache_key, embedding)
                results.append((idx, embedding))
        
        # Step 4: Sort results by original order
        results.sort(key=lambda x: x[0])
        final_embeddings = [embedding for _, embedding in results]
        
        logger.info(f"🚀 Generated {len(uncached_texts)} new embeddings, {len(results) - len(uncached_texts)} from cache")
        return final_embeddings
    
    def get_performance_stats(self) -> Dict:
        """Comprehensive performance statistics"""
        cache_stats = self.cache.get_stats()
        preprocess_stats = self._preprocess_cache.get_stats()
        
        return {
            "main_cache": cache_stats,
            "preprocessing_cache": preprocess_stats,
            "batch_processing_enabled": self.enable_batch_processing,
            "model_loaded": self.model is not None,
            "embedding_dimension": self._embedding_dim
        }
    
    def clear_all_caches(self):
        """Clear all caches"""
        self.cache.clear()
        self._preprocess_cache.clear()
        logger.info("🗑️ All caches cleared")
    
    def warm_up_cache(self, common_texts: List[str]):
        """Pre-populate cache with common texts"""
        logger.info(f"🔥 Warming up cache with {len(common_texts)} common texts")
        self.generate_embeddings_batch(common_texts)
        logger.info("✅ Cache warm-up completed")


# Global instance with better defaults
@lru_cache()
def get_optimized_embedding_service(cache_size: int = 2000) -> OptimizedEmbeddingService:
    """Get global optimized embedding service instance"""
    return OptimizedEmbeddingService(cache_size=cache_size)


# Backward compatibility
def get_embedding_service():
    """Backward compatibility - returns optimized version"""
    return get_optimized_embedding_service()