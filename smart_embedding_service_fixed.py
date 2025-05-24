# app/services/embedding_service.py
"""
Unified Embedding Service - Production Ready
Consolidates all embedding strategies into one clean service
"""

import asyncio
import time
import hashlib
import logging
from typing import List, Optional, Dict, Any, Union
import numpy as np
from enum import Enum

# Optional dependencies with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class EmbeddingProvider(Enum):
    """Available embedding providers"""
    OPENAI = "openai"
    SENTENCE_TRANSFORMER = "sentence_transformer" 
    HASH_FALLBACK = "hash_fallback"

class EmbeddingService:
    """
    Production-ready unified embedding service
    
    Priority order:
    1. Memory cache (instant)
    2. OpenAI API (best quality, costs money)
    3. SentenceTransformers (good quality, free)
    4. Hash fallback (basic functionality)
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        cache_size: int = 1000
    ):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.dimension = dimension
        self.cache_size = cache_size
        
        # Service instances
        self.openai_client: Optional[AsyncOpenAI] = None
        self.transformer_model: Optional[SentenceTransformer] = None
        
        # Cache and stats
        self.cache: Dict[str, List[float]] = {}
        self.stats = {
            "cache_hits": 0,
            "openai_calls": 0,
            "transformer_calls": 0,
            "hash_fallback_calls": 0,
            "errors": 0
        }
        
        self.initialized = False
        logger.info(f"🧠 EmbeddingService initialized (target dim: {dimension})")
    
    async def initialize(self) -> None:
        """Initialize available embedding providers"""
        if self.initialized:
            return
            
        providers = []
        
        # 1. Try OpenAI
        if (self.openai_api_key 
            and OPENAI_AVAILABLE 
            and self.openai_api_key != "your_openai_api_key_here"):
            
            try:
                self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
                
                # Quick test
                response = await self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input="test"
                )
                providers.append(EmbeddingProvider.OPENAI)
                logger.info("✅ OpenAI API ready")
                
            except Exception as e:
                logger.warning(f"❌ OpenAI failed: {str(e)[:100]}...")
                self.openai_client = None
        
        # 2. Try SentenceTransformers
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"📥 Loading model: {self.model_name}")
                self.transformer_model = SentenceTransformer(self.model_name)
                providers.append(EmbeddingProvider.SENTENCE_TRANSFORMER)
                logger.info("✅ SentenceTransformers ready")
                
            except Exception as e:
                logger.warning(f"❌ SentenceTransformers failed: {e}")
                self.transformer_model = None
        
        # 3. Hash fallback always available
        providers.append(EmbeddingProvider.HASH_FALLBACK)
        
        self.initialized = True
        logger.info(f"🎯 Available providers: {[p.value for p in providers]}")
        
        if len(providers) == 1:  # Only hash fallback
            logger.warning("⚠️ Only hash fallback available. Consider installing sentence-transformers or configuring OpenAI")
    
    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding using best available provider"""
        if not self.initialized:
            await self.initialize()
        
        if not text or not text.strip():
            return [0.0] * self.dimension
        
        text = text.strip()
        start_time = time.time()
        
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.cache:
            self.stats["cache_hits"] += 1
            elapsed = (time.time() - start_time) * 1000
            logger.debug(f"💨 Cache hit: {elapsed:.1f}ms")
            return self.cache[cache_key]
        
        # Try providers in order
        embedding = await self._create_embedding_with_fallback(text)
        
        # Cache result
        self._cache_embedding(cache_key, embedding)
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"📊 Embedding created: {elapsed:.1f}ms")
        
        return embedding
    
    async def create_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32
    ) -> List[List[float]]:
        """Create embeddings for multiple texts efficiently"""
        if not texts:
            return []
        
        # Separate cached and uncached
        cached_results = {}
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                cached_results[i] = [0.0] * self.dimension
                continue
                
            cache_key = hashlib.md5(text.strip().encode()).hexdigest()
            if cache_key in self.cache:
                cached_results[i] = self.cache[cache_key]
                self.stats["cache_hits"] += 1
            else:
                uncached_texts.append(text.strip())
                uncached_indices.append(i)
        
        logger.info(f"📊 Batch: {len(texts)} total, {len(uncached_texts)} need embedding")
        
        # Process uncached texts
        if uncached_texts:
            if self.transformer_model and len(uncached_texts) > 3:
                # Use transformer for batch processing
                try:
                    embeddings = self.transformer_model.encode(
                        uncached_texts, 
                        batch_size=batch_size,
                        show_progress_bar=False
                    )
                    
                    for i, embedding in enumerate(embeddings):
                        idx = uncached_indices[i]
                        embedding_list = embedding.tolist()
                        cached_results[idx] = embedding_list
                        
                        # Cache it
                        cache_key = hashlib.md5(uncached_texts[i].encode()).hexdigest()
                        self._cache_embedding(cache_key, embedding_list)
                    
                    self.stats["transformer_calls"] += len(uncached_texts)
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Fallback to individual processing
                    for i, text in enumerate(uncached_texts):
                        idx = uncached_indices[i]
                        cached_results[idx] = await self.create_embedding(text)
            else:
                # Process individually
                for i, text in enumerate(uncached_texts):
                    idx = uncached_indices[i]
                    cached_results[idx] = await self.create_embedding(text)
        
        # Reconstruct results in original order
        return [cached_results[i] for i in range(len(texts))]
    
    async def _create_embedding_with_fallback(self, text: str) -> List[float]:
        """Try embedding providers in priority order"""
        
        # 1. Try OpenAI
        if self.openai_client:
            try:
                response = await self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                embedding = response.data[0].embedding
                
                # Normalize to target dimension if needed
                if len(embedding) != self.dimension:
                    embedding = self._normalize_dimension(embedding)
                
                self.stats["openai_calls"] += 1
                return embedding
                
            except Exception as e:
                logger.warning(f"OpenAI failed: {e}")
                self.openai_client = None  # Disable for session
        
        # 2. Try SentenceTransformers
        if self.transformer_model:
            try:
                embedding = self.transformer_model.encode(text, show_progress_bar=False)
                self.stats["transformer_calls"] += 1
                return embedding.tolist()
                
            except Exception as e:
                logger.error(f"SentenceTransformers failed: {e}")
                self.transformer_model = None
        
        # 3. Hash fallback
        self.stats["hash_fallback_calls"] += 1
        if self.stats["hash_fallback_calls"] == 1:  # Only warn once
            logger.warning("🚨 Using hash fallback - consider installing sentence-transformers")
        
        return self._create_hash_embedding(text)
    
    def _normalize_dimension(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to target dimension"""
        if len(embedding) == self.dimension:
            return embedding
        
        if len(embedding) > self.dimension:
            # Truncate and normalize
            truncated = np.array(embedding[:self.dimension])
            norm = np.linalg.norm(truncated)
            if norm > 0:
                return (truncated / norm).tolist()
            return truncated.tolist()
        else:
            # Pad with zeros
            return embedding + [0.0] * (self.dimension - len(embedding))
    
    def _create_hash_embedding(self, text: str) -> List[float]:
        """Enhanced hash-based embedding for fallback"""
        embedding = np.zeros(self.dimension)
        
        # Multi-hash approach for better distribution
        for i in range(0, self.dimension, 32):
            hash_input = f"{text}_{i}".encode()
            hash_val = hashlib.sha256(hash_input).hexdigest()
            
            for j in range(min(32, self.dimension - i)):
                if j * 2 + 1 < len(hash_val):
                    hex_byte = hash_val[j*2:j*2+2]
                    embedding[i + j] = (int(hex_byte, 16) - 128) / 128.0
        
        # Add semantic keywords
        keywords = {
            'laptop': [50, 100, 150], 'computer': [50, 100, 150],
            'electronics': [75, 125, 175], 'device': [75, 125, 175],
            'liquid': [200, 210, 220], 'baggage': [25, 75, 125],
            'weight': [250, 260], 'carry': [175, 225, 275],
            'checked': [100, 150, 200], 'prohibited': [300, 310, 320]
        }
        
        text_lower = text.lower()
        for keyword, dims in keywords.items():
            if keyword in text_lower:
                for dim_idx in dims:
                    if dim_idx < len(embedding):
                        embedding[dim_idx] += 0.3
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def _cache_embedding(self, cache_key: str, embedding: List[float]) -> None:
        """Cache embedding with size management"""
        self.cache[cache_key] = embedding
        
        # Manage cache size
        if len(self.cache) > self.cache_size:
            # Remove oldest 20% of entries
            remove_count = len(self.cache) // 5
            old_keys = list(self.cache.keys())[:remove_count]
            for key in old_keys:
                del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        total_calls = (
            self.stats["openai_calls"] + 
            self.stats["transformer_calls"] + 
            self.stats["hash_fallback_calls"]
        )
        
        cache_hit_rate = 0.0
        if total_calls + self.stats["cache_hits"] > 0:
            cache_hit_rate = self.stats["cache_hits"] / (total_calls + self.stats["cache_hits"])
        
        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.1%}",
            "cache_size": len(self.cache),
            "providers_available": {
                "openai": self.openai_client is not None,
                "sentence_transformers": self.transformer_model is not None,
                "hash_fallback": True
            },
            "dimension": self.dimension,
            "initialized": self.initialized
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("🔒 EmbeddingService cleanup")
        self.cache.clear()
        if hasattr(self, 'openai_client'):
            self.openai_client = None
        if hasattr(self, 'transformer_model'):
            self.transformer_model = None