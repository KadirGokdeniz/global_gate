import asyncio
import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
import logging
from functools import lru_cache
import os
from functools import lru_cache
from typing import Dict
import hashlib
import redis

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Sentence Transformers ile embedding service"""
    
    def __init__(self):
        self.model = None
        self._model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self._embedding_dim = 384
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self.max_cache_size = 100
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = 3600  # 1 hour
        
    def _load_model(self):
        """Model'i lazy loading ile yükle"""
        if self.model is None:
            logger.info(f"🧠 Loading embedding model: {self._model_name}")
            self.model = SentenceTransformer(self._model_name)
            logger.info("✅ Embedding model loaded successfully")
        return self.model
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Tek text için embedding oluştur"""
        model = self._load_model()
        
        # Text preprocessing
        cleaned_text = self._preprocess_text(text)
        
        # Generate embedding
        embedding = model.encode(cleaned_text, normalize_embeddings=True)
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Cache'de var mı kontrol et
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Generate new embedding
        embedding = self._generate_new_embedding(text)
        
        # Cache'e ekle (size limit ile)
        if len(self._embedding_cache) < self.max_cache_size:
            self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Batch text'ler için embeddings oluştur"""
        model = self._load_model()
        
        # Batch preprocessing
        cleaned_texts = [self._preprocess_text(text) for text in texts]
        
        # Batch encode
        embeddings = model.encode(cleaned_texts, normalize_embeddings=True, batch_size=32)
        
        return embeddings
    
    def generate_embedding_with_redis(self, text: str) -> np.ndarray:
        cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        
        # Redis'ten kontrol et
        cached = self.redis_client.get(cache_key)
        if cached:
            return pickle.loads(cached)
        
        # Generate ve cache'e koy
        embedding = self._generate_new_embedding(text)
        self.redis_client.setex(cache_key, self.cache_ttl, pickle.dumps(embedding))
        return embedding
    
    def normalize_query(self, query: str) -> str:
        # Aynı anlamdaki sorular aynı cache key'e düşsün
        query = query.lower().strip()
        query = re.sub(r'[^\w\s]', '', query)  # Punctuation remove
        query = ' '.join(sorted(query.split()))  # Word order normalize
        return query
    
    def _preprocess_text(self, text: str) -> str:
        """Text preprocessing"""
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Truncate if too long (model has 512 token limit)
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        return text
    
    @property
    def embedding_dimension(self) -> int:
        """Embedding boyutu"""
        return self._embedding_dim

# Global instance
@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()