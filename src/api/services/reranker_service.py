"""
Cohere Rerank service — RAG pipeline'ın ikinci aşaması.

similarity_search 20 aday belge getirir → rerank bu 20'den en iyi 5'i seçer.

Model: rerank-v3.5 (multilingual, 100+ dil, TR+EN optimize)

Kullanım:
    reranker = get_reranker_service()
    if reranker.is_available():
        reranked = await reranker.rerank(
            query="Pegasus köpekle seyahat",
            documents=[{"id": 1, "content": "..."}, ...],
            top_n=5
        )
"""
import logging
import time
import asyncio
from typing import List, Dict, Optional
from functools import lru_cache

import cohere
from api.core.secrets_loader import SecretsLoader

logger = logging.getLogger(__name__)


class CohereRerankerService:
    """Cohere-based reranker with graceful fallback."""

    # Model seçenekleri:
    #   rerank-v3.5                  (multilingual, stabil) ← kullanılan
    #   rerank-v4.0-pro              (en iyi kalite, yeni)
    #   rerank-v4.0-fast             (hızlı ama düşük kalite)
    #   rerank-multilingual-v3.0     (eski, legacy)
    MODEL = "rerank-v3.5"

    # Cohere'e gönderilecek max belge sayısı (maliyet + latency)
    MAX_DOCS = 50

    # Tek bir belge için max token (Cohere otomatik chunk'lar ama önce biz kısaltalım)
    MAX_DOC_CHARS = 2000

    def __init__(self):
        loader = SecretsLoader()
        self.api_key = loader.get_secret('cohere_api_key', 'COHERE_API_KEY')

        if not self.api_key:
            logger.warning("⚠️ COHERE_API_KEY bulunamadı — reranker devre dışı")
            self.client = None
            self._available = False
            return

        try:
            self.client = cohere.ClientV2(api_key=self.api_key)
            self._test_connection()
            self._available = True
            logger.info(f"✅ Cohere Reranker hazır (model: {self.MODEL})")
        except Exception as e:
            logger.error(f"❌ Cohere client init başarısız: {e}")
            self.client = None
            self._available = False

    def _test_connection(self):
        """Lightweight connection test — API key ve model erişimi doğrulama."""
        try:
            self.client.rerank(
                query="test query",
                documents=["sample document content"],
                top_n=1,
                model=self.MODEL
            )
        except cohere.errors.UnauthorizedError as e:
            raise RuntimeError(f"Cohere API key invalid: {e}")
        except Exception as e:
            raise RuntimeError(f"Cohere connection test failed: {e}")

    def is_available(self) -> bool:
        """Reranker aktif ve kullanıma hazır mı?"""
        return self._available

    async def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_n: int = 5,
        content_key: str = "content"
    ) -> List[Dict]:
        """
        Rerank documents by relevance to query.

        Args:
            query: Kullanıcının sorgusu
            documents: [{"id": ..., "content": ..., "similarity_score": ...}, ...]
            top_n: Dönecek belge sayısı
            content_key: Belgedeki metin field'ı (default "content")

        Returns:
            Reranked documents with additional fields:
                'rerank_score': Cohere'in verdiği relevance skoru (0-1)
                'original_rank': similarity_search'teki orijinal sıra
            Reranker kullanılamazsa → orijinal listeyi top_n kadarla kırpıp döndürür.
        """
        if not documents:
            return []

        # Fallback: reranker yoksa orijinal sıralamayı koru
        if not self._available:
            return documents[:top_n]

        # Cohere'e göndermeden önce sınırla + content'i kısalt
        candidates = documents[:self.MAX_DOCS]
        texts = []
        for doc in candidates:
            content = doc.get(content_key, "")
            if len(content) > self.MAX_DOC_CHARS:
                content = content[:self.MAX_DOC_CHARS]
            texts.append(content)

        start = time.time()

        try:
            # Cohere SDK sync — asyncio.to_thread ile async hale getir
            response = await asyncio.to_thread(
                self.client.rerank,
                query=query,
                documents=texts,
                top_n=min(top_n, len(texts)),
                model=self.MODEL
            )

            # Cohere response: response.results = [{index, relevance_score}, ...]
            reranked = []
            for result in response.results:
                # Orijinal belgenin tam halini kopyala, skor ekle
                doc = dict(candidates[result.index])
                doc['rerank_score'] = float(result.relevance_score)
                doc['original_rank'] = result.index
                reranked.append(doc)

            elapsed_ms = (time.time() - start) * 1000
            logger.info(
                f"Rerank: {len(candidates)} → {len(reranked)} in {elapsed_ms:.0f}ms"
            )
            return reranked

        except cohere.errors.TooManyRequestsError:
            logger.warning("⚠️ Cohere rate limit — fallback to original order")
            return documents[:top_n]

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.warning(
                f"⚠️ Rerank failed ({elapsed_ms:.0f}ms): {type(e).__name__}: {e} "
                f"— fallback to original order"
            )
            return documents[:top_n]


# Singleton
@lru_cache()
def get_reranker_service() -> CohereRerankerService:
    """Get Cohere reranker service (singleton)."""
    return CohereRerankerService()