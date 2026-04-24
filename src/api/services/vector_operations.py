import asyncpg
from typing import List, Dict, Optional, Tuple
import numpy as np
from api.services.embedding_service import get_embedding_service
import logging
import json
import asyncio
import hashlib
import time
from functools import lru_cache
from api.services.reranker_service import get_reranker_service

logger = logging.getLogger(__name__)

class EnhancedVectorOperations:
    """PostgreSQL pgvector operations with Cohere reranking pipeline"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.embedding_service = get_embedding_service()
        self.reranker = get_reranker_service()
        # Semantic category embeddings cache
        self.category_embeddings_cache = {}
        self.category_embeddings_loaded = False
        self._embedding_lock = asyncio.Lock()
        
        # Content embeddings cache - PERFORMANCE OPTIMIZATION
        self.content_embeddings_cache = {}
        self.max_content_cache_size = 1000  # Max 1000 content embeddings in memory
        
        # Initialize category embeddings
        asyncio.create_task(self._initialize_category_embeddings())
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
            
        return dot_product / norm_product
    
    async def _initialize_category_embeddings(self):
        """Compute category centroids from actual DB embeddings"""
        if self.category_embeddings_loaded:
            return
            
        async with self._embedding_lock:
            if self.category_embeddings_loaded:
                return
                
            logger.info("🚀 Computing category centroids from DB...")
            start_time = time.time()
            
            async with self.db_pool.acquire() as conn:
                # Her source için tüm embedding'leri çek
                rows = await conn.fetch("""
                    SELECT source, embedding
                    FROM policy
                    WHERE embedding IS NOT NULL
                    ORDER BY source
                """)
            
            if not rows:
                logger.warning("No embeddings found in DB, category routing disabled")
                self.category_embeddings_loaded = True
                return
            
            # Source'a göre grupla
            from collections import defaultdict
            source_embeddings = defaultdict(list)
            
            for row in rows:
                source = row['source']
                # pgvector string'ini numpy array'e çevir
                embedding_str = row['embedding']
                if isinstance(embedding_str, str):
                    # "[0.1, 0.2, ...]" formatından array'e
                    embedding_array = np.array(
                        [float(x) for x in embedding_str.strip('[]').split(',')]
                    )
                else:
                    embedding_array = np.array(embedding_str)
                
                source_embeddings[source].append(embedding_array)
            
            # Her source için centroid hesapla
            self.category_embeddings_cache = {}
            for source, embeddings in source_embeddings.items():
                if embeddings:
                    centroid = np.mean(embeddings, axis=0)
                    # Normalize et
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm
                    self.category_embeddings_cache[source] = centroid
            
            self.category_embeddings_loaded = True
            
            load_time = time.time() - start_time
            logger.info(
                f"✅ Category centroids computed: "
                f"{len(self.category_embeddings_cache)} categories from "
                f"{len(rows)} embeddings in {load_time:.2f}s"
            )
    
    def _get_cached_content_embedding(self, content: str) -> np.ndarray:
        """
        Get content embedding from cache or generate and cache it
        PERFORMANCE: Eliminates re-embedding of same content
        
        NOTE: Cohere reranker kullanıldığında bu cache'e ihtiyaç kalmaz,
        ama semantic_category_detection hâlâ embedding üretiyor — orada gerekli.
        """
        # Use first 750 chars for consistency with reranking
        content_snippet = content[:750]
        
        content_hash = hashlib.md5(content_snippet.encode()).hexdigest()[:16]
        
        # Check cache first
        if content_hash in self.content_embeddings_cache:
            return self.content_embeddings_cache[content_hash].copy()
        
        # Generate new embedding
        embedding = self.embedding_service.generate_embedding(content_snippet)
        
        # Cache management - LRU style
        if len(self.content_embeddings_cache) >= self.max_content_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.content_embeddings_cache))
            del self.content_embeddings_cache[oldest_key]
        
        # Store in cache
        self.content_embeddings_cache[content_hash] = embedding.copy()
        
        return embedding
    
    async def semantic_category_detection(
        self, 
        query: str, 
        airline_filter: Optional[str] = None, 
        similarity_threshold: float = 0.35,
        max_categories: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Semantic category detection using pre-computed category embeddings
        Returns: List of (category_name, semantic_score) tuples
        """
        
        # Ensure category embeddings are loaded
        await self._initialize_category_embeddings()
        
        if not query or len(query.strip()) < 3:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Filter available categories by airline if specified
        available_categories = self.category_embeddings_cache.copy()
        
        if airline_filter == 'pegasus':
            # Pegasus specific categories + general ones
            pegasus_categories = {
                k: v for k, v in available_categories.items() 
                if k.startswith(('baggage_allowance', 'extra_services', 'travelling_with_pets', 'general_rules', 'pets', 'sports_equipment', 'musical_instruments', 'restrictions'))
            }
            available_categories = pegasus_categories
        elif airline_filter == 'turkish_airlines':
            # Turkish Airlines specific categories (keep all)
            pass  # Use all categories
        
        # Calculate semantic similarities
        category_scores = []
        for category_name, category_embedding in available_categories.items():
            similarity = self.cosine_similarity(query_embedding, category_embedding)
            
            if similarity >= similarity_threshold:
                category_scores.append((category_name, similarity))
        
        # Sort by similarity score and limit
        category_scores.sort(key=lambda x: x[1], reverse=True)
        result = category_scores[:max_categories]
        
        if result:
            categories_str = ", ".join([f"{cat}({score:.2f})" for cat, score in result])
            scope = f"airline: {airline_filter}" if airline_filter else "all airlines"
            logger.info(f"Semantic detection for '{query[:50]}...' ({scope}): {categories_str}")
        
        return result
    
    def detect_query_categories_hybrid(
        self, 
        query: str, 
        airline_filter: Optional[str] = None, 
        max_categories: int = 3
    ) -> List[str]:
        """
        Hybrid approach: Fast keyword detection + semantic validation
        This is the fallback/validation method for semantic detection
        """
        if not query or len(query.strip()) < 3:
            return []
        
        query_lower = query.lower()
        
        # Simplified keyword mapping for validation
        KEYWORD_CATEGORIES = {
            'turkish_airlines': {
                'checked_baggage': ['bavul', 'suitcase', 'checked', 'hold', 'ecofly'],
                'carry_on_baggage': ['kabin', 'cabin', 'hand', 'overhead'],
                'excess_baggage': ['extra', 'additional', 'fazla', 'overweight'],
                'sports_golf': ['golf', 'sopa', 'club'],
                'sports_bicycle': ['bisiklet', 'bicycle', 'bike'],
                'sports_skiing': ['kayak', 'ski', 'snowboard'],
                'pets_cargo': ['kargo', 'cargo', 'hold'],
                'pets_cabin': ['kabin', 'cabin', 'small'],
                'musical_instruments': ['müzik', 'music', 'instrument'],
                'restrictions': ['yasak', 'forbidden', 'restricted']
            },
            'pegasus': {
                'baggage_allowance': ['paket', 'package', 'light', 'saver'],
                'extra_services_pricing': ['TRY', 'EUR', 'pricing'],
                'travelling_with_pets': ['PETC', 'evcil', 'pet'],
                'general_rules': ['kural', 'rules', 'bolbol']
            }
        }
        
        # Select appropriate categories
        if airline_filter and airline_filter in KEYWORD_CATEGORIES:
            available_categories = KEYWORD_CATEGORIES[airline_filter]
        else:
            available_categories = {}
            for airline_cats in KEYWORD_CATEGORIES.values():
                available_categories.update(airline_cats)
        
        # Keyword matching
        detected = []
        for category, keywords in available_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(category)
        
        return detected[:max_categories]
    
    async def embed_existing_policies(self, batch_size: int = 32) -> int:
        """Embed existing policies without embeddings"""
        logger.info("Starting to embed existing policies...")
        
        async with self.db_pool.acquire() as conn:
            unembedded = await conn.fetch("""
                SELECT id, content 
                FROM policy
                WHERE embedding IS NULL
                ORDER BY id
            """)
            
            if not unembedded:
                logger.info("All policies already have embeddings")
                return 0
            
            logger.info(f"Found {len(unembedded)} policies to embed")
            
            total_embedded = 0
            for i in range(0, len(unembedded), batch_size):
                batch = unembedded[i:i + batch_size]
                
                texts = [row['content'] for row in batch]
                embeddings = self.embedding_service.generate_embeddings_batch(texts)
                
                for j, (row, embedding) in enumerate(zip(batch, embeddings)):
                    await conn.execute("""
                        UPDATE policy 
                        SET embedding = $1::vector
                        WHERE id = $2
                    """, "[" + ",".join(map(str, embedding.tolist())) + "]", row['id'])
                
                total_embedded += len(batch)
                logger.info(f"  Embedded batch {i//batch_size + 1}: {total_embedded}/{len(unembedded)}")
            
            logger.info(f"Successfully embedded {total_embedded} policies!")
            return total_embedded
    
    async def similarity_search(
        self, 
        query: str, 
        limit: int = 5,
        similarity_threshold: float = 0.3,
        airline_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        use_semantic_categories: bool = True
    ) -> List[Dict]:
        """Pure semantic similarity search with semantic category detection"""
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Pure semantic category detection
        semantic_categories = []
        if use_semantic_categories:
            semantic_results = await self.semantic_category_detection(
                query, airline_filter, similarity_threshold=0.35, max_categories=3
            )
            semantic_categories = [cat for cat, _ in semantic_results]
        
        # Build SQL query
        sql = """
            SELECT 
                id, airline, source, content, quality_score,
                created_at, updated_at, url,
                1 - (embedding <=> $1::vector) as similarity_score
            FROM policy
            WHERE embedding IS NOT NULL
        """
        params = ["[" + ",".join(map(str, query_embedding.tolist())) + "]"]

        # Airline filter
        if airline_filter:
            sql += f" AND airline = ${len(params) + 1}"
            params.append(airline_filter)
        
        # Source filter 
        if source_filter:
            sql += f" AND source = ${len(params) + 1}"
            params.append(source_filter)
        
        # Semantic category filtering
        if semantic_categories:
            placeholders = []
            for category in semantic_categories:
                placeholders.append(f"${len(params) + 1}")
                params.append(category)
            
            if placeholders:
                sql += f" AND source = ANY(ARRAY[{', '.join(placeholders)}])"
        
        # Similarity threshold
        sql += f" AND (1 - (embedding <=> $1::vector)) >= ${len(params) + 1}"
        params.append(similarity_threshold)
        
        # Pure semantic ordering
        sql += f" ORDER BY embedding <=> $1::vector LIMIT ${len(params) + 1}"
        params.append(limit)
        
        async with self.db_pool.acquire() as conn:
            results = await conn.fetch(sql, *params)
            
            formatted_results = []
            for row in results:
                result_dict = dict(row)
                if result_dict.get('created_at'):
                    result_dict['created_at'] = result_dict['created_at'].isoformat()
                if result_dict.get('updated_at'):
                    result_dict['updated_at'] = result_dict['updated_at'].isoformat()
                formatted_results.append(result_dict)
            
            # Enhanced logging
            filter_info = f"(airline: {airline_filter or 'all'})"
            if semantic_categories:
                filter_info += f" (semantic categories: {', '.join(semantic_categories)})"
            
            logger.info(f"Pure semantic search for '{query[:50]}...': {len(formatted_results)} results {filter_info}")
            return formatted_results
    
    async def bm25_search(
        self,
        query: str,
        limit: int = 20,
        airline_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Keyword-based search using PostgreSQL full-text search (tsvector + GIN).
        
        BM25-like ranking via ts_rank_cd. Complements dense search by catching
        exact matches dense can't: acronyms (PETC, BolBol), codes (3.2.1.b),
        proper nouns (Esenboğa), numbers (20 kg).
        
        Returns same format as similarity_search but with bm25_score field.
        Gracefully degrades on error (returns [] instead of crashing).
        """
        if not query or len(query.strip()) < 2:
            return []
        
        # websearch_to_tsquery kullanıyoruz (plainto_tsquery değil):
        # plainto_tsquery: tüm kelimeleri AND'ler. Bir kelime eksikse sıfır match.
        #   Örn: "Pegasus Airlines | satin alabilir miyim" →
        #        'pegasus' & 'airlines' & 'satin' & 'alabilir' & 'miyim'
        #        "satin" veya "miyim" İngilizce chunk'ta yoksa hiç match yok.
        # websearch_to_tsquery: Google-style search syntax, default OR, stop words tolere edilir.
        #   Aynı query → 'pegasus' | 'airlines' | 'satin' | ...
        #   Herhangi bir kelime match yeterli, exact phrase "..." ile de desteklenir.
        # Bu hybrid search için ideal — BM25 relaxed match, sonra Cohere rerank kesin ayıklar.
        sql = """
            SELECT 
                id, airline, source, content, quality_score,
                created_at, updated_at, url, metadata,
                ts_rank_cd(content_tsv, websearch_to_tsquery('simple', $1)) AS bm25_score
            FROM policy
            WHERE content_tsv @@ websearch_to_tsquery('simple', $1)
        """
        params = [query]
        
        if airline_filter:
            sql += f" AND airline = ${len(params) + 1}"
            params.append(airline_filter)
        
        sql += f" ORDER BY bm25_score DESC LIMIT ${len(params) + 1}"
        params.append(limit)
        
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
                
                results = []
                for row in rows:
                    result_dict = dict(row)
                    if result_dict.get('created_at'):
                        result_dict['created_at'] = result_dict['created_at'].isoformat()
                    if result_dict.get('updated_at'):
                        result_dict['updated_at'] = result_dict['updated_at'].isoformat()
                    # BM25 skorunu aynı zamanda similarity_score olarak da set et
                    # (aşağıdaki pipeline adımları similarity_score bekliyor)
                    result_dict['similarity_score'] = float(result_dict.get('bm25_score', 0.0))
                    results.append(result_dict)
                
                filter_info = f"(airline: {airline_filter or 'all'})"
                logger.info(f"BM25 search for '{query[:50]}...': {len(results)} results {filter_info}")
                return results
        
        except Exception as e:
            # tsvector kolonu yok, query parse hatası, vs - graceful fallback
            # Hybrid search dense-only moda düşer, sistem çökmez
            logger.warning(f"BM25 search failed ({type(e).__name__}: {e}) — returning []")
            return []
    
    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60,
    ) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) — iki ranked list'i tek skora birleştir.
        
        Formula: rrf_score(doc) = 1/(k + dense_rank) + 1/(k + bm25_rank)
        
        k=60 standart (TREC default). Küçük k = top results dominate,
        büyük k = uniform contribution.
        
        Neden RRF?
          - Absolute skorları birleştirmez (cosine 0-1, bm25 0-∞ karşılaştırılamaz)
          - Sadece rank bilgisi kullanır (hangi listede kaçıncı sırada)
          - Her iki listede de olan belgeler otomatik olarak öne çıkar
          - Birinde olup diğerinde olmayanlar da kazanabilir ama daha düşük skor
        """
        fused_scores: Dict = {}  # doc_id -> {score, doc}
        
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc['id']
            contribution = 1.0 / (k + rank)
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {'score': 0.0, 'doc': doc, 'dense_rank': rank, 'bm25_rank': None}
            fused_scores[doc_id]['score'] += contribution
        
        for rank, doc in enumerate(bm25_results, start=1):
            doc_id = doc['id']
            contribution = 1.0 / (k + rank)
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {'score': 0.0, 'doc': doc, 'dense_rank': None, 'bm25_rank': rank}
            else:
                fused_scores[doc_id]['bm25_rank'] = rank
            fused_scores[doc_id]['score'] += contribution
        
        # RRF skoruna göre sırala, doc objelerini döndür
        sorted_items = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        
        fused_docs = []
        for item in sorted_items:
            doc = item['doc'].copy()
            doc['rrf_score'] = item['score']
            doc['dense_rank'] = item['dense_rank']
            doc['bm25_rank'] = item['bm25_rank']
            fused_docs.append(doc)
        
        return fused_docs
    
    async def semantic_reranking_search(
        self,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.3,
        airline_filter: Optional[str] = None,
        rerank_factor: float = 1.2,
        expanded_limit_factor: float = 4.0
    ) -> List[Dict]:
        """
        Advanced HYBRID reranking with Dense + BM25 + Cohere Rerank v3.5.

        PIPELINE:
        1. Semantic category detection (kept)
        2. Parallel retrieval:
           - Dense search (pgvector cosine, 20 candidates)
           - BM25 search (tsvector + GIN, 20 candidates)
        3. Reciprocal Rank Fusion (merged unique candidates)
        4. Cohere cross-encoder reranking (TRUE relevance scoring)
        5. Category boost (cohere_score × category_boost)
        6. Final top-N selection

        Why hybrid?
          Dense catches meaning ("köpek" ≈ "pet" ≈ "evcil").
          BM25 catches exact tokens (PETC, 3.2.1.b, BolBol, "20 kg").
          RRF lets each method find what the other misses.

        Fallback chain:
          - If BM25 fails (no index, query parse error): use only dense
          - If Cohere fails (no API key, rate limit): use RRF order
          - If everything fails: return []
        """
        start_time = time.time()

        # ─────────────────────────────────────────────────────────────
        # Step 1: Semantic category detection (unchanged)
        # ─────────────────────────────────────────────────────────────
        semantic_results = await self.semantic_category_detection(
            query, airline_filter, similarity_threshold=0.35, max_categories=3
        )
        detected_categories = [cat for cat, _ in semantic_results]
        category_scores = {cat: score for cat, score in semantic_results}

        # ─────────────────────────────────────────────────────────────
        # Step 2: Parallel retrieval — Dense + BM25
        # ─────────────────────────────────────────────────────────────
        expanded_limit = int(limit * expanded_limit_factor)  # 5 × 4 = 20
        
        # BM25 için query temizliği:
        # retrieve_relevant_docs query'yi "Pegasus Airlines | Pegasus Extra Seat..." 
        # şeklinde enhance ediyor. Dense bu prefix'ten faydalanır (airline kontexti
        # embedding'e yardımcı), ama BM25 için prefix gereksiz — hem zaten airline_filter
        # var, hem de gereksiz tokenlar OR'landığında alakasız chunk'ları da çeker.
        # Son "|" karakterinden sonrasını al, yoksa query'nin tamamı.
        bm25_query = query.split("|")[-1].strip() if "|" in query else query
        
        # Both searches run concurrently (asyncio.gather)
        dense_task = self.similarity_search(
            query=query,
            limit=expanded_limit,
            similarity_threshold=similarity_threshold,
            airline_filter=airline_filter,
            use_semantic_categories=True
        )
        bm25_task = self.bm25_search(
            query=bm25_query,  # temiz query — enhanced prefix'siz
            limit=expanded_limit,
            airline_filter=airline_filter,
        )
        
        dense_results, bm25_results = await asyncio.gather(
            dense_task, bm25_task, return_exceptions=False
        )
        
        # Step 3: Reciprocal Rank Fusion
        if bm25_results:
            # Both methods contributed → fuse
            candidates = self._reciprocal_rank_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
                k=60,  # TREC standard
            )
            # Limit to expanded_limit unique candidates
            candidates = candidates[:expanded_limit]
            retrieval_mode = "hybrid"
        else:
            # BM25 failed or returned empty → dense only
            candidates = dense_results
            retrieval_mode = "dense_only"

        if not candidates:
            logger.warning(f"No candidates for query '{query[:50]}...' — returning []")
            return []

        # ─────────────────────────────────────────────────────────────
        # Step 4: Cohere reranking (true cross-encoder)
        # ─────────────────────────────────────────────────────────────
        if self.reranker.is_available():
            # Over-fetch slightly so category boost has room to reshuffle
            rerank_top_n = min(limit * 2, len(candidates))
            reranked = await self.reranker.rerank(
                query=query,
                documents=candidates,
                top_n=rerank_top_n,
                content_key="content"
            )
            rerank_used = True
        else:
            # Fallback: keep RRF/cosine order
            logger.warning("Reranker unavailable, falling back to RRF/cosine order")
            reranked = candidates[:limit * 2]
            for i, r in enumerate(reranked):
                # Use RRF score if available, else similarity_score
                r['rerank_score'] = r.get('rrf_score', r.get('similarity_score', 0.0))
                r['original_rank'] = i
            rerank_used = False

        # ─────────────────────────────────────────────────────────────
        # Step 5: Category boost (applied on top of Cohere score)
        # ─────────────────────────────────────────────────────────────
        for candidate in reranked:
            source_category = candidate.get('source', '')
            category_boost = 1.0

            if source_category in detected_categories:
                semantic_score = category_scores.get(source_category, 0.5)
                category_boost = 1.0 + (semantic_score * (rerank_factor - 1.0))

            rerank_score = candidate.get('rerank_score', 0.0)
            enhanced_score = rerank_score * category_boost

            # Attach scores (backward-compatible with prior callers)
            candidate['category_boost'] = category_boost
            candidate['enhanced_relevance'] = enhanced_score
            candidate['content_similarity'] = rerank_score  # legacy alias
            candidate['original_similarity'] = candidate.get('similarity_score', 0.0)

        # ─────────────────────────────────────────────────────────────
        # Step 6: Final sort + limit
        # ─────────────────────────────────────────────────────────────
        final_results = sorted(
            reranked,
            key=lambda x: x['enhanced_relevance'],
            reverse=True
        )[:limit]

        # ─────────────────────────────────────────────────────────────
        # Logging (hybrid pipeline metrics)
        # ─────────────────────────────────────────────────────────────
        total_ms = (time.time() - start_time) * 1000
        categories_str = ", ".join([f"{cat}({score:.2f})" for cat, score in semantic_results])
        rerank_tag = "cohere" if rerank_used else "fallback"

        logger.info(
            f"Hybrid search '{query[:50]}...' "
            f"[mode={retrieval_mode}, dense={len(dense_results)}, bm25={len(bm25_results)}, "
            f"fused={len(candidates)}, rerank={rerank_tag}, final={len(final_results)}] "
            f"in {total_ms:.0f}ms, categories: {categories_str}"
        )

        return final_results
    
    async def preference_aware_search(
        self,
        query: str,
        airline_preference: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.3,
        use_semantic_reranking: bool = True
    ) -> List[Dict]:
        """
        Preference-aware search with semantic reranking option
        """
        
        if use_semantic_reranking:
            # Use the new semantic reranking pipeline
            return await self.semantic_reranking_search(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold,
                airline_filter=airline_preference,
                rerank_factor=1.2 if airline_preference else 1.0
            )
        else:
            # Fallback to basic similarity search
            return await self.similarity_search(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold,
                airline_filter=airline_preference,
                use_semantic_categories=True
            )
    
    async def get_airline_stats(self) -> Dict:
        """Get airline distribution statistics with category breakdown"""
        async with self.db_pool.acquire() as conn:
            stats = await conn.fetch("""
                SELECT 
                    airline,
                    COUNT(*) as total_policies,
                    COUNT(embedding) as embedded_policies,
                    COUNT(DISTINCT source) as unique_sources
                FROM policy 
                GROUP BY airline
                ORDER BY total_policies DESC
            """)
            
            # Source category distribution
            source_stats = await conn.fetch("""
                SELECT 
                    source,
                    COUNT(*) as policy_count,
                    COUNT(DISTINCT airline) as airline_count
                FROM policy
                WHERE source IS NOT NULL
                GROUP BY source
                ORDER BY policy_count DESC
            """)
            
            return {
                "airline_breakdown": [dict(row) for row in stats],
                "source_breakdown": [dict(row) for row in source_stats],
                "total_airlines": len(stats)
            }
    
    async def get_embedding_stats(self) -> Dict:
        """Enhanced embedding statistics with reranking pipeline info"""
        async with self.db_pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_policies,
                    COUNT(embedding) as embedded_policies,
                    COUNT(*) - COUNT(embedding) as missing_embeddings,
                    ROUND(
                        (COUNT(embedding)::float / COUNT(*) * 100)::numeric, 2
                    ) as embedding_coverage_percent
                FROM policy
            """)
            
            airline_stats = await self.get_airline_stats()
            
            return {
                **dict(stats),
                **airline_stats,
                "search_enhancements": {
                    "pipeline": "cohere-reranking + category-boost",
                    "reranker_available": self.reranker.is_available() if hasattr(self, 'reranker') else False,
                    "reranker_model": self.reranker.MODEL if hasattr(self, 'reranker') else None,
                    "category_embeddings_loaded": self.category_embeddings_loaded,
                    "semantic_detection_threshold": 0.35,
                    "reranking_strategy": "cohere_v3.5 + category_boost",
                    "supported_airlines": ["turkish_airlines", "pegasus", "all_combined"],
                    "pipeline_stages": ["semantic_category_detection", "expanded_vector_search", "cohere_reranking", "category_boost", "final_selection"]
                }
            }

    async def get_semantic_stats(self) -> Dict:
        """Get statistics about semantic enhancement system"""
        await self._initialize_category_embeddings()
        
        # Basic stats
        async with self.db_pool.acquire() as conn:
            basic_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_policies,
                    COUNT(embedding) as embedded_policies,
                    COUNT(DISTINCT airline) as total_airlines,
                    COUNT(DISTINCT source) as total_sources
                FROM policy
            """)
            
            source_distribution = await conn.fetch("""
                SELECT 
                    source,
                    COUNT(*) as policy_count,
                    airline
                FROM policy
                GROUP BY source, airline
                ORDER BY airline, policy_count DESC
            """)
        
        return {
            **dict(basic_stats),
            "semantic_enhancements": {
                "reranker_available": self.reranker.is_available() if hasattr(self, 'reranker') else False,
                "reranker_model": self.reranker.MODEL if hasattr(self, 'reranker') else None,
                "category_embeddings_loaded": self.category_embeddings_loaded,
                "total_semantic_categories": len(self.category_embeddings_cache),
                "semantic_detection_threshold": 0.35,
                "reranking_strategy": "cohere_v3.5 + category_boost",
                "supported_airlines": ["turkish_airlines", "pegasus", "all_combined"],
                "pipeline_stages": ["semantic_category_detection", "expanded_vector_search", "cohere_reranking", "category_boost", "final_selection"]
            },
            "source_distribution": [dict(row) for row in source_distribution],
            "category_embedding_status": {
                cat: "loaded" for cat in self.category_embeddings_cache.keys()
            } if self.category_embeddings_loaded else {"status": "loading"}
        }
    
    async def test_semantic_pipeline(self, test_queries: List[str], airline_filter: Optional[str] = None) -> Dict:
        """Test the semantic reranking pipeline with sample queries"""
        results = {}
        
        for query in test_queries:
            # Test semantic category detection
            semantic_categories = await self.semantic_category_detection(query, airline_filter)
            
            # Test basic vs semantic reranking
            basic_results = await self.similarity_search(query, limit=5, airline_filter=airline_filter, use_semantic_categories=False)
            semantic_results = await self.semantic_reranking_search(query, limit=5, airline_filter=airline_filter)
            
            # Compare results
            results[query] = {
                "airline_filter": airline_filter or "all_airlines",
                "semantic_categories": [f"{cat}({score:.2f})" for cat, score in semantic_categories],
                "basic_search_results": len(basic_results),
                "semantic_search_results": len(semantic_results),
                "position_changes": self._calculate_position_changes(basic_results, semantic_results),
                "avg_enhancement_score": np.mean([r.get('enhanced_relevance', 0) for r in semantic_results]) if semantic_results else 0,
                "avg_rerank_score": np.mean([r.get('rerank_score', 0) for r in semantic_results]) if semantic_results else 0
            }
        
        return {
            "test_results": results,
            "semantic_system_status": {
                "reranker_available": self.reranker.is_available() if hasattr(self, 'reranker') else False,
                "category_embeddings_loaded": self.category_embeddings_loaded,
                "total_categories": len(self.category_embeddings_cache),
                "embedding_service_ready": self.embedding_service.is_ready()
            }
        }
    
    def _calculate_position_changes(self, basic_results: List[Dict], semantic_results: List[Dict]) -> int:
        """Calculate how many positions changed between basic and semantic results"""
        basic_ids = [r['id'] for r in basic_results]
        semantic_ids = [r['id'] for r in semantic_results]
        
        changes = 0
        for i, semantic_id in enumerate(semantic_ids):
            if i < len(basic_ids) and basic_ids[i] != semantic_id:
                changes += 1
        
        return changes

# Convenience function for backward compatibility
async def get_enhanced_vector_operations(db_pool) -> EnhancedVectorOperations:
    """Get enhanced vector operations instance"""
    return EnhancedVectorOperations(db_pool)