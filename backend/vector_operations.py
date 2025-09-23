import asyncpg
from typing import List, Dict, Optional, Tuple
import numpy as np
from embedding_service import get_embedding_service
import logging
import json
import asyncio
import hashlib
import time
from functools import lru_cache

logger = logging.getLogger(__name__)

class EnhancedVectorOperations:
    """PostgreSQL pgvector operations with semantic reranking pipeline"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.embedding_service = get_embedding_service()
        
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
        """Pre-compute category representative embeddings for semantic detection"""
        if self.category_embeddings_loaded:
            return
            
        async with self._embedding_lock:
            if self.category_embeddings_loaded:  # Double check
                return
                
            logger.info("🚀 Pre-computing category embeddings for semantic detection...")
            start_time = time.time()
            
            # Optimized Dense Semantic Representatives - Concise & Distinctive
            CATEGORY_REPRESENTATIVES = {
                # Baggage Categories - Core Distinctions
                'checked_baggage': "checked bavul hold compartment 23kg 32kg ecofly business free allowance",
                'carry_on_baggage': "cabin kabin hand luggage 8kg 55x40x23 overhead personal item", 
                'excess_baggage': "additional ek fazla overweight fee ücret charge payment extra cost",
                
                # Sports - Distinctive Equipment Terms Only
                'sports_hockey': "hockey hokey stick sopa puck paten skates ice protective helmet",
                'sports_golf': "golf clubs balls tees putters drivers irons cart bag equipment",
                'sports_bicycle': "bicycle bisiklet bike helmet pedals wheels chain handlebar saddle",
                'sports_skiing': "ski kayak snowboard boots poles bindings snow mountain winter",
                'sports_diving': "diving dalış scuba regulator tank mask fins underwater equipment",
                'sports_surfing': "surfing sörf surfboard waves ocean beach water board equipment",
                'sports_mountaineering': "mountaineering dağcılık climbing rope harness carabiner ice axe boots",
                'sports_archery': "archery okçuluk bow yay arrow ok target shooting equipment",
                'sports_fishing': "fishing balık rod olta reel tackle bait hook angling equipment",
                'sports_hunting': "hunting avcılık rifle shotgun ammunition scope firearms equipment",
                'sports_canoeing': "canoe kano kayak paddle kürek river water navigation boat",
                'sports_rafting': "rafting inflatable boat river rapids water adventure equipment",
                'sports_windsurfing': "windsurfing sail board wind mast boom harness water sports",
                'sports_water_skiing': "water skiing wakeboard tow rope boat lake sports",
                'sports_bowling': "bowling ball pins lane shoes strike spare game equipment",
                'sports_tenting': "camping kamp tent çadır sleeping bag backpack hiking outdoor",
                'sports_parachuting': "parachuting paraşüt skydiving jump altitude safety gear equipment",
                
                # Pets - Size/Location Based Distinction
                'pets_cargo': "cargo kargo large dog büyük köpek cage kafes kennel hold",
                'pets_cabin': "cabin kabin small küçük cat kedi bird kuş carrier bag",
                'pets_service_animals': "service servis guide dog rehber köpeği trained assistance emotional",
                'pets_country_rules': "documents evrak vaccination aşı certificate sertifika country ülke import",
                'pets_terms': "conditions koşullar health sağlık sedative sedatif medication ilaç requirements",
                'pets_onboard': "onboard uçak allowed izinli regulations size weight limits cabin",
                
                # Other Categories - Core Distinctive Terms
                'musical_instruments': "instrument enstrüman guitar violin piano fragile kırılgan case protection",
                'restrictions': "prohibited yasak forbidden dangerous lighter çakmak battery sharp liquid",
                'sports_equipment': "sports spor equipment ekipman athletic gear general protection",
                'pets': "pet evcil animal hayvan domestic travel companion veterinary care",
                
                # Pegasus - Brand Specific Terms
                'baggage_allowance': "allowance hak package paket light saver comfort included dahil pegasus",
                'extra_services_pricing': "services hizmet pricing fee ücret TRY EUR SPEQ AVIH charge",
                'travelling_with_pets': "pets evcil PETC travel seyahat companion regulations pegasus animal",
                'general_rules': "rules kural terms koşul conditions bolbol points rewards pegasus policy"
            }
            
            # Generate embeddings for all categories
            category_texts = list(CATEGORY_REPRESENTATIVES.values())
            category_names = list(CATEGORY_REPRESENTATIVES.keys())
            
            # Batch generate embeddings
            category_embeddings = self.embedding_service.generate_embeddings_batch(
                category_texts, 
                batch_size=16
            )
            
            # Store in cache
            self.category_embeddings_cache = {
                name: embedding 
                for name, embedding in zip(category_names, category_embeddings)
            }
            
            self.category_embeddings_loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"✅ Category embeddings loaded: {len(self.category_embeddings_cache)} categories in {load_time:.2f}s")
    
    def _get_cached_content_embedding(self, content: str) -> np.ndarray:
        """
        Get content embedding from cache or generate and cache it
        PERFORMANCE: Eliminates re-embedding of same content
        """
        # Use first 500 chars for consistency with reranking
        content_snippet = content[:500]
        
        # Generate cache key
        import hashlib
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
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
            
        return dot_product / norm_product
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
            
        return dot_product / norm_product
    
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
    
    async def semantic_reranking_search(
        self,
        query: str,
        limit: int = 5,
        similarity_threshold: float = 0.3,
        airline_filter: Optional[str] = None,
        rerank_factor: float = 1.2,
        expanded_limit_factor: float = 2.0
    ) -> List[Dict]:
        """
        Advanced semantic reranking search pipeline
        
        PIPELINE:
        1. Semantic category detection
        2. Expanded vector search  
        3. Content-level semantic reranking
        4. Final result selection
        """
        
        # Step 1: Semantic category detection
        semantic_results = await self.semantic_category_detection(
            query, airline_filter, similarity_threshold=0.35, max_categories=3
        )
        detected_categories = [cat for cat, score in semantic_results]
        category_scores = {cat: score for cat, score in semantic_results}
        
        # Step 2: Expanded search for reranking candidates
        expanded_limit = int(limit * expanded_limit_factor)
        candidates = await self.similarity_search(
            query=query,
            limit=expanded_limit,
            similarity_threshold=similarity_threshold,
            airline_filter=airline_filter,
            use_semantic_categories=True  # Use semantic categories for pre-filtering
        )
        
        if not candidates:
            return []
        
        # Step 3: Content-level semantic reranking with CACHING
        query_embedding = self.embedding_service.generate_embedding(query)
        
        cache_hits = 0
        cache_misses = 0
        
        for candidate in candidates:
            # CACHED content embedding - NO re-embedding overhead!
            content_embedding = self._get_cached_content_embedding(candidate['content'])
            
            # Track cache performance
            content_hash = hashlib.md5(candidate['content'][:500].encode()).hexdigest()[:16]
            if content_hash in self.content_embeddings_cache:
                cache_hits += 1
            else:
                cache_misses += 1
            
            # Calculate direct semantic similarity
            content_similarity = self.cosine_similarity(query_embedding, content_embedding)
            
            # Category boost factor
            source_category = candidate.get('source', '')
            category_boost = 1.0
            
            if source_category in detected_categories:
                # Use the semantic score for more precise boosting
                semantic_score = category_scores.get(source_category, 0.5)
                category_boost = 1.0 + (semantic_score * (rerank_factor - 1.0))
            
            # Combined reranking score
            original_similarity = candidate['similarity_score']
            enhanced_score = (content_similarity * 0.6 + original_similarity * 0.4) * category_boost
            
            # Store both scores for analysis
            candidate['content_similarity'] = content_similarity
            candidate['category_boost'] = category_boost
            candidate['enhanced_relevance'] = enhanced_score
            candidate['original_similarity'] = original_similarity
        
        # Step 4: Rerank by enhanced relevance
        reranked_results = sorted(
            candidates, 
            key=lambda x: x['enhanced_relevance'], 
            reverse=True
        )[:limit]
        
        # Enhanced logging with cache performance
        rerank_changes = sum(1 for i, result in enumerate(reranked_results) 
                           if candidates.index(result) != i)
        
        cache_hit_rate = (cache_hits / (cache_hits + cache_misses) * 100) if (cache_hits + cache_misses) > 0 else 0
        categories_str = ", ".join([f"{cat}({score:.2f})" for cat, score in semantic_results])
        
        logger.info(f"Semantic reranking for '{query[:50]}...': {rerank_changes}/{limit} positions changed, "
                   f"cache: {cache_hits}/{cache_hits + cache_misses} hits ({cache_hit_rate:.1f}%), "
                   f"categories: {categories_str}")
        
        return reranked_results
    
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
                    "pipeline": "semantic-reranking + content-caching",
                    "category_embeddings_loaded": self.category_embeddings_loaded,
                    "semantic_detection_threshold": 0.35,
                    "reranking_strategy": "content_similarity + category_boost",
                    "supported_airlines": ["turkish_airlines", "pegasus", "all_combined"],
                    "pipeline_stages": ["semantic_category_detection", "expanded_vector_search", "content_reranking", "final_selection"]
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
                "category_embeddings_loaded": self.category_embeddings_loaded,
                "total_semantic_categories": len(self.category_embeddings_cache),
                "semantic_detection_threshold": 0.35,
                "reranking_strategy": "content_similarity + category_boost",
                "supported_airlines": ["turkish_airlines", "pegasus", "all_combined"],
                "pipeline_stages": ["semantic_category_detection", "expanded_vector_search", "content_reranking", "final_selection"]
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
                "avg_enhancement_score": np.mean([r.get('enhanced_relevance', 0) for r in semantic_results]) if semantic_results else 0
            }
        
        return {
            "test_results": results,
            "semantic_system_status": {
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