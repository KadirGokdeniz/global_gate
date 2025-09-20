import asyncpg
from typing import List, Dict, Optional, Tuple
import numpy as np
from embedding_service import get_embedding_service
import logging

logger = logging.getLogger(__name__)

class VectorOperations:
    """PostgreSQL pgvector operations with simple category enhancement"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.embedding_service = get_embedding_service()
    
    async def embed_existing_policies(self, batch_size: int = 50) -> int:
        """Mevcut tüm policy'leri embed et"""
        logger.info("Starting to embed existing policies...")
        
        async with self.db_pool.acquire() as conn:
            # Get policies without embeddings
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
            
            # Process in batches
            total_embedded = 0
            for i in range(0, len(unembedded), batch_size):
                batch = unembedded[i:i + batch_size]
                
                # Generate embeddings for batch
                texts = [row['content'] for row in batch]
                embeddings = self.embedding_service.generate_embeddings_batch(texts)
                
                # Update database
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
    
    def detect_query_categories(self, query: str, airline_filter: Optional[str] = None, max_categories: int = 3) -> List[str]:
        """Airline-aware kategori tespiti - Sadece seçilen havayoluna ait kategorilerde arama"""
        if not query or len(query.strip()) < 3:
            return []
        
        query_lower = query.lower()
        
        # AIRLINE-SPECIFIC CATEGORY MAPPING
        AIRLINE_CATEGORIES = {
            'turkish_airlines': {
                # Baggage Categories
                'checked_baggage': ['bavul', 'suitcase', 'checked', 'hold', 'ecofly', 'star'],
                'carry_on_baggage': ['kabin', 'cabin', 'el çantası', 'hand bag', 'overhead'],
                'excess_baggage': ['extra baggage fee', 'excess baggage fee', 'additional baggage fee',
                                   'fazla bagaj', 'ek bagaj ücreti', 'bagaj aşım',
                                    'overweight baggage', 'weight exceeded'],
                
                # Sports Categories (spesifik spor terimleri)
                'sports_hockey': ['hokey', 'hockey', 'paten', 'skates', 'stick', 'puck'],
                'sports_golf': ['golf', 'sopa', 'club', 'tee', 'green'],
                'sports_bicycle': ['bisiklet', 'bicycle', 'gidon', 'handlebar', 'sele', 'saddle'],
                'sports_skiing': ['kayak', 'ski', 'snowboard', 'pist', 'slope', 'kar', 'snow'],
                'sports_diving': ['dalış', 'diving', 'scuba', 'regulator', 'maske', 'mask'],
                'sports_surfing': ['sörf', 'surf', 'surfboard', 'dalga', 'wave'],
                'sports_mountaineering': ['dağcılık', 'mountaineering', 'tırmanış', 'climbing', 'buz', 'ice'],
                'sports_archery': ['okçuluk', 'archery', 'yay', 'bow', 'ok', 'arrow'],
                'sports_fishing': ['balık', 'fishing', 'olta', 'rod', 'makara', 'reel'],
                'sports_hunting': ['avcılık', 'hunting', 'tüfek', 'rifle', 'av', 'game'],
                'sports_canoeing': ['kano', 'canoe', 'kürek', 'paddle', 'akarsu', 'river'],
                'sports_rafting': ['rafting', 'şişirilebilir', 'inflatable'],
                'sports_windsurfing': ['windsurf', 'rüzgar', 'wind', 'yelken', 'sail'],
                'sports_water_skiing': ['wakeboard', 'slalom', 'water'],
                'sports_bowling': ['bowling', 'lane', 'oyun', 'game'],
                'sports_tenting': ['çadır', 'tent', 'camping'],
                'sports_parachuting': ['paraşüt', 'parachute', 'skydiving', 'atlama', 'jump'],
                
                # Pets Categories
                'pets_cargo': ['kargo', 'cargo', 'hold', 'compartment', 'kafes', 'cage'],
                'pets_cabin': ['kabin', 'cabin', 'muhabbet', 'budgie', 'kanarya', 'canary'],
                'pets_service_animals': ['servis', 'service', 'rehber', 'guide', 'eğitim', 'training'],
                'pets_country_rules': ['ülke', 'country', 'aşı', 'vaccination', 'sertifika', 'certificate'],
                'pets_terms': ['şartlar', 'terms', 'koşullar', 'conditions', 'sedatif', 'sedative'],
                'pets_onboard': ['uçak', 'onboard', 'izinli', 'allowed'],
                
                # Other Categories
                'musical_instruments': ['müzik', 'music', 'enstrüman', 'instrument', 'çalgı', 'fragile'],
                'restrictions': ['yasak', 'forbidden', 'kısıtlı', 'restricted', 'çakmak', 'lighter'],
                'sports_equipment': ['spor', 'sports', 'ekipman', 'equipment'], # Genel spor terimi
                'pets': ['evcil', 'pet', 'hayvan', 'animal'] # Genel pet terimi
            },
            
            'pegasus': {
                'baggage_allowance': ['paket', 'package', 'light', 'saver', 'comfort'],
                'extra_services_pricing': ['TRY', 'EUR', 'SPEQ', 'AVIH', 'pricing'],
                'travelling_with_pets': ['PETC', 'seyahat', 'travel', 'evcil', 'pet'],
                'general_rules': ['kural', 'rules', 'genel', 'general', 'bolbol', 'puan', 'points']
            }
        }
        
        # Default kategoriler (airline belirtilmemişse)
        DEFAULT_CATEGORIES = {
            'baggage': ['bagaj', 'baggage', 'bavul', 'luggage', 'kilo', 'weight', 'allowance', 'checked', 'carry', 'cabin', 'el', 'çanta'],
            'pets': ['evcil', 'pet', 'hayvan', 'animal', 'köpek', 'dog', 'kedi', 'cat', 'kuş', 'bird'],
            'sports': ['spor', 'sports', 'ekipman', 'equipment', 'golf', 'bisiklet', 'bicycle', 'kayak', 'ski'],
            'music': ['müzik', 'music', 'enstrüman', 'instrument', 'gitar', 'guitar', 'piano'],
            'restrictions': ['kısıtlı', 'restricted', 'yasak', 'forbidden', 'çakmak', 'lighter', 'pil', 'battery'],
            'rules': ['kural', 'rules', 'şart', 'terms', 'koşul', 'conditions', 'bilet', 'ticket']
        }
        
        # Airline-specific category selection
        if airline_filter and airline_filter in AIRLINE_CATEGORIES:
            available_categories = AIRLINE_CATEGORIES[airline_filter]
            search_scope = f"airline-specific ({airline_filter})"
        else:
            available_categories = DEFAULT_CATEGORIES
            search_scope = "global"
        
        detected = []
        for category, keywords in available_categories.items():
            # Basit keyword matching
            if any(keyword in query_lower for keyword in keywords):
                detected.append(category)
        
        # En fazla max_categories döndür
        result = detected[:max_categories]
        
        if result:
            logger.debug(f"Query '{query[:50]}...' detected categories in {search_scope}: {result}")
        else:
            logger.debug(f"No categories detected in {search_scope} for query: '{query[:50]}...'")
        
        return result
    
    async def similarity_search(
        self, 
        query: str, 
        limit: int = 5,
        similarity_threshold: float = 0.3,
        airline_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        use_category_hint: bool = True
    ) -> List[Dict]:
        """Vector similarity search - Opsiyonel kategori hinting ile geliştirildi"""
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Kategori tespiti (airline-aware)
        category_hints = []
        if use_category_hint:
            category_hints = self.detect_query_categories(query, airline_filter, max_categories=2)
        
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
        
        # Source filter (mevcut)
        if source_filter:
            sql += f" AND source = ${len(params) + 1}"
            params.append(source_filter)
        
        # Kategori filtering (yeni - opsiyonel)
        if category_hints:
            category_conditions = []
            for hint in category_hints:
                category_conditions.append(f"source ILIKE '%{hint}%'")
            
            if category_conditions:
                sql += f" AND ({' OR '.join(category_conditions)})"
        
        # Similarity threshold and ordering
        similarity_param_idx = len(params) + 1
        sql += f" AND (1 - (embedding <=> $1::vector)) >= ${similarity_param_idx}"
        params.append(similarity_threshold)
        
        sql += " ORDER BY embedding <=> $1::vector LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
        async with self.db_pool.acquire() as conn:
            results = await conn.fetch(sql, *params)
            
            # Convert to dict and format
            formatted_results = []
            for row in results:
                result_dict = dict(row)
                # Format dates
                if result_dict.get('created_at'):
                    result_dict['created_at'] = result_dict['created_at'].isoformat()
                if result_dict.get('updated_at'):
                    result_dict['updated_at'] = result_dict['updated_at'].isoformat()
                formatted_results.append(result_dict)
            
            # Enhanced logging
            filter_info = f"(airline: {airline_filter or 'all'})"
            if category_hints:
                filter_info += f" (categories: {', '.join(category_hints)})"
            
            logger.info(f"Vector search for '{query[:50]}...': {len(formatted_results)} results {filter_info}")
            return formatted_results
    
    async def preference_aware_search(
        self,
        query: str,
        airline_preference: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.3,
        boost_factor: float = 1.2
    ) -> List[Dict]:
        """Preference search with optional category enhancement"""
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Kategori detection (airline-aware)
        category_hints = self.detect_query_categories(query, airline_preference, max_categories=2)
        
        if airline_preference:
            # Single query with conditional scoring in PostgreSQL
            sql = f"""
                SELECT 
                    id, airline, source, content, quality_score, 
                    created_at, updated_at, url,
                    CASE WHEN airline = $2 
                         THEN {boost_factor} * (1 - (embedding <=> $1::vector))
                         ELSE (1 - (embedding <=> $1::vector))
                    END as similarity_score,
                    (1 - (embedding <=> $1::vector)) as original_similarity_score,
                    (airline = $2) as preference_boost
                FROM policy
                WHERE embedding IS NOT NULL
            """
            
            params = [
                "[" + ",".join(map(str, query_embedding.tolist())) + "]",
                airline_preference
            ]
            
            # Add category filtering if detected
            if category_hints:
                category_conditions = []
                for hint in category_hints:
                    category_conditions.append(f"source ILIKE '%{hint}%'")
                sql += f" AND ({' OR '.join(category_conditions)})"
            
            sql += f" AND (1 - (embedding <=> $1::vector)) >= ${len(params) + 1}"
            params.append(similarity_threshold)
            
            sql += f" ORDER BY similarity_score DESC LIMIT ${len(params) + 1}"
            params.append(limit)
            
        else:
            # Regular search without preference
            sql = """
                SELECT 
                    id, airline, source, content, quality_score, 
                    created_at, updated_at, url,
                    (1 - (embedding <=> $1::vector)) as similarity_score,
                    (1 - (embedding <=> $1::vector)) as original_similarity_score,
                    false as preference_boost
                FROM policy
                WHERE embedding IS NOT NULL
            """
            
            params = ["[" + ",".join(map(str, query_embedding.tolist())) + "]"]
            
            # Add category filtering
            if category_hints:
                category_conditions = []
                for hint in category_hints:
                    category_conditions.append(f"source ILIKE '%{hint}%'")
                sql += f" AND ({' OR '.join(category_conditions)})"
            
            sql += f" AND (1 - (embedding <=> $1::vector)) >= ${len(params) + 1}"
            params.append(similarity_threshold)
            
            sql += f" ORDER BY similarity_score DESC LIMIT ${len(params) + 1}"
            params.append(limit)
        
        async with self.db_pool.acquire() as conn:
            results = await conn.fetch(sql, *params)
            
            # Convert to dict and format
            formatted_results = []
            for row in results:
                result_dict = dict(row)
                # Format dates
                if result_dict.get('created_at'):
                    result_dict['created_at'] = result_dict['created_at'].isoformat()
                if result_dict.get('updated_at'):
                    result_dict['updated_at'] = result_dict['updated_at'].isoformat()
                formatted_results.append(result_dict)
            
            # Enhanced logging with category info
            if airline_preference:
                boosted_count = len([r for r in formatted_results if r.get('preference_boost')])
                category_info = f" | Categories: {', '.join(category_hints)}" if category_hints else ""
                logger.info(f"Preference search: {len(formatted_results)} results, {boosted_count} boosted{category_info}")
            else:
                category_info = f" | Categories: {', '.join(category_hints)}" if category_hints else ""
                logger.info(f"Regular search: {len(formatted_results)} results{category_info}")
            
            return formatted_results
        
    async def get_airline_stats(self) -> Dict:
        """Get airline distribution statistics"""
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
            
            return {
                "airline_breakdown": [dict(row) for row in stats],
                "total_airlines": len(stats)
            }
    
    async def get_embedding_stats(self) -> Dict:
        """Embedding istatistikleri"""
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
                "category_enhancement": {
                    "enabled": True,
                    "supported_categories": ["baggage", "pets", "sports", "music", "restrictions", "rules"],
                    "multilingual": True
                }
            }
    
    # Test function for category detection
    async def test_category_detection(self, test_queries: List[str], airline_filter) -> Dict:
        """Test category detection with sample queries"""
        results = {}
        for query in test_queries:
            detected = self.detect_query_categories(query, airline_filter)
            results[query] = detected
        return results