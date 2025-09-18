import asyncpg
from typing import List, Dict, Optional, Tuple
import numpy as np
from embedding_service import get_embedding_service
import logging

logger = logging.getLogger(__name__)

class VectorOperations:
    """PostgreSQL pgvector operations"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.embedding_service = get_embedding_service()
    
    async def embed_existing_policies(self, batch_size: int = 50) -> int:
        """Mevcut tüm policy'leri embed et"""
        logger.info("📄 Starting to embed existing policies...")
        
        async with self.db_pool.acquire() as conn:
            # Get policies without embeddings
            unembedded = await conn.fetch("""
                SELECT id, content 
                FROM policy
                WHERE embedding IS NULL
                ORDER BY id
            """)
            
            if not unembedded:
                logger.info("✅ All policies already have embeddings")
                return 0
            
            logger.info(f"📊 Found {len(unembedded)} policies to embed")
            
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
                logger.info(f"  ✅ Embedded batch {i//batch_size + 1}: {total_embedded}/{len(unembedded)}")
            
            logger.info(f"🎉 Successfully embedded {total_embedded} policies!")
            return total_embedded
    
    async def similarity_search(
        self, 
        query: str, 
        limit: int = 5,
        similarity_threshold: float = 0.3,
        airline_filter: Optional[str] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict]:
        """Vector similarity search - GÜNCELLENDİ: url ve updated_at eklendi"""
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Build SQL query - GÜNCELLENDİ: url, updated_at eklendi
        sql = """
            SELECT 
                id, airline, source, content, quality_score, 
                created_at, updated_at, url,
                1 - (embedding <=> $1::vector) as similarity_score
            FROM policy
            WHERE embedding IS NOT NULL
        """
        params = ["[" + ",".join(map(str, query_embedding.tolist())) + "]"]

        if airline_filter:
            sql += f" AND airline = ${len(params) + 1}"
            params.append(airline_filter)
        
        # Add source filter
        if source_filter:
            sql += f" AND source = ${len(params) + 1}"
            params.append(source_filter)
        
        # Add similarity threshold and ordering
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
            
            logger.info(f"🔍 Vector search for '{query}': {len(formatted_results)} results (airline: {airline_filter or 'all'})")
            return formatted_results
    
    async def preference_aware_search(
        self,
        query: str,
        airline_preference: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.3,
        boost_factor: float = 1.2
    ) -> List[Dict]:
        """Simplified preference search with database-level scoring - GÜNCELLENDİ"""
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)
        
        if airline_preference:
            # Single query with conditional scoring in PostgreSQL - GÜNCELLENDİ: url, updated_at eklendi
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
                AND (1 - (embedding <=> $1::vector)) >= $3
                ORDER BY similarity_score DESC
                LIMIT $4
            """
            
            params = [
                "[" + ",".join(map(str, query_embedding.tolist())) + "]",
                airline_preference,
                similarity_threshold,
                limit
            ]
            
        else:
            # Regular search without preference - GÜNCELLENDİ: url, updated_at eklendi
            sql = """
                SELECT 
                    id, airline, source, content, quality_score, 
                    created_at, updated_at, url,
                    (1 - (embedding <=> $1::vector)) as similarity_score,
                    (1 - (embedding <=> $1::vector)) as original_similarity_score,
                    false as preference_boost
                FROM policy
                WHERE embedding IS NOT NULL
                AND (1 - (embedding <=> $1::vector)) >= $2
                ORDER BY similarity_score DESC
                LIMIT $3
            """
            
            params = [
                "[" + ",".join(map(str, query_embedding.tolist())) + "]",
                similarity_threshold,
                limit
            ]
        
        async with self.db_pool.acquire() as conn:
            results = await conn.fetch(sql, *params)
            
            # Convert to dict and format
            formatted_results = []
            for row in results:
                result_dict = dict(row)
                # Format dates - GÜNCELLENDİ
                if result_dict.get('created_at'):
                    result_dict['created_at'] = result_dict['created_at'].isoformat()
                if result_dict.get('updated_at'):
                    result_dict['updated_at'] = result_dict['updated_at'].isoformat()
                formatted_results.append(result_dict)
            
            if airline_preference:
                boosted_count = len([r for r in formatted_results if r.get('preference_boost')])
                logger.info(f"Database-level preference search: {len(formatted_results)} results, {boosted_count} boosted")
            else:
                logger.info(f"Regular search: {len(formatted_results)} results")
            
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
                **airline_stats
            }
    
    async def similarity_search_fast(self, query: str, limit: int = 2):
        # İlk threshold'ı geçen 2 sonucu bulunca dur
        sql += " AND (1 - (embedding <=> $1::vector)) >= 0.6"  # Yüksek threshold
        sql += " ORDER BY embedding <=> $1::vector LIMIT 2"  # Küçük limit
    
    def optimize_query(query: str) -> str:
        # Stop words removal (basit)
        stop_words = ['the', 'is', 'at', 'which', 'on', 'a', 'an']
        words = [w for w in query.split() if w.lower() not in stop_words]
        return ' '.join(words[:10])  # Max 10 kelime
    
    CATEGORIES = {
        'baggage': [
            'baggage', 'luggage', 'suitcase', 'weight', 'allowance', 
            'checked', 'carry', 'excess', 'restrictions', 'limits'
        ],
        'pets': [
            'pet', 'dog', 'cat', 'animal', 'pets', 'cabin', 'cargo', 
            'service', 'onboard', 'travelling', 'travel'
        ],
        'sports': [
            'sports', 'equipment', 'golf', 'ski', 'skiing', 'snowboard',
            'bicycle', 'mountaineering', 'canoeing', 'archery', 'parachuting',
            'rafting', 'surfing', 'windsurfing', 'water_skiing', 'diving',
            'hockey', 'bowling', 'tenting', 'fishing', 'hunting'
        ],
        'musical_instruments': [
            'musical', 'instrument', 'instruments', 'guitar', 'piano',
            'violin', 'drums', 'music'
        ],
        'services': [
            'services', 'pricing', 'extra', 'additional', 'fees',
            'charges', 'cost', 'price', 'table'
        ],
        'general_rules': [
            'rules', 'general', 'regulations', 'terms', 'conditions',
            'policy', 'policies', 'info', 'flights'
        ]
    }

    def category_pre_filter(query: str) -> str:
        # Önce kategori belirle, sonra o kategoride ara
        for category, keywords in CATEGORIES.items():
            if any(keyword in query.lower() for keyword in keywords):
                return f"source LIKE '%{category}%'"
        return ""