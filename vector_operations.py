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
        logger.info("🔄 Starting to embed existing policies...")
        
        async with self.db_pool.acquire() as conn:
            # Get policies without embeddings
            unembedded = await conn.fetch("""
                SELECT id, content 
                FROM baggage_policies 
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
                        UPDATE baggage_policies 
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
        source_filter: Optional[str] = None
    ) -> List[Dict]:
        """Vector similarity search"""
        
        # Generate query embedding
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Build SQL query
        sql = """
            SELECT 
                id, source, content, quality_score, created_at,
                1 - (embedding <=> $1::vector) as similarity_score
            FROM baggage_policies 
            WHERE embedding IS NOT NULL
        """
        params = ["[" + ",".join(map(str, query_embedding.tolist())) + "]"]
        
        # Add source filter
        if source_filter:
            sql += " AND source = $2"
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
                if result_dict.get('created_at'):
                    result_dict['created_at'] = result_dict['created_at'].isoformat()
                formatted_results.append(result_dict)
            
            logger.info(f"🔍 Vector search for '{query}': {len(formatted_results)} results")
            return formatted_results
    
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
                FROM baggage_policies
            """)
            
            return dict(stats)