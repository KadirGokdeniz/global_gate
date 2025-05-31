# VECTOR SEARCH QUALITY ENHANCEMENT PATCH
# Bu patch vector operations'da search quality'sini artırır

import asyncpg
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)

class EnhancedSearchQuality:
    """Vector search kalitesini artıran sınıf"""
    
    def __init__(self):
        # Domain-specific keyword weights for Turkish Airlines baggage
        self.keyword_boosts = {
            # High priority terms
            'weight': 1.3,
            'kg': 1.3, 
            'kilogram': 1.3,
            'size': 1.2,
            'dimension': 1.2,
            'cm': 1.2,
            'fee': 1.4,
            'cost': 1.4,
            'price': 1.4,
            'excess': 1.3,
            'additional': 1.2,
            
            # Baggage types
            'carry-on': 1.2,
            'checked': 1.2,
            'cabin': 1.1,
            'hand': 1.1,
            
            # Specific items
            'sports': 1.1,
            'equipment': 1.1,
            'musical': 1.1,
            'instrument': 1.1,
            'pet': 1.2,
            'animal': 1.2,
            
            # Airlines specific
            'turkish': 1.1,
            'airlines': 1.1,
            'policy': 1.1
        }
        
        # Content type priorities
        self.content_type_boosts = {
            'pricing_info': 1.3,
            'weight_info': 1.2,
            'dimension_info': 1.2,
            'restriction': 1.4,  # Restrictions are very important
            'structured_data': 1.1,
            'general_policy': 1.0
        }
    
    def calculate_enhanced_similarity_threshold(self, query: str, base_threshold: float = 0.3) -> float:
        """Query'ye göre adaptive similarity threshold hesapla"""
        
        query_lower = query.lower()
        
        # Specific queries need higher threshold
        specific_keywords = ['exact', 'specific', 'precisely', 'exactly', 'how much', 'what is']
        has_specific = any(keyword in query_lower for keyword in specific_keywords)
        
        # Question complexity
        word_count = len(query.split())
        
        if has_specific or word_count <= 4:
            # Short, specific questions need higher similarity
            return min(0.7, base_threshold + 0.2)
        elif word_count <= 8:
            # Medium questions
            return min(0.6, base_threshold + 0.1)
        else:
            # Complex questions can have lower threshold
            return max(0.4, base_threshold)
    
    def enhance_search_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Search sonuçlarını enhance et ve re-rank et"""
        
        if not results:
            return results
        
        query_lower = query.lower()
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Calculate enhanced score
            base_similarity = result.get('similarity_score', 0)
            content = result.get('content', '').lower()
            content_type = result.get('content_type', 'general_policy')
            
            # Keyword boost calculation
            keyword_boost = 1.0
            for keyword, boost in self.keyword_boosts.items():
                if keyword in query_lower and keyword in content:
                    keyword_boost *= boost
            
            # Content type boost
            content_boost = self.content_type_boosts.get(content_type, 1.0)
            
            # Quality boost
            quality_score = result.get('quality_score', 0)
            quality_boost = 1.0 + (quality_score * 0.1)  # Max 10% boost
            
            # Calculate final enhanced score
            enhanced_similarity = base_similarity * keyword_boost * content_boost * quality_boost
            enhanced_similarity = min(1.0, enhanced_similarity)  # Cap at 1.0
            
            # Update result
            enhanced_result['original_similarity'] = base_similarity
            enhanced_result['similarity_score'] = enhanced_similarity
            enhanced_result['boost_factors'] = {
                'keyword_boost': keyword_boost,
                'content_boost': content_boost,
                'quality_boost': quality_boost
            }
            
            enhanced_results.append(enhanced_result)
        
        # Re-sort by enhanced similarity
        enhanced_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.info(f"Enhanced {len(results)} search results with average boost: {np.mean([r['similarity_score']/r['original_similarity'] for r in enhanced_results]):.2f}x")
        
        return enhanced_results
    
    def filter_low_quality_results(self, results: List[Dict], min_quality: float = 0.3) -> List[Dict]:
        """Düşük kaliteli sonuçları filtrele"""
        
        filtered = []
        for result in results:
            quality_score = result.get('quality_score', 0)
            similarity_score = result.get('similarity_score', 0)
            
            # Combined quality check
            combined_score = (quality_score * 0.4) + (similarity_score * 0.6)
            
            if combined_score >= min_quality:
                result['combined_quality_score'] = combined_score
                filtered.append(result)
        
        logger.info(f"Filtered {len(results)} → {len(filtered)} results (quality threshold: {min_quality})")
        
        return filtered

# Enhanced vector operations patch
def patch_vector_operations():
    """Vector operations'a quality enhancement patch'i uygula"""
    
    try:
        from vector_operations import VectorOperations
        
        # Quality enhancer instance
        quality_enhancer = EnhancedSearchQuality()
        
        # Store original method
        original_similarity_search = VectorOperations.similarity_search
        
        async def enhanced_similarity_search(self, 
                                           query: str, 
                                           limit: int = 5,
                                           similarity_threshold: float = 0.3,
                                           source_filter: Optional[str] = None) -> List[Dict]:
            """Enhanced similarity search with quality improvements"""
            
            # Calculate adaptive threshold
            adaptive_threshold = quality_enhancer.calculate_enhanced_similarity_threshold(
                query, similarity_threshold
            )
            
            logger.info(f"🎯 Adaptive threshold: {similarity_threshold} → {adaptive_threshold} for query: '{query[:50]}...'")
            
            # Get more results than requested (for filtering)
            extended_limit = min(limit * 3, 20)  # Get 3x results but max 20
            
            # Call original method with adaptive threshold
            raw_results = await original_similarity_search(
                self, query, extended_limit, adaptive_threshold, source_filter
            )
            
            if not raw_results:
                logger.info("No results found with adaptive threshold, trying lower threshold")
                # Fallback with lower threshold
                fallback_threshold = max(0.2, adaptive_threshold - 0.2)
                raw_results = await original_similarity_search(
                    self, query, extended_limit, fallback_threshold, source_filter
                )
            
            # Enhance and re-rank results
            enhanced_results = quality_enhancer.enhance_search_results(raw_results, query)
            
            # Filter low quality
            quality_filtered = quality_enhancer.filter_low_quality_results(
                enhanced_results, min_quality=0.4
            )
            
            # Return top N results
            final_results = quality_filtered[:limit]
            
            logger.info(f"🚀 Enhanced search: {len(raw_results)} → {len(final_results)} results")
            
            return final_results
        
        # Apply patch
        VectorOperations.similarity_search = enhanced_similarity_search
        
        print("✅ Vector search quality enhancement patch applied successfully")
        return True
        
    except ImportError:
        print("❌ Could not import VectorOperations")
        return False
    except Exception as e:
        print(f"❌ Vector patch application failed: {e}")
        return False

# Utility function to get enhanced vector operations
def get_enhanced_vector_operations(db_pool):
    """Get vector operations with quality enhancement"""
    
    from vector_operations import VectorOperations
    
    # Apply patch
    patch_vector_operations()
    
    # Return enhanced instance
    return VectorOperations(db_pool)