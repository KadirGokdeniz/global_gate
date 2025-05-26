import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class OpenAIService:
    def __init__(self):
        self.default_model = 'template-based'
        logger.info('✅ Template OpenAI service initialized')
    
    def test_connection(self):
        return {'success': True, 'message': 'Template service working', 'model': 'template-based'}
    
    def generate_rag_response(self, retrieved_docs, user_question, model=None):
        # Smart template responses using retrieved context
        if not retrieved_docs:
            answer = f"I understand you're asking about '{user_question}'. Based on Turkish Airlines policies, please contact customer service at +90 212 444 0 849 for specific information."
        else:
            # Use the most relevant document
            best_doc = max(retrieved_docs, key=lambda x: x.get('similarity_score', 0))
            content = best_doc.get('content', '')[:300]
            source = best_doc.get('source', 'policy')
            
            answer = f"According to Turkish Airlines {source} policy: {content}..."
            
            # Add specific guidance based on question type
            if any(word in user_question.lower() for word in ['weight', 'limit', 'kg']):
                answer += "\n\nFor exact weight limits and fees, please check your specific ticket class and destination."
            elif any(word in user_question.lower() for word in ['carry', 'cabin']):
                answer += "\n\nCarry-on restrictions may vary by aircraft type and route."
            elif any(word in user_question.lower() for word in ['excess', 'fee', 'cost']):
                answer += "\n\nExcess baggage fees depend on your route and fare type."
        
        return {
            'success': True,
            'answer': answer,
            'usage': {'estimated_cost': 0.0, 'total_tokens': len(answer.split())},
            'model_used': 'turkish-airlines-template',
            'context_used': len(retrieved_docs) > 0
        }

_openai_service = None
def get_openai_service():
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service
