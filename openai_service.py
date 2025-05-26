import openai
from typing import List, Dict
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

class OpenAIService:
    """OpenAI service for RAG responses - FIXED VERSION"""
    
    def __init__(self):
        """Initialize OpenAI client with correct syntax"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # FIXED: Simple initialization without deprecated parameters
        self.client = openai.OpenAI(api_key=api_key)
        
        # Configuration
        self.default_model = 'gpt-3.5-turbo'
        self.max_tokens = 400
        self.temperature = 0.7
        
        logger.info(f"✅ OpenAI service initialized with model: {self.default_model}")
    
    def generate_rag_response(self, retrieved_docs: List[Dict], user_question: str, model: str = None) -> Dict:
        """Generate response using OpenAI with retrieved context"""
        model = model or self.default_model
        
        try:
            logger.info(f"🧠 Generating response for: {user_question[:50]}...")
            
            # Prepare context
            context = self._prepare_context(retrieved_docs)
            
            # Create prompts
            system_prompt = """You are a helpful Turkish Airlines customer service assistant. 
Answer questions about baggage policies based only on the provided context.
Be specific, accurate, and professional."""
            
            user_prompt = f"""Based on these Turkish Airlines baggage policies:

{context}

Customer Question: {user_question}

Please provide a helpful and accurate answer:"""
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract response
            answer = response.choices[0].message.content.strip()
            
            # Calculate cost
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            total_tokens = response.usage.total_tokens
            estimated_cost = (prompt_tokens * 0.0015 + completion_tokens * 0.002) / 1000
            
            logger.info(f"✅ Response generated. Tokens: {total_tokens}, Cost: ${estimated_cost:.4f}")
            
            return {
                "success": True,
                "answer": answer,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost": estimated_cost
                },
                "model_used": model,
                "context_used": len(retrieved_docs) > 0
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            return {
                "success": False,
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "error": str(e)
            }
    
    def _prepare_context(self, retrieved_docs: List[Dict], max_context_length: int = 1500) -> str:
        """Prepare context from retrieved documents"""
        if not retrieved_docs:
            return "No relevant Turkish Airlines baggage policies found."
        
        context_parts = []
        current_length = 0
        
        # Sort by similarity score
        sorted_docs = sorted(retrieved_docs, key=lambda x: x.get('similarity_score', 0), reverse=True)
        
        for doc in sorted_docs:
            content = doc.get('content', '')
            source = doc.get('source', 'unknown')
            similarity = doc.get('similarity_score', 0)
            
            if similarity > 0.25:
                doc_text = f"[Source: {source}]\n{content}\n"
                
                if current_length + len(doc_text) < max_context_length:
                    context_parts.append(doc_text)
                    current_length += len(doc_text)
                else:
                    break
        
        return "\n---\n".join(context_parts)
    
    def test_connection(self) -> Dict:
        """Test OpenAI connection"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, this is a connection test."}],
                max_tokens=10
            )
            
            return {
                "success": True,
                "message": "OpenAI connection successful",
                "model": "gpt-3.5-turbo"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"OpenAI connection failed: {str(e)}",
                "error": str(e)
            }

# Global instance
_openai_service = None

def get_openai_service() -> OpenAIService:
    """Get global OpenAI service instance"""
    global _openai_service
    if _openai_service is None:
        _openai_service = OpenAIService()
    return _openai_service