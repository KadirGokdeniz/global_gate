import openai
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

class OpenAIService:
    """OpenAI service for RAG responses"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found - OpenAI features will be limited")
            self.client = None
        else:
            try:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("✅ OpenAI service initialized successfully")
            except Exception as e:
                logger.error(f"❌ OpenAI initialization failed: {e}")
                self.client = None
        
        # Configuration
        self.default_model = 'gpt-3.5-turbo'
        self.max_tokens = 400
        self.temperature = 0.2

    def test_connection(self) -> Dict:
        """Test OpenAI API connection"""
        if not self.client:
            return {
                "success": False,
                "message": "OpenAI client not initialized (missing API key)"
            }
        
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return {
                "success": True,
                "message": "OpenAI API connection successful",
                "model": self.default_model
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"OpenAI API error: {str(e)}"
            }

    def generate_rag_response(self, retrieved_docs: List[Dict], question: str, model: str = None) -> Dict:
        """Generate RAG response using OpenAI"""
        
        if not self.client:
            return {
                "success": False,
                "error": "OpenAI client not available",
                "answer": "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.",
                "model_used": "none",
                "context_used": False,
                "usage": {}
            }
        
        try:
            # Use provided model or default
            model_to_use = model or self.default_model
            
            # Build context from retrieved documents
            if retrieved_docs:
                context_parts = []
                for i, doc in enumerate(retrieved_docs[:3], 1):  # Top 3 docs
                    context_parts.append(f"""
                    Document {i} (Source: {doc.get('source', 'Unknown')}):
                    {doc.get('content', '')[:500]}...
                    """)
                context = "\n".join(context_parts)
                context_used = True
            else:
                context = "No specific policy documents found."
                context_used = False
            
            # Create system prompt
            system_prompt = """You are a helpful airlines customer service assistant. 
                               Answer questions about baggage policies clearly and accurately based on the provided context.
                               If no relevant context is provided, politely indicate that you don't have specific policy information."""
            
            # Create user prompt
            user_prompt = f"""Context from airlines policies:
                              {context}
                              Customer Question: {question}
                              Please provide a helpful and accurate answer based on the context above."""
            
            # Generate response
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Extract usage information
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens, 
                "total_tokens": response.usage.total_tokens,
                "estimated_cost": self._estimate_cost(response.usage, model_to_use)
            }
            
            return {
                "success": True,
                "answer": response.choices[0].message.content,
                "model_used": model_to_use,
                "context_used": context_used,
                "usage": usage_info
            }
            
        except Exception as e:
            logger.error(f"❌ OpenAI RAG generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": f"I apologize, but I'm having trouble processing your request right now. Error: {str(e)}",
                "model_used": model_to_use if 'model_to_use' in locals() else "unknown",
                "context_used": False,
                "usage": {}
            }
    
    def _estimate_cost(self, usage, model: str) -> float:
        """Estimate cost based on token usage"""
        
        # Pricing per 1K tokens
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03}, 
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0003} 
        }
        
        if model not in pricing:
            model = "gpt-3.5-turbo"  # Default pricing
        
        input_cost = (usage.prompt_tokens / 1000) * pricing[model]["input"]
        output_cost = (usage.completion_tokens / 1000) * pricing[model]["output"]
        
        return input_cost + output_cost

_openai_service_instance = None

def get_openai_service():
    """Get OpenAI service instance (singleton pattern)"""
    global _openai_service_instance
    
    if _openai_service_instance is None:
        _openai_service_instance = OpenAIService()
    
    return _openai_service_instance