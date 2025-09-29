import openai
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
import os
from secrets_loader import SecretsLoader

loader = SecretsLoader()

logger = logging.getLogger(__name__)

class OpenAIService:
    """OpenAI service for RAG responses"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        api_key = loader.get_secret('openai_api_key', 'OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found - OpenAI features will be limited")
            self.client = None
        else:
            try:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("✅ OpenAI service initialized successfully")
            except Exception as e:
                logger.error(f"❌OpenAI initialization failed: {e}")
                self.client = None
        
        # Configuration
        self.default_model = 'gpt-3.5-turbo'
        self.max_tokens = 500
        self.temperature = 0.2

        self.LANGUAGE_PROMPTS = {
            "tr": {
                "system_instruction": """Sen profesyonel bir havayolu müşteri hizmetleri asistanısın. 
                Yanıtlarını SADECE Türkçe ver. Havayolu politakaları hakkında doğal, akıcı ve anlaşılır yanıtlar oluştur.
                Verilen belgeler İngilizce olsa bile, yanıtını Türkçe yap.""",
                "context_prefix": "Havayolu Politika Belgeleri:",
                "question_prefix": "Müşteri Sorusu:",
                "instruction": "Yukarıdaki politika belgelerine dayanarak soruyu Türkçe yanıtla:"
            },
            "en": {
                "system_instruction": """You are a professional airline customer service assistant.
                Answer ONLY in English. Provide clear and helpful responses about airline policies.""",
                "context_prefix": "Airline Policy Documents:",
                "question_prefix": "Customer Question:",
                "instruction": "Please answer the question based on the policy documents above:"
            }
        }

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
    def _get_error_response(self, language: str = "en", error_msg: str = "") -> Dict:
        """Get language-specific error response"""
        error_messages = {
            "tr": f"Üzgünüm, şu anda talebinizi işleyemiyorum. Lütfen tekrar deneyin. Hata: {error_msg}",
            "en": f"I apologize, but I'm having trouble processing your request right now. Error: {error_msg}"
        }
        
        return {
            "success": False,
            "error": error_msg,
            "answer": error_messages.get(language, error_messages["en"]),
            "model_used": "none",
            "language": language,
            "context_used": False,
            "usage": {}
        }

    def generate_rag_response(self, retrieved_docs: List[Dict], question: str, 
                              model: str = None, language:str="en") -> Dict:
        """Generate RAG response using OpenAI"""
        
        if not self.client:
            return self._get_error_response(language)
        
        try:
            # Use provided model or default
            model_to_use = model or self.default_model

            lang_config = self.LANGUAGE_PROMPTS.get(language, self.LANGUAGE_PROMPTS["en"])
            
            # Build context from retrieved documents
            if retrieved_docs:
                context_parts = []
                for i, doc in enumerate(retrieved_docs[:5], 1):  # Top 5 docs
                    airline_info = doc.get('airline', 'Bilinmeyen' if language == 'tr' else 'Unknown')
                    source = doc.get('source', 'Bilinmeyen' if language == 'tr' else 'Unknown')
                    content = doc.get('content', '')[:600]
                    
                    if language == 'tr':
                        context_parts.append(f"""Belge {i} (Kaynak: {source} - Havayolu: {airline_info}):
                                                {content}...""")
                    else:
                        context_parts.append(f"""Document {i} (Source: {source} - Airline: {airline_info}):
                                                {content}...""")
                        
                context = "\n".join(context_parts)
                context_used = True
            else:
                no_doc_messages = {
                    "tr": "İlgili politika belgesi bulunamadı.",
                    "en": "No specific policy documents found."
                }
                context = no_doc_messages.get(language, no_doc_messages["en"])
                context_used = False
            
            
            # Create user prompt
            user_prompt = f"""{context} {lang_config['question_prefix']} {question} {lang_config['instruction']}"""
                                        
            # Generate response
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": lang_config["system_instruction"]},
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
                "language" : language,
                "context_used": context_used,
                "usage": usage_info
            }
            
        except Exception as e:
            logger.error(f"❌ OpenAI RAG generation error: {e}")
            return self._get_error_response(language, str(e))
    
    def _estimate_cost(self, usage, model: str) -> float:
        """Estimate cost based on token usage"""
        
        # Pricing per 1K tokens
        pricing = {
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        }
        
        if model not in pricing:
            logger.warning(f"No pricing info for model {model}, using gpt-4-turbo pricing")
            model = "gpt-4-turbo"  # Default pricing
        
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