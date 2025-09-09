# claude_service.py - FIXED VERSION

import anthropic
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
import os

load_dotenv()
logger = logging.getLogger(__name__)

class ClaudeService:
    """Claude service for RAG responses - FIXED VERSION"""
    
    def __init__(self):
        """Initialize Claude client"""
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found - Claude features will be limited")
            self.client = None
        else:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info("✅ Claude service initialized successfully")
            except Exception as e:
                logger.error(f"❌ Claude initialization failed: {e}")
                self.client = None
        
        # Configuration
        self.default_model = 'claude-3-haiku-20240307'
        self.max_tokens = 400
        self.temperature = 0.2

        # FIXED: Better Turkish prompts
        self.LANGUAGE_PROMPTS = {
            "tr": {
                "system_instruction": """Sen uzman bir havayolu müşteri hizmetleri asistanısın. 
MUTLAKA Türkçe yanıt ver. Doğal, akıcı ve anlaşılır Türkçe kullan.
Havayolu politikaları konusunda doğru ve faydalı bilgiler ver.
Verilen politika belgeleri İngilizce olsa bile, yanıtını kesinlikle Türkçe yap.
Müşteriye saygılı ve yardımsever bir şekilde yaklaş.""",
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

    def test_connection(self) -> Dict:
        """Test Claude API connection"""
        if not self.client:
            return {
                "success": False,
                "message": "Claude client not initialized (missing API key)"
            }
        
        try:
            # Simple test call
            response = self.client.messages.create(
                model=self.default_model,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return {
                "success": True,
                "message": "Claude API connection successful",
                "model": self.default_model
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Claude API error: {str(e)}"
            }

    def generate_rag_response(self, retrieved_docs: List[Dict], question: str,
                              model: str = None, language: str = "en") -> Dict:
        """Generate RAG response using Claude - FIXED VERSION"""
        
        if not self.client:
            return self._get_error_response(language)
        
        try:
            # Use provided model or default
            model_to_use = model or self.default_model

            lang_config = self.LANGUAGE_PROMPTS.get(language, self.LANGUAGE_PROMPTS["en"])
            
            # Build context from retrieved documents
            if retrieved_docs:
                context_parts = []
                for i, doc in enumerate(retrieved_docs[:3], 1):  # Top 3 docs
                    airline_info = doc.get('airline', 'Bilinmiyor' if language == 'tr' else 'Unknown')
                    source = doc.get('source', 'Bilinmiyor' if language == 'tr' else 'Unknown')
                    content = doc.get('content', '')[:600]  # More content for Claude
                    
                    if language == 'tr':
                        context_parts.append(f"""
Belge {i} (Kaynak: {source} - Havayolu: {airline_info}):
{content}...
""")
                    else:
                        context_parts.append(f"""
Document {i} (Source: {source} - Airline: {airline_info}):
{content}...
""")
                        
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
            user_prompt = f"""{lang_config['context_prefix']}
{context}

{lang_config['question_prefix']} {question}

{lang_config['instruction']}"""
            
            # FIXED: Use correct Claude API
            response = self.client.messages.create(
                model=model_to_use,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=lang_config["system_instruction"],
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract usage information
            usage_info = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "estimated_cost": self._estimate_cost(response.usage, model_to_use)
            }
            
            return {
                "success": True,
                "answer": response.content[0].text,  # FIXED: Correct Claude response format
                "model_used": model_to_use,
                "language": language,
                "context_used": context_used,
                "usage": usage_info
            }
            
        except Exception as e:
            logger.error(f"❌ Claude RAG generation error: {e}")
            return self._get_error_response(language, str(e))
    
    def get_available_models(self) -> List[str]:
        """Get list of Claude models with CORRECT API names (ordered by capability)"""
        return [
            "claude-3-haiku-20240307",      # Claude Haiku 3 - Confirmed working
            "claude-3-5-haiku-20241022",    # Claude Haiku 3.5 - Fastest  
            "claude-3-7-sonnet-20250219",   # Claude Sonnet 3.7 - High performance
            "claude-sonnet-4-20250514",     # Claude Sonnet 4 - High performance
            "claude-opus-4-20250514",       # Claude Opus 4 - Previous flagship
            "claude-opus-4-1-20250805"      # Claude Opus 4.1 - Most capable
        ]

    def _estimate_cost(self, usage, model: str) -> float:
        """Estimate cost based on token usage with UPDATED pricing"""
        
        # Updated Claude pricing (2025 models)
        pricing = {
            "claude-opus-4-1-20250805": {"input": 18.00, "output": 90.00},   # Most expensive
            "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
            "claude-sonnet-4-20250514": {"input": 4.00, "output": 20.00},
            "claude-3-7-sonnet-20250219": {"input": 3.50, "output": 17.50},
            "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}        # Cheapest
        }
        
        # Use default pricing if model not found
        if model not in pricing:
            logger.warning(f"No pricing info for model {model}, using Haiku 3 pricing")
            model = "claude-3-haiku-20240307"  # Cheapest as fallback
        
        input_cost = (usage.input_tokens / 1000000) * pricing[model]["input"]
        output_cost = (usage.output_tokens / 1000000) * pricing[model]["output"]
        
        return input_cost + output_cost

_claude_service_instance = None

def get_claude_service():
    """Get Claude service instance (singleton pattern)"""
    global _claude_service_instance
    
    if _claude_service_instance is None:
        _claude_service_instance = ClaudeService()
    
    return _claude_service_instance