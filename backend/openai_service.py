import openai
from typing import List, Dict, Optional
import logging
from secrets_loader import SecretsLoader

loader = SecretsLoader()
logger = logging.getLogger(__name__)

class OpenAIService:
    """OpenAI service for RAG responses with Chain of Thought"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        api_key = loader.get_secret('openai_api_key', 'OPENAI_API_KEY')
        if not api_key:
            logger.warning("OPENAI_API_KEY not found")
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

        # CoT-Enhanced Prompts
        self.LANGUAGE_PROMPTS = {
            "tr": {
                "system_instruction": """Sen profesyonel bir havayolu müşteri hizmetleri asistanısın. 
                Yanıtlarını SADECE Türkçe ver. Havayolu politikaları hakkında doğal, akıcı ve anlaşılır yanıtlar oluştur.
                
                Soruları yanıtlarken adım adım düşün:
                1. Soruyu anla ve anahtar noktaları belirle
                2. İlgili politika bilgilerini çıkar
                3. Bilgileri birleştirerek net bir yanıt oluştur
                
                Her adımda mantığını açıkla.""",
                
                "context_prefix": "Havayolu Politika Belgeleri:",
                "question_prefix": "Müşteri Sorusu:",
                
                "cot_instruction": """
                Şu adımları izleyerek yanıt ver:
                
                DÜŞÜNCE SÜRECİ:
                1. Soru Analizi: [Sorunun ne sorduğunu açıkla]
                2. Belge Taraması: [Hangi belgeler ilgili, anahtar bilgiler neler]
                3. Yanıt Yapılandırma: [Bilgileri nasıl birleştireceğini açıkla]
                
                SON YANITIM:
                [Kullanıcıya verilecek Türkçe yanıt]
                """
            },
            "en": {
                "system_instruction": """You are a professional airline customer service assistant.
                Answer ONLY in English. Provide clear and helpful responses about airline policies.
                
                When answering questions, think step by step:
                1. Understand the question and identify key points
                2. Extract relevant policy information
                3. Combine information into a clear response
                
                Explain your reasoning at each step.""",
                
                "context_prefix": "Airline Policy Documents:",
                "question_prefix": "Customer Question:",
                
                "cot_instruction": """
                Answer following these steps:
                
                THOUGHT PROCESS:
                1. Question Analysis: [Explain what the question is asking]
                2. Document Review: [Which documents are relevant, key information]
                3. Response Construction: [Explain how you'll combine the information]
                
                FINAL ANSWER:
                [English response for the user]
                """
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
            "usage": {},
            "reasoning": None
        }

    def generate_rag_response(self, retrieved_docs: List[Dict], question: str, 
                              model: str = None, language: str = "en",
                              use_cot: bool = False) -> Dict:
        """Generate RAG response using OpenAI with optional Chain of Thought"""
        
        if not self.client:
            return self._get_error_response(language)
        
        try:
            model_to_use = model or self.default_model
            lang_config = self.LANGUAGE_PROMPTS.get(language, self.LANGUAGE_PROMPTS["en"])
            
            # Build context from retrieved documents
            if retrieved_docs:
                context_parts = []
                for i, doc in enumerate(retrieved_docs[:5], 1):
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
            
            # Create user prompt - CoT or standard
            if use_cot:
                user_prompt = f"""{lang_config['context_prefix']}
                {context}
                
                {lang_config['question_prefix']} {question}
                
                {lang_config['cot_instruction']}"""
            else:
                # Standard prompt without CoT
                user_prompt = f"""{lang_config['context_prefix']}
                {context}
                
                {lang_config['question_prefix']} {question}
                
                Yukarıdaki politika belgelerine dayanarak soruyu yanıtla:"""
            
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
            
            # Parse response - extract reasoning and answer if CoT used
            full_response = response.choices[0].message.content
            reasoning = None
            final_answer = full_response
            
            if use_cot:
                # Try to separate reasoning from final answer
                if "DÜŞÜNCE SÜRECİ:" in full_response or "THOUGHT PROCESS:" in full_response:
                    # Find the reasoning section
                    reasoning_marker = "DÜŞÜNCE SÜRECİ:" if language == "tr" else "THOUGHT PROCESS:"
                    answer_marker = "SON YANITIM:" if language == "tr" else "FINAL ANSWER:"
                    
                    if reasoning_marker in full_response and answer_marker in full_response:
                        reasoning_start = full_response.find(reasoning_marker) + len(reasoning_marker)
                        reasoning_end = full_response.find(answer_marker)
                        reasoning = full_response[reasoning_start:reasoning_end].strip()
                        
                        answer_start = full_response.find(answer_marker) + len(answer_marker)
                        final_answer = full_response[answer_start:].strip()
            
            # Extract usage information
            usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens, 
                "total_tokens": response.usage.total_tokens,
                "estimated_cost": self._estimate_cost(response.usage, model_to_use)
            }
            
            return {
                "success": True,
                "answer": final_answer,
                "reasoning": reasoning if use_cot else None,  # CoT düşünce süreci
                "model_used": model_to_use,
                "language": language,
                "context_used": context_used,
                "cot_enabled": use_cot,
                "usage": usage_info
            }
            
        except Exception as e:
            logger.error(f"❌ OpenAI RAG generation error: {e}")
            return self._get_error_response(language, str(e))
    
    def _estimate_cost(self, usage, model: str) -> float:
        """Estimate cost based on token usage"""
        pricing = {
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        }
        
        if model not in pricing:
            logger.warning(f"No pricing info for model {model}")
            model = "gpt-4-turbo"
        
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