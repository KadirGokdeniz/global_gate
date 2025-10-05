# claude_service.py - CoT Enhanced Version

import anthropic
from typing import List, Dict, Optional
import logging
from secrets_loader import SecretsLoader

loader = SecretsLoader()
logger = logging.getLogger(__name__)

class ClaudeService:
    """Claude service for RAG responses with Chain of Thought"""
    
    def __init__(self):
        """Initialize Claude client"""
        api_key = loader.get_secret('anthropic_api_key', 'ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found")
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
        self.max_tokens = 800  # CoT için daha fazla token
        self.temperature = 0.2

        # CoT-Enhanced Prompts
        self.LANGUAGE_PROMPTS = {
            "tr": {
                "system_instruction": """Sen uzman bir havayolu müşteri hizmetleri asistanısın. 
                MUTLAKA Türkçe yanıt ver. Doğal, akıcı ve anlaşılır Türkçe kullan.
                
                Soruları yanıtlarken şu adımları izle:
                1. Soruyu analiz et ve anahtar kavramları belirle
                2. Verilen politika belgelerinden ilgili bilgileri çıkar
                3. Bilgileri sentezleyerek tutarlı bir yanıt oluştur
                4. Yanıtını kullanıcıya sun
                
                Her adımda düşüncelerini açıkça belirt.""",
                
                "context_prefix": "Havayolu Politika Belgeleri:",
                "question_prefix": "Müşteri Sorusu:",
                
                "cot_instruction": """
                Lütfen aşağıdaki adımları izleyerek yanıt ver:
                
                <thinking>
                Adım 1: Soru Analizi
                - Sorunun ana konusu nedir?
                - Hangi politika alanıyla ilgili?
                - Hangi bilgilere ihtiyaç var?
                
                Adım 2: Belge İncelemesi
                - Hangi belgeler soruyla ilgili?
                - Belgelerden çıkarılacak anahtar bilgiler neler?
                - Çelişkili bilgiler var mı?
                
                Adım 3: Yanıt Sentezi
                - Bilgileri nasıl birleştireceğim?
                - Eksik bilgi var mı?
                - Kullanıcı için en faydalı format nedir?
                </thinking>
                
                <answer>
                [Buraya Türkçe yanıtını yaz]
                </answer>
                """
            },
            "en": {
                "system_instruction": """You are a professional airline customer service assistant.
                Answer ONLY in English. Provide clear and helpful responses about airline policies.
                
                Follow these steps when answering:
                1. Analyze the question and identify key concepts
                2. Extract relevant information from policy documents
                3. Synthesize information into a coherent response
                4. Present your answer to the user
                
                Think through each step explicitly.""",
                
                "context_prefix": "Airline Policy Documents:",
                "question_prefix": "Customer Question:",
                
                "cot_instruction": """
                Please answer following these steps:
                
                <thinking>
                Step 1: Question Analysis
                - What is the main topic of the question?
                - Which policy area is it related to?
                - What information is needed?
                
                Step 2: Document Review
                - Which documents are relevant to the question?
                - What are the key facts to extract from documents?
                - Are there any conflicting information?
                
                Step 3: Response Synthesis
                - How should I combine the information?
                - Is any information missing?
                - What's the most helpful format for the user?
                </thinking>
                
                <answer>
                [Write your English response here]
                </answer>
                """
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
            "usage": {},
            "reasoning": None
        }

    def generate_rag_response(self, retrieved_docs: List[Dict], question: str,
                              model: str = None, language: str = "en", 
                              use_cot: bool = True) -> Dict:
        """Generate RAG response using Claude with optional Chain of Thought"""
        
        if not self.client:
            return self._get_error_response(language)
        
        try:
            model_to_use = model or self.default_model
            lang_config = self.LANGUAGE_PROMPTS.get(language, self.LANGUAGE_PROMPTS["en"])
            
            # Build context from retrieved documents
            if retrieved_docs:
                context_parts = []
                for i, doc in enumerate(retrieved_docs[:5], 1):
                    airline_info = doc.get('airline', 'Bilinmiyor' if language == 'tr' else 'Unknown')
                    source = doc.get('source', 'Bilinmiyor' if language == 'tr' else 'Unknown')
                    content = doc.get('content', '')[:600]
                    
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
                
                Lütfen yukarıdaki politika belgelerine dayanarak soruyu yanıtla:"""
            
            response = self.client.messages.create(
                model=model_to_use,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=lang_config["system_instruction"],
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Parse response - extract thinking and answer if CoT used
            full_response = response.content[0].text
            reasoning = None
            final_answer = full_response
            
            if use_cot:
                # Extract thinking section if present
                if "<thinking>" in full_response and "</thinking>" in full_response:
                    start = full_response.find("<thinking>") + len("<thinking>")
                    end = full_response.find("</thinking>")
                    reasoning = full_response[start:end].strip()
                
                # Extract answer section if present
                if "<answer>" in full_response and "</answer>" in full_response:
                    start = full_response.find("<answer>") + len("<answer>")
                    end = full_response.find("</answer>")
                    final_answer = full_response[start:end].strip()
            
            # Extract usage information
            usage_info = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
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
            logger.error(f"❌ Claude RAG generation error: {e}")
            return self._get_error_response(language, str(e))
    
    def get_available_models(self) -> List[str]:
        """Get list of Claude models"""
        return [
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
            "claude-sonnet-4-20250514",
        ]

    def _estimate_cost(self, usage, model: str) -> float:
        """Estimate cost based on token usage"""
        pricing = {
            "claude-sonnet-4-20250514": {"input": 4.00, "output": 20.00},
            "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
            "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25}
        }
        
        if model not in pricing:
            logger.warning(f"No pricing info for model {model}")
            model = "claude-3-5-haiku-20241022"
        
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