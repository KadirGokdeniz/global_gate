# openai_service.py - FIXED VERSION v2

import openai
from typing import List, Dict, Optional
import logging
import re
from api.core.secrets_loader import SecretsLoader

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
                logger.info("OpenAI service initialized successfully")
            except Exception as e:
                logger.error(f"OpenAI initialization failed: {e}")
                self.client = None
        
        # Configuration
        self.default_model = 'gpt-3.5-turbo'
        self.max_tokens = 500  
        self.temperature = 0.2

        # SEPARATE PROMPTS FOR COT AND NON-COT
        self.LANGUAGE_PROMPTS = {
            "tr": {
                # NON-COT: Direct answer, no thinking steps
                "system_instruction": """Sen profesyonel bir havayolu musteri hizmetleri asistanisin.
Yanitlarini SADECE Turkce ver.
Kullaniciya SADECE nihai cevabi ver.
Dusunce surecini, analiz adimlarini veya ara adimlari GOSTERME.
Dogrudan, net ve anlasilir bir yanit ver.""",

                # COT: With reasoning (internal use)
                "system_instruction_cot": """Sen profesyonel bir havayolu musteri hizmetleri asistanisin.
Yanitlarini SADECE Turkce ver.
Verilen formati KESINLIKLE takip et.""",
                
                "context_prefix": "Havayolu Politika Belgeleri:",
                "question_prefix": "Musteri Sorusu:",
                
                "cot_instruction": """
KRITIK: Asagidaki formati KESINLIKLE kullan. Baska format KABUL EDILMEZ.

[REASONING]
- Soru ne soruyor: 
- Ilgili belgeler:
- Onemli bilgiler:

[ANSWER]
(Sadece kullaniciya gosterilecek nihai Turkce cevap. Analiz veya adimlar YAZMA.)
""",
                "answer_instruction": "Soruyu DOGRUDAN yanitla. Dusunce adimlarini veya analizi GOSTERME:"
            },
            "en": {
                # NON-COT: Direct answer, no thinking steps
                "system_instruction": """You are a professional airline customer service assistant.
Answer ONLY in English.
Give the user ONLY the final answer.
Do NOT show your thinking process, analysis steps, or intermediate steps.
Provide a direct, clear, and helpful response.""",

                # COT: With reasoning (internal use)
                "system_instruction_cot": """You are a professional airline customer service assistant.
Answer ONLY in English.
You MUST follow the exact format provided.""",
                
                "context_prefix": "Airline Policy Documents:",
                "question_prefix": "Customer Question:",
                
                "cot_instruction": """
CRITICAL: You MUST use EXACTLY this format. No other format is accepted.

[REASONING]
- What the question asks:
- Relevant documents:
- Key information:

[ANSWER]
(Only the final answer for the user. Do NOT include analysis or steps here.)
""",
                "answer_instruction": "Answer the question DIRECTLY. Do NOT show thinking steps or analysis:"
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
            "tr": f"Uzgunum, su anda talebinizi isleyemiyorum. Lutfen tekrar deneyin. Hata: {error_msg}",
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

    def _parse_cot_response(self, full_response: str, language: str) -> tuple:
        """
        Parse CoT response to extract reasoning and final answer.
        Multiple fallback strategies for robust parsing.
        Returns: (reasoning, final_answer)
        """
        reasoning = None
        final_answer = full_response
        
        # Strategy 1: ASCII-safe markers [REASONING] and [ANSWER]
        if "[REASONING]" in full_response and "[ANSWER]" in full_response:
            try:
                reasoning_start = full_response.find("[REASONING]") + len("[REASONING]")
                reasoning_end = full_response.find("[ANSWER]")
                reasoning = full_response[reasoning_start:reasoning_end].strip()
                
                answer_start = full_response.find("[ANSWER]") + len("[ANSWER]")
                final_answer = full_response[answer_start:].strip()
                
                logger.debug("CoT parsed with [REASONING]/[ANSWER] markers")
                return reasoning, final_answer
            except Exception as e:
                logger.warning(f"Primary marker parsing failed: {e}")
        
        # Strategy 2: Regex for various "Answer" patterns
        answer_patterns = [
            r'\[ANSWER\](.*?)$',
            r'(?:^|\n)Answer[:\s]*\n?(.*?)$',
            r'(?:^|\n)ANSWER[:\s]*\n?(.*?)$',
            r'(?:^|\n)Final Answer[:\s]*\n?(.*?)$',
            r'(?:^|\n)FINAL ANSWER[:\s]*\n?(.*?)$',
            r'(?:^|\n)Response[:\s]*\n?(.*?)$',
            r'(?:^|\n)SON YANIT[:\s]*\n?(.*?)$',
            r'(?:^|\n)YANIT[:\s]*\n?(.*?)$',
            r'(?:^|\n)3\.\s*Response[:\s]*\n?(.*?)$',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)
            if match:
                potential_answer = match.group(1).strip()
                if len(potential_answer) > 20:  # Sanity check
                    # Everything before is reasoning
                    answer_start_pos = match.start()
                    reasoning = full_response[:answer_start_pos].strip()
                    final_answer = potential_answer
                    
                    logger.debug(f"CoT parsed with regex pattern: {pattern[:30]}...")
                    return reasoning, final_answer
        
        # Strategy 3: Look for numbered structure and extract last section
        if re.search(r'^\s*\d+\.', full_response, re.MULTILINE):
            # Find the last numbered section that looks like an answer
            sections = re.split(r'\n(?=\d+\.)', full_response)
            if len(sections) >= 2:
                # Check if last section contains answer-like content
                last_section = sections[-1]
                if any(keyword in last_section.lower() for keyword in ['response', 'answer', 'yanit', 'therefore', 'so,', 'in summary']):
                    reasoning = '\n'.join(sections[:-1]).strip()
                    # Remove the number prefix from answer
                    final_answer = re.sub(r'^\d+\.\s*(?:Response|Answer|Yanit)[:\s]*', '', last_section, flags=re.IGNORECASE).strip()
                    
                    logger.debug("CoT parsed with numbered section detection")
                    return reasoning, final_answer
        
        # Strategy 4: If nothing worked, log and return full response
        logger.warning("No CoT markers found, returning full response as answer")
        return None, full_response

    def generate_rag_response(self, retrieved_docs: List[Dict], question: str, 
                              model: str = None, language: str = "en",
                              use_cot: bool = False) -> Dict:
        """Generate RAG response using OpenAI with optional Chain of Thought"""
        
        if not self.client:
            return self._get_error_response(language)
        
        try:
            model_to_use = model or self.default_model
            lang_config = self.LANGUAGE_PROMPTS.get(language, self.LANGUAGE_PROMPTS["en"])
            
            # Select appropriate system instruction based on CoT mode
            if use_cot:
                system_instruction = lang_config["system_instruction_cot"]
            else:
                system_instruction = lang_config["system_instruction"]
            
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
                        
                context = "\n\n".join(context_parts)
                context_used = True
            else:
                no_doc_messages = {
                    "tr": "Ilgili politika belgesi bulunamadi.",
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
                user_prompt = f"""{lang_config['context_prefix']}
{context}

{lang_config['question_prefix']} {question}

{lang_config['answer_instruction']}"""
            
            # Generate response
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Safety check for response
            if not response.choices or not response.choices[0].message.content:
                return self._get_error_response(language, "Empty response from API")
            
            full_response = response.choices[0].message.content
            
            # Parse response - extract reasoning and answer if CoT used
            if use_cot:
                reasoning, final_answer = self._parse_cot_response(full_response, language)
            else:
                reasoning = None
                final_answer = full_response
            
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
                "reasoning": reasoning if use_cot else None,
                "model_used": model_to_use,
                "language": language,
                "context_used": context_used,
                "cot_enabled": use_cot,
                "usage": usage_info
            }
            
        except Exception as e:
            logger.error(f"OpenAI RAG generation error: {e}")
            return self._get_error_response(language, str(e))
    
    def _estimate_cost(self, usage, model: str) -> float:
        """Estimate cost based on token usage"""
        pricing = {
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
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