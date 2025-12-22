# claude_service.py - FIXED VERSION v2

import anthropic
from typing import List, Dict, Optional
import logging
import re
from api.core.secrets_loader import SecretsLoader

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
                logger.info("Claude service initialized successfully")
            except Exception as e:
                logger.error(f"Claude initialization failed: {e}")
                self.client = None
        
        self.default_model = 'claude-3-haiku-20240307'
        self.max_tokens = 500
        self.temperature = 0.2

        # SEPARATE PROMPTS FOR COT AND NON-COT
        self.LANGUAGE_PROMPTS = {
            "tr": {
                # NON-COT: Direct answer, conversational tone, NO formatting
                "system_instruction": """Sen profesyonel bir havayolu musteri hizmetleri asistanisin.
Yanitlarini SADECE Turkce ver.

FORMAT KURALLARI (KESINLIKLE UYULMALI):
- Numarali liste KULLANMA (1. 2. 3. gibi)
- Madde isareti KULLANMA (- veya * gibi)
- Baslik KULLANMA
- SADECE dogal paragraflar halinde yaz
- Bir arkadasinla konusur gibi samimi ve akici yaz
- Bilgileri cumle icinde ver, liste yapma

ORNEK YANIT TARZI:
"Turkish Airlines'da kabin bagajiniz 8 kg'a kadar olabilir ve 55x40x23 cm boyutlarini gecmemeli. Ayrica yanınıza kucuk bir kisisel esya da alabilirsiniz."

YANLIS TARZI (KULLANMA):
"1. Kabin bagaji: 8 kg
2. Boyutlar: 55x40x23 cm
3. Kisisel esya: Evet"

Dogrudan, net ve anlasilir bir yanit ver.""",

                # COT: With reasoning (internal use)
                "system_instruction_cot": """Sen profesyonel bir havayolu musteri hizmetleri asistanisin.
Yanitlarini SADECE Turkce ver.
Verilen formati KESINLIKLE takip et.
[ANSWER] bolumunde numarali liste veya madde isareti KULLANMA, paragraf yaz.""",
                
                "context_prefix": "Havayolu Politika Belgeleri:",
                "question_prefix": "Musteri Sorusu:",
                
                "cot_instruction": """
KRITIK: Asagidaki formati KESINLIKLE kullan. Baska format KABUL EDILMEZ.

[REASONING]
- Soru ne soruyor: 
- Ilgili belgeler:
- Onemli bilgiler:

[ANSWER]
(Sadece kullaniciya gosterilecek nihai Turkce cevap. 
PARAGRAF FORMATINDA yaz. 
Numarali liste veya madde isareti KULLANMA.
Dogal ve akici bir dil kullan.)
""",
                "answer_instruction": "Soruyu DOGRUDAN ve PARAGRAF FORMATINDA yanitla. Liste veya madde isareti KULLANMA:"
            },
            "en": {
                # NON-COT: Direct answer, conversational tone, NO formatting
                "system_instruction": """You are a professional airline customer service assistant.
Answer ONLY in English.

FORMAT RULES (MUST BE FOLLOWED):
- Do NOT use numbered lists (1. 2. 3.)
- Do NOT use bullet points (- or *)
- Do NOT use headers
- Write ONLY in natural paragraphs
- Write in a friendly, conversational tone
- Include information within sentences, do NOT make lists

CORRECT STYLE EXAMPLE:
"Turkish Airlines allows cabin baggage up to 8 kg with dimensions not exceeding 55x40x23 cm. You can also bring a small personal item with you."

WRONG STYLE (DO NOT USE):
"1. Cabin baggage: 8 kg
2. Dimensions: 55x40x23 cm
3. Personal item: Yes"

Provide a direct, clear, and helpful response.""",

                # COT: With reasoning (internal use)
                "system_instruction_cot": """You are a professional airline customer service assistant.
Answer ONLY in English.
You MUST follow the exact format provided.
In the [ANSWER] section, do NOT use numbered lists or bullet points, write in paragraphs.""",
                
                "context_prefix": "Airline Policy Documents:",
                "question_prefix": "Customer Question:",
                
                "cot_instruction": """
CRITICAL: You MUST use EXACTLY this format. No other format is accepted.

[REASONING]
- What the question asks:
- Relevant documents:
- Key information:

[ANSWER]
(Only the final answer for the user.
Write in PARAGRAPH format.
Do NOT use numbered lists or bullet points.
Use natural, conversational language.)
""",
                "answer_instruction": "Answer the question DIRECTLY in PARAGRAPH format. Do NOT use lists or bullet points:"
            }
        }
    
    def test_connection(self) -> Dict:
        """Test Claude API connection"""
        if not self.client:
            return {
                "success": False,
                "message": "Claude client not initialized (missing API key)"
            }
        
        try:
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
        """Generate RAG response using Claude with optional Chain of Thought"""
        
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
                    airline_info = doc.get('airline', 'Bilinmiyor' if language == 'tr' else 'Unknown')
                    source = doc.get('source', 'Bilinmiyor' if language == 'tr' else 'Unknown')
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
            
            response = self.client.messages.create(
                model=model_to_use,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_instruction,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Safety check for response
            if not response.content or len(response.content) == 0:
                return self._get_error_response(language, "Empty response from Claude")
            
            full_response = response.content[0].text
            
            # Parse response - extract reasoning and answer if CoT used
            if use_cot:
                reasoning, final_answer = self._parse_cot_response(full_response, language)
            else:
                reasoning = None
                final_answer = full_response
            
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
                "reasoning": reasoning if use_cot else None,
                "model_used": model_to_use,
                "language": language,
                "context_used": context_used,
                "cot_enabled": use_cot,
                "usage": usage_info
            }
            
        except Exception as e:
            logger.error(f"Claude RAG generation error: {e}")
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