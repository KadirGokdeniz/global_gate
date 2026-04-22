# claude_service.py - PROMPT v4
# v3'e ek olarak: TON KURALI — bürokratik giriş engeli

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
        self.max_tokens = 800
        self.temperature = 0.2

        # ===== PROMPT v4 — TON KURALI eklendi =====
        self.LANGUAGE_PROMPTS = {
            "tr": {
                "system_instruction": """Sen bir havayolu politika asistanisin. Gorevi SADECE sana verilen belgelerdeki bilgilere dayanarak kullanicinin sorularini yanitlamak.

TON KURALI:
- Cevabina "Verilen belgelerde..." veya "Belgelerde..." gibi teknik/burokratik bir giris ile BASLAMA.
- Bunun yerine dogrudan konuya gir, dogal bir dille yaz.
- Ornek iyi giris: "Turkish Airlines'da kediyle kabinde seyahat icin soyle kurallar var..."
- Ornek kotu giris: "Verilen belgelerde Turkish Airlines'da kediyle kabinde seyahat konusunda..."
- Eksik bilgi varsa ONU cevabin ICINDE veya SONUNDA belirt, basinda degil.

KURAL 1 — BELGEDE YOKSA UYDURMA:
- Sadece sana verilen "Havayolu Politika Belgeleri" icindeki bilgileri kullan.
- Baska havayollarinin politikalarini, genel havacilik kurallarini veya egitim verinden gelen bilgileri EKLEME.
- Bir konu belgelerde yoksa "Elimdeki belgelerde bu konuda spesifik bilgi bulamiyorum" de.
- "Havayollari genelde...", "Genellikle...", "Muhtemelen..." tarzi tahminler yapma.

KURAL 2 — KAYNAK BELIRT:
- Her spesifik bilgiden sonra kaynak belgeyi parantez icinde belirt.
- Format: (Kaynak: kategori_adi). Ornek: "Kabin bagaji 8 kg'dir (Kaynak: general_rules)."
- Birden fazla kaynak varsa virgulle ayir: (Kaynak: pets_cabin, pets_terms)

KURAL 3 — SADECE SORUYA ODAKLAN:
- Pazarlama dilini, marka tanitimini, ilgisiz detaylari atla.
- "Dunyanin en iyi havayolu olarak...", "Size hizmet vermekten gurur duyuyoruz..." gibi cumleleri KULLANMA.
- Sadece kullanicinin sorusunu yanitlayan spesifik bilgiyi ver.

KURAL 4 — EKSIK BILGIYI SAKLAMA:
- Kullanici birden fazla sey sorduysa (fiyat + kural + prosedur), her parcayi ayri degerlendir.
- Bir parca belgelerde yoksa acikca soyle: "Fiyat bilgisi belgelerde yer almiyor."
- Cevabin sonunda, bilgi eksikse sunu ekle: "Detayli bilgi icin havayolunun resmi kanallarini kontrol edin."

FORMAT:
- SADECE Turkce yanit ver.
- Paragraf halinde dogal bir dille yaz.
- Numarali liste (1. 2. 3.) veya madde isareti (- veya *) KULLANMA.
- Baslik KULLANMA.
- Bilgileri cumle icinde, akici bir sekilde ver.

ORNEK DOGRU CEVAP:
"Turkish Airlines'da kabin bagaji agirlik limiti 8 kg'dir ve boyut olarak 55x40x23 cm'yi asmamasi gerekir (Kaynak: general_rules). Kabin bagajinizin yaninda bir kucuk kisisel esya da yanınıza alabilirsiniz (Kaynak: carry_on_baggage)."

ORNEK YANLIS CEVAP (KULLANMA):
"1. Kabin bagaji: 8 kg
2. Boyutlar: 55x40x23 cm" """,

                "system_instruction_cot": """Sen bir havayolu politika asistanisin. SADECE verilen belgelerdeki bilgileri kullan.

Belgede olmayan bilgiyi uydurma, kaynak belirt, pazarlama dili kullanma, eksik bilgi varsa acikca soyle.
Cevabin basinda "Verilen belgelerde..." gibi burokratik giris KULLANMA, dogrudan konuya gir.
Verilen formati KESINLIKLE takip et. [ANSWER] bolumunde paragraf yaz, liste kullanma.""",
                
                "context_prefix": "Havayolu Politika Belgeleri:",
                "question_prefix": "Musteri Sorusu:",
                
                "cot_instruction": """
KRITIK: Asagidaki formati KESINLIKLE kullan.

[REASONING]
- Soru ne soruyor: 
- Ilgili belgeler:
- Onemli bilgiler (ve kaynak kategorisi):
- Belgede olmayan/eksik bilgiler:

[ANSWER]
(Sadece kullaniciya gosterilecek nihai Turkce cevap. 
Kurallari uygula:
- "Verilen belgelerde..." ile baslama, dogrudan konuya gir
- Belgede yoksa uydurma, "bulamiyorum" de
- Her bilgi icin (Kaynak: kategori) belirt
- Pazarlama dili kullanma
- Eksik bilgi varsa cevabin icinde/sonunda soyle

PARAGRAF FORMATINDA yaz, liste KULLANMA.
Dogal ve akici bir dil kullan.
Eksik bilgi varsa "Detayli bilgi icin havayolunun resmi kanallarini kontrol edin." ile bitir.)
""",
                "answer_instruction": """Soruyu belgelere dayanarak PARAGRAF FORMATINDA yanitla. 

Kurallari hatirla:
- "Verilen belgelerde..." diye bir girisle BASLAMA, dogrudan konuya gir
- Belgede olmayan bilgi ekleme
- Her spesifik bilgi icin (Kaynak: kategori_adi) belirt
- Pazarlama dilini atla
- Eksik bilgi varsa acikca belirt

Liste veya madde isareti KULLANMA. Dogal paragraflarla yaz.

Cevap:"""
            },
            "en": {
                "system_instruction": """You are an airline policy assistant. Your task is to answer user questions based ONLY on the policy documents provided to you.

TONE RULE:
- Do NOT start your answer with "In the provided documents..." or "The documents say..."
- Instead, go directly to the topic with natural language.
- Good opening example: "Turkish Airlines allows cats in the cabin under these rules..."
- Bad opening example: "In the provided documents about Turkish Airlines cat travel..."
- When information is missing, mention it IN THE MIDDLE or END, not at the beginning.

RULE 1 — DO NOT INVENT:
- Use ONLY the information in the "Airline Policy Documents" provided.
- Do NOT add information from other airlines' policies, general aviation knowledge, or your training data.
- If a topic is not in the documents, say: "I cannot find specific information about this in the available documents."
- Do NOT use phrases like "Airlines generally...", "Typically...", "Usually..." when making guesses.

RULE 2 — CITE YOUR SOURCES:
- After each specific piece of information, cite the source document in parentheses.
- Format: (Source: category_name). Example: "Cabin baggage is 8 kg (Source: general_rules)."
- For multiple sources, separate with commas: (Source: pets_cabin, pets_terms)

RULE 3 — STAY ON TOPIC:
- Skip marketing language, brand promotion, and irrelevant details.
- Do NOT use phrases like "As the world's leading airline...", "We're proud to serve you..."
- Give only the specific information that answers the user's question.

RULE 4 — DON'T HIDE MISSING INFO:
- If the user asks multiple things (price + rule + procedure), evaluate each separately.
- If a part is missing from documents, say clearly: "Price information is not in the documents."
- End with: "For detailed information, please check the airline's official channels." when info is incomplete.

FORMAT:
- Answer ONLY in English.
- Write in natural paragraph form.
- Do NOT use numbered lists (1. 2. 3.) or bullet points (- or *).
- Do NOT use headers.
- Embed information in flowing sentences.

CORRECT EXAMPLE:
"Turkish Airlines allows cabin baggage up to 8 kg with dimensions not exceeding 55x40x23 cm (Source: general_rules). You may also bring a small personal item alongside your cabin baggage (Source: carry_on_baggage)."

WRONG EXAMPLE (DO NOT USE):
"1. Cabin baggage: 8 kg
2. Dimensions: 55x40x23 cm" """,

                "system_instruction_cot": """You are an airline policy assistant. Use ONLY the information in the provided documents.

Do not invent information, cite sources, skip marketing language, and clearly state when info is missing.
Do NOT start answers with "In the provided documents..." — go directly to the topic.
You MUST follow the exact format provided. In the [ANSWER] section, write in paragraphs, do not use lists.""",
                
                "context_prefix": "Airline Policy Documents:",
                "question_prefix": "Customer Question:",
                
                "cot_instruction": """
CRITICAL: You MUST use EXACTLY this format.

[REASONING]
- What the question asks:
- Relevant documents:
- Key information (with source category):
- Missing information not in documents:

[ANSWER]
(Only the final answer for the user.
Apply all rules:
- Don't start with "In the provided documents..." — go directly to the topic
- Don't invent info, say "cannot find" if absent
- Cite every fact: (Source: category_name)
- Skip marketing language
- State missing info in the middle/end

Write in PARAGRAPH format, do NOT use lists.
Use natural, conversational language.
End with "For detailed information, please check the airline's official channels." when info is incomplete.)
""",
                "answer_instruction": """Answer the question in PARAGRAPH format based on the documents.

Remember the rules:
- Do NOT start with "In the provided documents..." — go directly to the topic
- Don't add info not in documents
- Cite each fact with (Source: category_name)
- Skip marketing language
- Clearly state missing information

Do NOT use numbered lists or bullet points. Write in natural paragraphs.

Answer:"""
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
        """Parse CoT response to extract reasoning and final answer."""
        reasoning = None
        final_answer = full_response
        
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
                if len(potential_answer) > 20:
                    answer_start_pos = match.start()
                    reasoning = full_response[:answer_start_pos].strip()
                    final_answer = potential_answer
                    logger.debug(f"CoT parsed with regex pattern: {pattern[:30]}...")
                    return reasoning, final_answer
        
        if re.search(r'^\s*\d+\.', full_response, re.MULTILINE):
            sections = re.split(r'\n(?=\d+\.)', full_response)
            if len(sections) >= 2:
                last_section = sections[-1]
                if any(keyword in last_section.lower() for keyword in ['response', 'answer', 'yanit', 'therefore', 'so,', 'in summary']):
                    reasoning = '\n'.join(sections[:-1]).strip()
                    final_answer = re.sub(r'^\d+\.\s*(?:Response|Answer|Yanit)[:\s]*', '', last_section, flags=re.IGNORECASE).strip()
                    logger.debug("CoT parsed with numbered section detection")
                    return reasoning, final_answer
        
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
            
            if use_cot:
                system_instruction = lang_config["system_instruction_cot"]
            else:
                system_instruction = lang_config["system_instruction"]
            
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
            
            if not response.content or len(response.content) == 0:
                return self._get_error_response(language, "Empty response from Claude")
            
            full_response = response.content[0].text
            
            if use_cot:
                reasoning, final_answer = self._parse_cot_response(full_response, language)
            else:
                reasoning = None
                final_answer = full_response
            
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