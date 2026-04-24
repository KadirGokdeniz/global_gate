"""
LLM base — tum LLM provider'larinin paylastigi mantik tek dosyada.

Icerdikleri:
    - Prompt'lar (TR/EN)        — system, CoT, answer instruction, error messages
    - CoT parser                — [REASONING] / [ANSWER] ayristirmasi + fallbacks
    - Context builder           — retrieved docs -> prompt context
    - Pricing tables            — cost estimation
    - BaseLLMService            — generate_rag_response sablonu

Provider dosyalari (claude_service.py, openai_service.py) sadece
API-spesifik cagri kodunu icerir. Prompt degisikligi icin BU dosya.

PROMPT_VERSION artirildigi zaman response metadata'sinda donuyor —
evaluation harness'inda v4 vs v5 karsilastirmasi yapabilmek icin.
"""
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# PROMPT VERSION
# ════════════════════════════════════════════════════════════════════
# v4: TON KURALI eklendi — burokratik giris engeli
# v3: Citation format spec'i netlestirildi
# v2: Few-shot ornekler eklendi
# v1: Baseline

PROMPT_VERSION = "v4"


# ════════════════════════════════════════════════════════════════════
# PROMPTS — TR / EN
# ════════════════════════════════════════════════════════════════════

LANGUAGE_PROMPTS = {
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

        "context_prefix":  "Havayolu Politika Belgeleri:",
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

Cevap:""",

        "error_generic":     "Uzgunum, su anda talebinizi isleyemiyorum. Lutfen tekrar deneyin.",
        "error_with_detail": "Uzgunum, su anda talebinizi isleyemiyorum. Lutfen tekrar deneyin. Hata: {error}",
        "no_docs":           "Ilgili politika belgesi bulunamadi.",
        "unknown_airline":   "Bilinmiyor",
        "unknown_source":    "Bilinmiyor",
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

        "context_prefix":  "Airline Policy Documents:",
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

Answer:""",

        "error_generic":     "I apologize, but I'm having trouble processing your request right now.",
        "error_with_detail": "I apologize, but I'm having trouble processing your request right now. Error: {error}",
        "no_docs":           "No specific policy documents found.",
        "unknown_airline":   "Unknown",
        "unknown_source":    "Unknown",
    },
}


def get_prompt_config(language: str) -> dict:
    """Return prompt config for a language, falling back to English."""
    return LANGUAGE_PROMPTS.get(language, LANGUAGE_PROMPTS["en"])


# ════════════════════════════════════════════════════════════════════
# PRICING TABLES
# ════════════════════════════════════════════════════════════════════
# OpenAI fiyatlari PER 1K token (eski format). Claude fiyatlari PER 1M token.
# Fiyatlar snapshot — production'da kullanmadan once dogrulaayin:
#   OpenAI:    https://openai.com/api/pricing
#   Anthropic: https://www.anthropic.com/pricing

OPENAI_PRICING = {
    # per 1K tokens
    "gpt-3.5-turbo":  {"input": 0.0005,  "output": 0.0015},
    "gpt-4":          {"input": 0.03,    "output": 0.06},
    "gpt-4-turbo":    {"input": 0.01,    "output": 0.03},
    "gpt-4o":         {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":    {"input": 0.00015, "output": 0.0006},
}

CLAUDE_PRICING = {
    # per 1M tokens
    "claude-opus-4-7":            {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6":          {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001":  {"input": 1.00, "output": 5.00},
}

OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
CLAUDE_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def estimate_openai_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """OpenAI usage -> approximate $ cost (per 1K token rates)."""
    if model not in OPENAI_PRICING:
        logger.warning(f"No OpenAI pricing for {model}, falling back to {OPENAI_DEFAULT_MODEL}")
        model = OPENAI_DEFAULT_MODEL
    rates = OPENAI_PRICING[model]
    return (prompt_tokens / 1_000) * rates["input"] + (completion_tokens / 1_000) * rates["output"]


def estimate_claude_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Claude usage -> approximate $ cost (per 1M token rates)."""
    if model not in CLAUDE_PRICING:
        logger.warning(f"No Claude pricing for {model}, falling back to {CLAUDE_DEFAULT_MODEL}")
        model = CLAUDE_DEFAULT_MODEL
    rates = CLAUDE_PRICING[model]
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


# ════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER — retrieved docs -> prompt context
# ════════════════════════════════════════════════════════════════════

MAX_DOCS_IN_CONTEXT = 5
MAX_CONTENT_CHARS   = 3500


def build_context(retrieved_docs: List[Dict], language: str) -> Tuple[str, bool]:
    """
    Build the context block to inject into the user prompt.

    Returns:
        (context_string, context_used). context_used=False when no docs.
    """
    lang = get_prompt_config(language)
    if not retrieved_docs:
        return lang["no_docs"], False

    parts = []
    for i, doc in enumerate(retrieved_docs[:MAX_DOCS_IN_CONTEXT], start=1):
        airline = doc.get("airline", lang["unknown_airline"])
        source  = doc.get("source",  lang["unknown_source"])
        content = doc.get("content", "")[:MAX_CONTENT_CHARS]

        if language == "tr":
            parts.append(f"Belge {i} (Kaynak: {source} - Havayolu: {airline}):\n{content}...")
        else:
            parts.append(f"Document {i} (Source: {source} - Airline: {airline}):\n{content}...")

    return "\n\n".join(parts), True


def build_user_prompt(context: str, question: str, language: str, use_cot: bool) -> str:
    """Assemble the full user-turn prompt."""
    lang = get_prompt_config(language)
    instruction = lang["cot_instruction"] if use_cot else lang["answer_instruction"]
    return (
        f"{lang['context_prefix']}\n{context}\n\n"
        f"{lang['question_prefix']} {question}\n\n"
        f"{instruction}"
    )


# ════════════════════════════════════════════════════════════════════
# CoT RESPONSE PARSER
# ════════════════════════════════════════════════════════════════════
# LLM ciktisindan [REASONING] ve [ANSWER] bolumlerini ayir.
# Fallback: marker yoksa regex pattern'leri dene, son care tum cevabi dondur.
#
# TODO: XML tags (Anthropic) veya JSON mode (OpenAI) ile structured output'a
# gecince bu fallback zinciri sadelecek.

_ANSWER_PATTERNS = [
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

_NUMBERED_KEYWORDS = ['response', 'answer', 'yanit', 'therefore', 'so,', 'in summary']


def parse_cot_response(full_response: str) -> Tuple[Optional[str], str]:
    """
    Parse CoT response into (reasoning, final_answer).
    reasoning may be None if no markers were found.
    """
    if not full_response:
        return None, ""

    # Primary: explicit markers
    if "[REASONING]" in full_response and "[ANSWER]" in full_response:
        try:
            r_start = full_response.find("[REASONING]") + len("[REASONING]")
            r_end   = full_response.find("[ANSWER]")
            reasoning = full_response[r_start:r_end].strip()
            final_answer = full_response[r_end + len("[ANSWER]"):].strip()
            logger.debug("CoT parsed with [REASONING]/[ANSWER] markers")
            return reasoning, final_answer
        except Exception as e:
            logger.warning(f"Primary marker parsing failed: {e}")

    # Fallback 1: regex patterns
    for pattern in _ANSWER_PATTERNS:
        match = re.search(pattern, full_response, re.DOTALL | re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            if len(answer) > 20:
                reasoning = full_response[:match.start()].strip()
                logger.debug(f"CoT parsed with regex: {pattern[:30]}...")
                return reasoning, answer

    # Fallback 2: numbered sections
    if re.search(r'^\s*\d+\.', full_response, re.MULTILINE):
        sections = re.split(r'\n(?=\d+\.)', full_response)
        if len(sections) >= 2 and any(kw in sections[-1].lower() for kw in _NUMBERED_KEYWORDS):
            reasoning = '\n'.join(sections[:-1]).strip()
            answer = re.sub(
                r'^\d+\.\s*(?:Response|Answer|Yanit)[:\s]*', '',
                sections[-1], flags=re.IGNORECASE
            ).strip()
            logger.debug("CoT parsed with numbered section detection")
            return reasoning, answer

    # Last resort
    logger.warning("No CoT markers found, returning full response as answer")
    return None, full_response


# ════════════════════════════════════════════════════════════════════
# BASE LLM SERVICE — abstract template
# ════════════════════════════════════════════════════════════════════

class LLMCallResult:
    """Normalized result from a provider's raw API call."""
    __slots__ = ("text", "input_tokens", "output_tokens", "model")

    def __init__(self, text: str, input_tokens: int, output_tokens: int, model: str):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model


class BaseLLMService(ABC):
    """
    Abstract LLM service — sablon: context -> prompt -> LLM -> (CoT) -> response dict.

    Subclass eden provider'lar sadece sunu implement eder:
        _init_client()          — SDK client setup
        _call_llm(...)          — provider API cagrisi
        _estimate_cost(...)     — pricing module'u kullanarak maliyet
        get_available_models()
        get_default_model()
        test_connection()
    """

    default_max_tokens:  int   = 800
    default_temperature: float = 0.2

    def __init__(self):
        self.client = None
        self._init_client()

    # ---- provider must implement ----

    @abstractmethod
    def _init_client(self) -> None: ...

    @abstractmethod
    def _call_llm(
        self, system_instruction: str, user_prompt: str,
        model: str, max_tokens: int, temperature: float,
    ) -> LLMCallResult: ...

    @abstractmethod
    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float: ...

    @abstractmethod
    def get_available_models(self) -> List[str]: ...

    @abstractmethod
    def get_default_model(self) -> str: ...

    @abstractmethod
    def test_connection(self) -> Dict: ...

    # ---- shared logic ----

    def is_available(self) -> bool:
        return self.client is not None

    def generate_rag_response(
        self,
        retrieved_docs: List[Dict],
        question: str,
        model: Optional[str] = None,
        language: str = "en",
        use_cot: bool = False,
    ) -> Dict:
        """RAG response — sablon, tum provider'lar icin ayni."""
        if not self.is_available():
            return self._error_response(language, "Client not initialized")

        try:
            model_to_use = model or self.get_default_model()
            lang = get_prompt_config(language)

            system_instruction = (
                lang["system_instruction_cot"] if use_cot
                else lang["system_instruction"]
            )

            context, context_used = build_context(retrieved_docs, language)
            user_prompt = build_user_prompt(context, question, language, use_cot)

            result = self._call_llm(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                model=model_to_use,
                max_tokens=self.default_max_tokens,
                temperature=self.default_temperature,
            )

            if not result.text:
                return self._error_response(language, "Empty response from API")

            if use_cot:
                reasoning, final_answer = parse_cot_response(result.text)
            else:
                reasoning, final_answer = None, result.text

            usage_info = {
                "input_tokens":   result.input_tokens,
                "output_tokens":  result.output_tokens,
                "total_tokens":   result.input_tokens + result.output_tokens,
                "estimated_cost": self._estimate_cost(
                    result.input_tokens, result.output_tokens, model_to_use
                ),
            }

            return {
                "success":        True,
                "answer":         final_answer,
                "reasoning":      reasoning,
                "model_used":     model_to_use,
                "language":       language,
                "context_used":   context_used,
                "cot_enabled":    use_cot,
                "prompt_version": PROMPT_VERSION,
                "usage":          usage_info,
            }

        except Exception as e:
            logger.error(f"{self.__class__.__name__} RAG generation error: {e}")
            return self._error_response(language, str(e))

    def _error_response(self, language: str, error_msg: str = "") -> Dict:
        """Language-aware error response envelope."""
        lang = get_prompt_config(language)
        user_message = (
            lang["error_with_detail"].format(error=error_msg) if error_msg
            else lang["error_generic"]
        )
        return {
            "success":        False,
            "error":          error_msg,
            "answer":         user_message,
            "reasoning":      None,
            "model_used":     "none",
            "language":       language,
            "context_used":   False,
            "cot_enabled":    False,
            "prompt_version": PROMPT_VERSION,
            "usage":          {},
        }