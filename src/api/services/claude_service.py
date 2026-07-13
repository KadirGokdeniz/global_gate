"""
Claude (Anthropic) LLM provider.

Tum ortak mantik (prompt'lar, CoT parsing, context, error shaping)
llm_base.py'de. Bu dosya sadece Anthropic SDK'sini cagiran ince bir
wrapper — ~80 satir.
"""
import logging
from typing import Dict, List

import anthropic

from api.core.secrets_loader import SecretsLoader
from api.services.llm_base import (
    BaseLLMService,
    LLMCallResult,
    CLAUDE_DEFAULT_MODEL,
    CLAUDE_PRICING,
    estimate_claude_cost,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Anthropic tarafindan emekliye ayrilmis (retired) modeller.
# Bu ID'lere yapilan istekler API'den 404 not_found_error doner.
# Frontend hala eski bir ID gonderse bile default'a cevirip 404'u onluyoruz.
# Kaynak: https://platform.claude.com/docs/en/about-claude/model-deprecations
# ─────────────────────────────────────────────────────────────────────
_RETIRED_CLAUDE_MODELS = {
    "claude-3-haiku-20240307",       # retired 2026-04-20
    "claude-3-5-haiku-20241022",     # retired 2026-02-19
    "claude-3-5-sonnet-20240620",    # retired 2025-10-28
    "claude-3-5-sonnet-20241022",    # retired 2025-10-28
    "claude-3-7-sonnet-20250219",    # retired 2026-02-19
    "claude-3-opus-20240229",        # retired 2026-01-05
    "claude-sonnet-4-20250514",      # retired 2026-06-15
    "claude-sonnet-4-0",             # alias -> retired
    "claude-opus-4-20250514",        # retired 2026-06-15
    "claude-opus-4-0",               # alias -> retired
}

# Bu modeller sampling parametrelerini (temperature/top_p/top_k) desteklemez;
# non-default bir deger gonderilirse 400 doner. Bu modellerde temperature
# parametresini hic gondermiyoruz. (Yeni modeller ciktikca listeyi guncelle.)
_NO_SAMPLING_PARAM_TAGS = ("opus-4-7", "opus-4-8", "sonnet-5", "fable-5", "mythos-5")


def _supports_temperature(model: str) -> bool:
    return not any(tag in model for tag in _NO_SAMPLING_PARAM_TAGS)


class ClaudeService(BaseLLMService):
    """Anthropic Claude RAG service."""

    def _init_client(self) -> None:
        loader = SecretsLoader()
        api_key = loader.get_secret("anthropic_api_key", "ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found")
            self.client = None
            return
        try:
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("Claude service initialized successfully")
        except Exception as e:
            logger.error(f"Claude initialization failed: {e}")
            self.client = None

    def _call_llm(
        self, system_instruction: str, user_prompt: str,
        model: str, max_tokens: int, temperature: float,
    ) -> LLMCallResult:
        # 1) Emekli model istendiyse calisan default'a cevir (404 onleme)
        if model in _RETIRED_CLAUDE_MODELS:
            logger.warning(
                f"Requested retired Claude model '{model}' — "
                f"falling back to default '{self.get_default_model()}'"
            )
            model = self.get_default_model()

        # 2) Parametreleri kur; temperature'i yalnizca destekleyen modellere gonder
        params = dict(
            model=model,
            max_tokens=max_tokens,
            system=system_instruction,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if _supports_temperature(model):
            params["temperature"] = temperature

        response = self.client.messages.create(**params)
        text = response.content[0].text if response.content else ""
        return LLMCallResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
        )

    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        return estimate_claude_cost(input_tokens, output_tokens, model)

    def get_available_models(self) -> List[str]:
        return list(CLAUDE_PRICING.keys())

    def get_default_model(self) -> str:
        return CLAUDE_DEFAULT_MODEL

    def test_connection(self) -> Dict:
        if not self.is_available():
            return {"success": False, "message": "Claude client not initialized (missing API key)"}
        try:
            self.client.messages.create(
                model=self.get_default_model(),
                max_tokens=5,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return {
                "success": True,
                "message": "Claude API connection successful",
                "model":   self.get_default_model(),
            }
        except Exception as e:
            return {"success": False, "message": f"Claude API error: {e}"}


# ═══ Singleton ═══
_claude_service_instance: ClaudeService = None


def get_claude_service() -> ClaudeService:
    """Get Claude service instance (singleton)."""
    global _claude_service_instance
    if _claude_service_instance is None:
        _claude_service_instance = ClaudeService()
    return _claude_service_instance