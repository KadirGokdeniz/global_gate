"""
OpenAI LLM provider.

Tum ortak mantik (prompt'lar, CoT parsing, context, error shaping)
llm_base.py'de. Bu dosya sadece OpenAI SDK'sini cagiran ince bir
wrapper — ~80 satir.
"""
import logging
from typing import Dict, List

import openai

from api.core.secrets_loader import SecretsLoader
from api.services.llm_base import (
    BaseLLMService,
    LLMCallResult,
    OPENAI_DEFAULT_MODEL,
    OPENAI_PRICING,
    estimate_openai_cost,
)

logger = logging.getLogger(__name__)


class OpenAIService(BaseLLMService):
    """OpenAI RAG service."""

    def _init_client(self) -> None:
        loader = SecretsLoader()
        api_key = loader.get_secret("openai_api_key", "OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found")
            self.client = None
            return
        try:
            self.client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI service initialized successfully")
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            self.client = None

    def _call_llm(
        self, system_instruction: str, user_prompt: str,
        model: str, max_tokens: int, temperature: float,
    ) -> LLMCallResult:
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = ""
        if response.choices and response.choices[0].message.content:
            text = response.choices[0].message.content
        return LLMCallResult(
            text=text,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=model,
        )

    def _estimate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        return estimate_openai_cost(input_tokens, output_tokens, model)

    def get_available_models(self) -> List[str]:
        return list(OPENAI_PRICING.keys())

    def get_default_model(self) -> str:
        return OPENAI_DEFAULT_MODEL

    def test_connection(self) -> Dict:
        if not self.is_available():
            return {"success": False, "message": "OpenAI client not initialized (missing API key)"}
        try:
            self.client.chat.completions.create(
                model=self.get_default_model(),
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return {
                "success": True,
                "message": "OpenAI API connection successful",
                "model":   self.get_default_model(),
            }
        except Exception as e:
            return {"success": False, "message": f"OpenAI API error: {e}"}


# ═══ Singleton ═══
_openai_service_instance: OpenAIService = None


def get_openai_service() -> OpenAIService:
    """Get OpenAI service instance (singleton)."""
    global _openai_service_instance
    if _openai_service_instance is None:
        _openai_service_instance = OpenAIService()
    return _openai_service_instance