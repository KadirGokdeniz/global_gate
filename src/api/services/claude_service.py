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
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_instruction,
            messages=[{"role": "user", "content": user_prompt}],
        )
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