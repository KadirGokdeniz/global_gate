# services/elevenlabs_tts.py
import requests
import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Dict
from api.core.secrets_loader import SecretsLoader

loader = SecretsLoader()
logger = logging.getLogger(__name__)


class ElevenLabsService:
    """ElevenLabs Text-to-Speech Service

    Uses turbo_v2_5 model for fast, high-quality multilingual TTS.

    Includes a monthly character budget guard to prevent runaway cost
    on the ElevenLabs Starter plan (30,000 chars/month). The guard is
    in-memory only — it resets on container restart — which is fine
    because Railway restarts are rare and slowapi already enforces
    per-IP rate limits on the endpoint level.
    """

    # Voice IDs - multilingual, low latency
    VOICE_MAPPING = {
        "tr-TR": "EXAVITQu4vr4xnSDxMaL",  # Sarah - works well for Turkish
        "en-US": "EXAVITQu4vr4xnSDxMaL",  # Sarah - works well for English
        "en-GB": "EXAVITQu4vr4xnSDxMaL",
    }

    MODEL_ID = "eleven_turbo_v2_5"
    API_BASE = "https://api.elevenlabs.io/v1"

    # ════════════════════════════════════════════════════════════════
    # BUDGET GUARD
    # ElevenLabs Starter plan: 30,000 chars/month.
    # We cap at 25,000 to leave headroom — if this guard triggers, TTS
    # stops working for the rest of the month but the API keeps
    # returning the text answer normally.
    # ════════════════════════════════════════════════════════════════
    MONTHLY_CHAR_BUDGET = 25_000

    def __init__(self):
        """Initialize ElevenLabs client"""
        try:
            self.api_key = loader.get_secret(
                'elevenlabs_api_key',
                'ELEVENLABS_API_KEY'
            )

            if not self.api_key:
                raise ValueError("ELEVENLABS_API_KEY not configured")

            # Budget tracking state — thread-safe via lock because FastAPI
            # runs endpoints concurrently.
            self._budget_lock = Lock()
            self._chars_used_this_month = 0
            self._budget_month = self._current_month_key()

            self._test_connection()
            logger.info("ElevenLabs TTS service initialized successfully")

        except Exception as e:
            logger.error(f"ElevenLabs initialization failed: {e}")
            raise

    @staticmethod
    def _current_month_key() -> str:
        """Returns 'YYYY-MM' for the current UTC month — used as budget reset key."""
        return datetime.now(timezone.utc).strftime("%Y-%m")

    def _check_and_reserve_budget(self, char_count: int) -> bool:
        """
        Atomically check whether `char_count` fits in the remaining monthly
        budget, and if so, reserve those chars. Returns True on success.

        Resets the counter when a new month starts.
        """
        with self._budget_lock:
            current_month = self._current_month_key()
            if current_month != self._budget_month:
                # New month — reset the counter
                logger.info(
                    f"TTS budget: month rolled over "
                    f"{self._budget_month} -> {current_month}, resetting counter"
                )
                self._budget_month = current_month
                self._chars_used_this_month = 0

            if self._chars_used_this_month + char_count > self.MONTHLY_CHAR_BUDGET:
                remaining = self.MONTHLY_CHAR_BUDGET - self._chars_used_this_month
                logger.warning(
                    f"TTS budget exhausted: {self._chars_used_this_month:,} / "
                    f"{self.MONTHLY_CHAR_BUDGET:,} chars used, "
                    f"{remaining:,} remaining, request needs {char_count:,}"
                )
                return False

            self._chars_used_this_month += char_count
            return True

    def _test_connection(self):
        """Test API connection by checking models endpoint"""
        try:
            response = requests.get(
                f"{self.API_BASE}/models",
                headers={"xi-api-key": self.api_key},
                timeout=10
            )
            if response.status_code == 200:
                logger.info("ElevenLabs API connected successfully")
            elif response.status_code == 401:
                raise ValueError("ElevenLabs API key invalid")
            else:
                logger.warning(f"ElevenLabs returned {response.status_code}")
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"ElevenLabs connection test failed: {e}")
            raise

    def get_budget_status(self) -> dict:
        """Expose budget state for /health or observability endpoints."""
        with self._budget_lock:
            current_month = self._current_month_key()
            # If month changed but _check_and_reserve hasn't been called,
            # report as-if reset.
            if current_month != self._budget_month:
                used = 0
                month = current_month
            else:
                used = self._chars_used_this_month
                month = self._budget_month

        return {
            "month": month,
            "chars_used": used,
            "chars_budget": self.MONTHLY_CHAR_BUDGET,
            "chars_remaining": self.MONTHLY_CHAR_BUDGET - used,
            "percent_used": round(100 * used / self.MONTHLY_CHAR_BUDGET, 1),
        }

    def text_to_audio(self, text: str, language: str = "tr-TR") -> dict:
        """Convert text to speech using ElevenLabs

        Same interface as AWS Polly for drop-in replacement.
        """
        try:
            voice_id = self.VOICE_MAPPING.get(language, self.VOICE_MAPPING["tr-TR"])

            # Truncate if too long (ElevenLabs handles up to 5000 chars in v2)
            if len(text) > 2500:
                text = text[:2500]
                logger.warning("Text truncated to 2500 characters")

            # Budget guard — check BEFORE the paid API call
            if not self._check_and_reserve_budget(len(text)):
                status = self.get_budget_status()
                return {
                    "success": False,
                    "audio_data": b"",
                    "error": (
                        f"Monthly TTS budget exhausted "
                        f"({status['chars_used']:,}/{status['chars_budget']:,} chars). "
                        f"Try again next month."
                    ),
                    "budget_exhausted": True,
                }

            url = f"{self.API_BASE}/text-to-speech/{voice_id}"

            payload = {
                "text": text,
                "model_id": self.MODEL_ID,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }

            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"
            }

            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                audio_data = response.content
                status = self.get_budget_status()
                logger.info(
                    f"TTS successful: {len(text)} chars -> "
                    f"{len(audio_data)} bytes ({language}) | "
                    f"budget: {status['chars_used']:,}/{status['chars_budget']:,} "
                    f"({status['percent_used']}%)"
                )
                return {
                    "success": True,
                    "audio_data": audio_data,
                    "voice_used": voice_id,
                    "language": language,
                    "error": None
                }
            else:
                # API call failed AFTER we reserved budget — refund the chars.
                # This is a best-effort: if the failure is intermittent, we
                # don't punish the user's budget.
                with self._budget_lock:
                    self._chars_used_this_month = max(
                        0, self._chars_used_this_month - len(text)
                    )
                error_msg = f"ElevenLabs API error {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "audio_data": b"",
                    "error": error_msg
                }

        except requests.exceptions.Timeout:
            # Refund budget on timeout too
            with self._budget_lock:
                self._chars_used_this_month = max(
                    0, self._chars_used_this_month - len(text)
                )
            error_msg = "ElevenLabs API timeout"
            logger.error(error_msg)
            return {"success": False, "audio_data": b"", "error": error_msg}

        except Exception as e:
            error_msg = f"TTS error: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "audio_data": b"", "error": error_msg}


# Singleton pattern - same as aws_speech.py
_elevenlabs_service = None


def get_elevenlabs_service() -> ElevenLabsService:
    """Get ElevenLabs service instance (singleton)"""
    global _elevenlabs_service
    if _elevenlabs_service is None:
        _elevenlabs_service = ElevenLabsService()
    return _elevenlabs_service