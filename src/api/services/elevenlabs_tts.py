# services/elevenlabs_tts.py
import requests
import logging
from typing import Dict
from api.core.secrets_loader import SecretsLoader

loader = SecretsLoader()
logger = logging.getLogger(__name__)


class ElevenLabsService:
    """ElevenLabs Text-to-Speech Service

    Uses turbo_v2_5 model for fast, high-quality multilingual TTS.
    """
    
    # Voice IDs - multilingual, low latency
    VOICE_MAPPING = {
        "tr-TR": "EXAVITQu4vr4xnSDxMaL",  # Sarah - works well for Turkish
        "en-US": "EXAVITQu4vr4xnSDxMaL",  # Sarah - works well for English
        "en-GB": "EXAVITQu4vr4xnSDxMaL",
    }
    
    MODEL_ID = "eleven_turbo_v2_5"
    API_BASE = "https://api.elevenlabs.io/v1"
    
    def __init__(self):
        """Initialize ElevenLabs client"""
        try:
            self.api_key = loader.get_secret(
                'elevenlabs_api_key',
                'ELEVENLABS_API_KEY'
            )
            
            if not self.api_key:
                raise ValueError("ELEVENLABS_API_KEY not configured")
            
            self._test_connection()
            logger.info("ElevenLabs TTS service initialized successfully")
            
        except Exception as e:
            logger.error(f"ElevenLabs initialization failed: {e}")
            raise
    
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
                logger.info(
                    f"TTS successful: {len(text)} chars -> "
                    f"{len(audio_data)} bytes ({language})"
                )
                return {
                    "success": True,
                    "audio_data": audio_data,
                    "voice_used": voice_id,
                    "language": language,
                    "error": None
                }
            else:
                error_msg = f"ElevenLabs API error {response.status_code}: {response.text[:200]}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "audio_data": b"",
                    "error": error_msg
                }
                
        except requests.exceptions.Timeout:
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