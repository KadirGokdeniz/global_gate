# services/assemblyai_stt.py
import assemblyai as aai
import os
import logging
import tempfile
from typing import Dict, Optional
import asyncio
from api.core.secrets_loader import SecretsLoader

loader = SecretsLoader()

logger = logging.getLogger(__name__)

class AssemblyAIService:
    """
    AssemblyAI Speech-to-Text Service
    Real-time transcription with high accuracy
    """
    
    def __init__(self):
        """Initialize AssemblyAI client"""
        try:
            self.api_key = loader.get_secret('assemblyai_api_key', 'ASSEMBLYAI_API_KEY')
            if not self.api_key:
                raise ValueError("ASSEMBLYAI_API_KEY environment variable not set")
            
            # Set global API key
            aai.settings.api_key = self.api_key
            
            # Test connection
            self._test_connection()
            logger.info("AssemblyAI service initialized successfully")
            
        except Exception as e:
            logger.error(f"AssemblyAI service initialization failed: {e}")
            raise
    
    def _test_connection(self):
        """Test AssemblyAI connection"""
        try:
            # Simple connection test - try to access API
            transcriber = aai.Transcriber()
            logger.info("AssemblyAI connection test successful")
        except Exception as e:
            logger.error(f"AssemblyAI connection test failed: {e}")
            raise
    
    async def transcribe_audio_file(self, audio_bytes: bytes, 
                                   language: str = "en",
                                   filename: str = "audio.mp3") -> Dict:
        """
        Transcribe audio file using AssemblyAI
        
        Args:
            audio_bytes: Raw audio data
            language: Language code (tr, en, etc.)
            filename: Original filename for logging
            
        Returns:
            {"success": bool, "transcript": str, "error": str, "details": dict}
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(audio_bytes)
                temp_filepath = temp_file.name
            
            logger.info(f"Processing audio file: {filename} ({len(audio_bytes)} bytes)")
            
            # Configure transcription
            config = aai.TranscriptionConfig(
                language_code=language,
                punctuate=True,
                format_text=True
            )
            
            # Transcribe
            transcriber = aai.Transcriber(config=config)
            transcript = await asyncio.to_thread(transcriber.transcribe, temp_filepath)
            
            # Clean up temp file
            os.unlink(temp_filepath)
            
            if transcript.status == aai.TranscriptStatus.error:
                error_msg = f"Transcription failed: {transcript.error}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "transcript": "",
                    "error": error_msg,
                    "details": {
                        "status": str(transcript.status),
                        "language": language
                    }
                }
            
            transcript_text = transcript.text or ""
            
            logger.info(f"Transcription successful: {len(transcript_text)} characters")
            
            return {
                "success": True,
                "transcript": transcript_text,
                "error": None,
                "details": {
                    "status": str(transcript.status),
                    "language": language,
                    "confidence": getattr(transcript, 'confidence', 0.0),
                    "audio_duration": getattr(transcript, 'audio_duration', 0),
                    "processing_time": getattr(transcript, 'processing_time', 0),
                    "words_count": len(transcript_text.split()) if transcript_text else 0
                }
            }
            
        except Exception as e:
            error_msg = f"AssemblyAI transcription error: {str(e)}"
            logger.error(error_msg)
            
            # Clean up temp file if it exists
            try:
                if 'temp_filepath' in locals():
                    os.unlink(temp_filepath)
            except:
                pass
            
            return {
                "success": False,
                "transcript": "",
                "error": error_msg,
                "details": {
                    "language": language,
                    "filename": filename
                }
            }
    
    async def transcribe_realtime_stream(self, audio_stream) -> Dict:
        """
        Real-time streaming transcription (placeholder for future implementation)
        
        Args:
            audio_stream: Audio stream data
            
        Returns:
            {"success": bool, "transcript": str, "error": str}
        """
        # Note: AssemblyAI real-time streaming requires WebSocket implementation
        # This is a placeholder for future enhancement
        return {
            "success": False,
            "transcript": "",
            "error": "Real-time streaming not implemented yet. Use file transcription.",
            "details": {
                "feature": "realtime_streaming",
                "status": "not_implemented"
            }
        }
    
    def get_service_info(self) -> Dict:
        """Get service information"""
        return {
            "service": "AssemblyAI Speech-to-Text",
            "features": {
                "file_transcription": True,
                "realtime_streaming": False,  # Future feature
                "multilingual": True,
                "punctuation": True,
                "formatting": True
            },
            "supported_languages": [
                "tr",  # Turkish
                "en"  # English  
            ],
            "api_key_configured": bool(self.api_key),
            "status": "ready"
        }

# Service instance getter (singleton pattern)
_assemblyai_service = None

def get_assemblyai_service() -> AssemblyAIService:
    """Get AssemblyAI service instance (singleton)"""
    global _assemblyai_service
    if _assemblyai_service is None:
        _assemblyai_service = AssemblyAIService()
    return _assemblyai_service