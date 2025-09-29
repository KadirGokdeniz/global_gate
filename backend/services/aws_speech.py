# services/aws_speech.py
import boto3
import tempfile
import os
import logging
import json
import time
from typing import Dict, Optional
from botocore.exceptions import ClientError, NoCredentialsError
from secrets_loader import SecretsLoader

loader = SecretsLoader()

logger = logging.getLogger(__name__)

class AWSpeechService:
    """AWS Speech Services wrapper - Polly for Text-to-Speech"""
    
    def __init__(self):
        """Initialize AWS clients"""
        try:
            # AWS credentials check
            self.region = os.getenv("AWS_REGION", "eu-north-1")
            
            # Load AWS credentials from secrets
            aws_access_key_id = loader.get_secret('aws_access_key_id', 'AWS_ACCESS_KEY_ID')
            aws_secret_access_key = loader.get_secret('aws_secret_access_key', 'AWS_SECRET_ACCESS_KEY')
            
            # Polly client (TTS) - lowercase parameters!
            self.polly_client = boto3.client(
                'polly',
                region_name=self.region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
            
            # Test connections
            self._test_connections()
            logger.info("AWS Speech Services initialized successfully")
            
        except Exception as e:
            logger.error(f"AWS Speech Services initialization failed: {e}")
            raise

    def _test_connections(self):
        """Test AWS service connections"""
        try:
            voices = self.polly_client.describe_voices(LanguageCode='tr-TR')
            logger.info(f"Polly connection OK. Turkish voices available: {len(voices['Voices'])}")
        except ClientError as e:
            logger.error(f"Polly connection failed: {e}")
            raise e

    def text_to_audio(self, text: str, language: str = "tr-TR") -> dict:
        """Convert text to speech using AWS Polly"""
        try:
            voice_mapping = {
                "tr-TR": "Filiz",
                "en-US": "Joanna",
                "en-GB": "Emma"
            }
            
            voice_id = voice_mapping.get(language, "Filiz")
            
            if len(text) > 3000:
                text = text[:3000]
                logger.warning(f"Text truncated to 3000 characters for Polly")
            
            response = self.polly_client.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice_id,
                Engine='standard'
            )
            
            audio_data = response['AudioStream'].read()
            
            logger.info(f"TTS successful: {len(text)} chars -> {len(audio_data)} bytes")
            
            return {
                "success": True,
                "audio_data": audio_data,
                "voice_used": voice_id,
                "language": language,
                "error": None
            }
            
        except ClientError as e:
            error_msg = f"Polly TTS error: {e.response['Error']['Message']}"
            logger.error(error_msg)
            return {
                "success": False,
                "audio_data": b"",
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"TTS error: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "audio_data": b"",
                "error": error_msg
            }

_aws_speech_service = None

def get_aws_speech_service() -> AWSpeechService:
    """Get AWS Speech Service instance (singleton)"""
    global _aws_speech_service
    if _aws_speech_service is None:
        _aws_speech_service = AWSpeechService()
    return _aws_speech_service