"""
Docker Secrets Loader Utility
Container icinde /run/secrets/ dizininden secrets'lari okur
Fallback olarak environment variables kullanir
"""

import os
from pathlib import Path
from typing import Optional


class SecretsLoader:
    """Docker Secrets veya .env dosyasindan secrets yukler"""
    
    SECRETS_DIR = Path("/run/secrets")
    
    @staticmethod
    def get_secret(name: str, fallback_env: Optional[str] = None) -> Optional[str]:
        """
        Docker secret'i veya environment variable'i oku
        
        Args:
            name: Secret dosya adi (orn: 'openai_api_key')
            fallback_env: Eger secret yoksa bu env variable'i dene
        
        Returns:
            Secret degeri veya None
        """
        # 1. Docker secret dosyasini kontrol et
        secret_file = SecretsLoader.SECRETS_DIR / name
        if secret_file.exists():
            try:
                with open(secret_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read secret {name}: {e}")
        
        # 2. _FILE suffix'li environment variable'i kontrol et
        file_env = os.getenv(f"{fallback_env}_FILE") if fallback_env else None
        if file_env and Path(file_env).exists():
            try:
                with open(file_env, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read secret file from env {fallback_env}_FILE: {e}")
        
        # 3. Direkt environment variable'i kontrol et (fallback)
        if fallback_env:
            return os.getenv(fallback_env)
        
        return None


# Kullanim ornekleri
if __name__ == "__main__":
    loader = SecretsLoader()
    
    # OpenAI key'i al (once Docker secret, sonra env variable)
    openai_key = loader.get_secret('openai_api_key', 'OPENAI_API_KEY')
    print(f"OpenAI Key: {openai_key[:10]}..." if openai_key else "Not found")
    
    # Claude key'i al
    claude_key = loader.get_secret('anthropic_api_key', 'ANTHROPIC_API_KEY')
    print(f"Claude Key: {claude_key[:10]}..." if claude_key else "Not found")
