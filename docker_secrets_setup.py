#!/usr/bin/env python3
"""
Docker Secrets Otomatik Kurulum Script'i
.env dosyasından Docker Secrets'a geçiş yapar
"""

import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

class DockerSecretsSetup:
    def __init__(self):
        self.secrets_dir = Path('secrets')
        self.env_file = Path('.env')
        
    def create_secrets_directory(self):
        """Secrets dizini oluştur"""
        self.secrets_dir.mkdir(exist_ok=True)
        print(f"✓ Secrets dizini oluşturuldu: {self.secrets_dir}")
        
        # .gitignore'a ekle
        gitignore = Path('.gitignore')
        with open(gitignore, 'r+') as f:
            content = f.read()
            if 'secrets/' not in content:
                f.write('\nsecrets/\n')
                print("✓ secrets/ dizini .gitignore'a eklendi")
    
    def load_env_variables(self):
        """env dosyasından değişkenleri yükle"""
        if not self.env_file.exists():
            print(f"✗ {self.env_file} bulunamadı!")
            return {}
        
        load_dotenv(self.env_file)
        
        secrets = {
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'anthropic_api_key': os.getenv('ANTHROPIC_API_KEY'),
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'assemblyai_api_key': os.getenv('ASSEMBLYAI_API_KEY'),
            'postgres_password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        }
        
        # None olmayan secrets'ları filtrele
        secrets = {k: v for k, v in secrets.items() if v}
        
        print(f"\n✓ {len(secrets)} adet secret bulundu:")
        for key in secrets.keys():
            print(f"  • {key}")
        
        return secrets
    
    def create_secret_files(self, secrets):
        """Her secret için dosya oluştur"""
        print("\n[1] Secret dosyaları oluşturuluyor...")
        
        for name, value in secrets.items():
            secret_file = self.secrets_dir / name
            with open(secret_file, 'w') as f:
                f.write(value)
            
            # Sadece owner okuyabilsin (güvenlik)
            os.chmod(secret_file, 0o600)
            print(f"  ✓ {name}")
        
        print(f"\n✓ Tüm secret dosyaları oluşturuldu: {self.secrets_dir}/")
    
    def generate_docker_compose_secrets(self, secrets):
        """Docker Compose için secrets konfigürasyonu oluştur"""
        print("\n[2] Docker Compose secrets konfigürasyonu oluşturuluyor...")
        
        # Secrets tanımları
        secrets_config = "\nsecrets:\n"
        for name in secrets.keys():
            secrets_config += f"  {name}:\n"
            secrets_config += f"    file: ./secrets/{name}\n"
        
        # Service secrets kullanımı
        service_secrets = "\n    secrets:\n"
        for name in secrets.keys():
            service_secrets += f"      - {name}\n"
        
        output_file = Path('docker-compose.secrets.yml')
        
        compose_content = f"""# Docker Compose Secrets Configuration
# Bu dosyayı docker-compose.yml ile birleştirin

version: '3.8'

{secrets_config}

services:
  backend:
{service_secrets}
    environment:
      # Secrets dosya yollarını environment variable olarak ekle
      OPENAI_API_KEY_FILE: /run/secrets/openai_api_key
      ANTHROPIC_API_KEY_FILE: /run/secrets/anthropic_api_key
      AWS_ACCESS_KEY_ID_FILE: /run/secrets/aws_access_key_id
      AWS_SECRET_ACCESS_KEY_FILE: /run/secrets/aws_secret_access_key
      ASSEMBLYAI_API_KEY_FILE: /run/secrets/assemblyai_api_key
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password

  scraper:
{service_secrets}
    environment:
      OPENAI_API_KEY_FILE: /run/secrets/openai_api_key
      ANTHROPIC_API_KEY_FILE: /run/secrets/anthropic_api_key
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password

  postgres:
    secrets:
      - postgres_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
"""
        
        with open(output_file, 'w') as f:
            f.write(compose_content)
        
        print(f"✓ {output_file} oluşturuldu")
        return output_file
    
    def generate_secret_loader(self):
        """Python kod için secret loader oluştur"""
        print("\n[3] Python secret loader oluşturuluyor...")
        
        loader_code = '''"""
Docker Secrets Loader Utility
Container içinde /run/secrets/ dizininden secrets'ları okur
"""

import os
from pathlib import Path
from typing import Optional

class SecretsLoader:
    """Docker Secrets veya .env dosyasından secrets yükler"""
    
    SECRETS_DIR = Path("/run/secrets")
    
    @staticmethod
    def get_secret(name: str, fallback_env: Optional[str] = None) -> Optional[str]:
        """
        Docker secret'ı veya environment variable'ı oku
        
        Args:
            name: Secret dosya adı (örn: 'openai_api_key')
            fallback_env: Eğer secret yoksa bu env variable'ı dene
        
        Returns:
            Secret değeri veya None
        """
        # 1. Docker secret dosyasını kontrol et
        secret_file = SecretsLoader.SECRETS_DIR / name
        if secret_file.exists():
            try:
                with open(secret_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read secret {name}: {e}")
        
        # 2. _FILE suffix'li environment variable'ı kontrol et
        file_env = os.getenv(f"{fallback_env}_FILE") if fallback_env else None
        if file_env and Path(file_env).exists():
            try:
                with open(file_env, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Warning: Could not read secret file from env {fallback_env}_FILE: {e}")
        
        # 3. Direkt environment variable'ı kontrol et (fallback)
        if fallback_env:
            return os.getenv(fallback_env)
        
        return None

# Kullanım örnekleri
if __name__ == "__main__":
    loader = SecretsLoader()
    
    # OpenAI key'i al (önce Docker secret, sonra env variable)
    openai_key = loader.get_secret('openai_api_key', 'OPENAI_API_KEY')
    print(f"OpenAI Key: {openai_key[:10]}..." if openai_key else "Not found")
    
    # Claude key'i al
    claude_key = loader.get_secret('anthropic_api_key', 'ANTHROPIC_API_KEY')
    print(f"Claude Key: {claude_key[:10]}..." if claude_key else "Not found")
'''
        
        loader_file = Path('secrets_loader.py')
        with open(loader_file, 'w') as f:
            f.write(loader_code)
        
        print(f"✓ {loader_file} oluşturuldu")
        print("\n  Kodunuzda şöyle kullanın:")
        print("  from secrets_loader import SecretsLoader")
        print("  openai_key = SecretsLoader.get_secret('openai_api_key', 'OPENAI_API_KEY')")
        
        return loader_file
    
    def create_backup(self):
        """env dosyasını yedekle"""
        if self.env_file.exists():
            backup_file = Path('.env.backup')
            import shutil
            shutil.copy(self.env_file, backup_file)
            print(f"\n✓ {self.env_file} yedeklendi → {backup_file}")
    
    def generate_instructions(self):
        """Kullanım talimatları oluştur"""
        instructions = """
╔══════════════════════════════════════════════════════════════════╗
║                   KURULUM TAMAMLANDI!                            ║
╚══════════════════════════════════════════════════════════════════╝

📁 Oluşturulan Dosyalar:
  ✓ secrets/               - Secret dosyaları (GİT'E EKLEMEYİN!)
  ✓ docker-compose.secrets.yml - Docker secrets konfigürasyonu
  ✓ secrets_loader.py      - Python secret loader utility

🔧 Sonraki Adımlar:

1. Docker Compose dosyanızı güncelleyin:
   
   Mevcut docker-compose.yml'nizde:
   
   A) environment: bölümünde ${OPENAI_API_KEY} gibi kullanımları kaldırın
   
   B) secrets: bölümünü ekleyin (docker-compose.secrets.yml'den kopyalayın)

2. Backend kodunuzu güncelleyin:
   
   # ESKİ YÖNTEMLERİN YERİNE:
   from secrets_loader import SecretsLoader
   
   loader = SecretsLoader()
   openai_key = loader.get_secret('openai_api_key', 'OPENAI_API_KEY')
   claude_key = loader.get_secret('anthropic_api_key', 'ANTHROPIC_API_KEY')

3. Test edin:
   
   docker-compose down
   docker-compose up -d
   docker-compose logs backend

4. .env dosyasını silin veya taşıyın:
   
   mv .env .env.old  # Yedek olarak saklayın
   
   Artık .env dosyasına ihtiyacınız yok!

⚠️  GÜVENLİK UYARILARI:

• secrets/ dizinini asla git'e eklemeyin (zaten .gitignore'da)
• Production'da secrets_loader.py'yi kullanmaya devam edin
• .env.backup dosyasını güvenli bir yerde saklayın

📖 Daha fazla bilgi:
   https://docs.docker.com/engine/swarm/secrets/

═══════════════════════════════════════════════════════════════════
"""
        
        instructions_file = Path('SECRETS_SETUP_INSTRUCTIONS.md')
        with open(instructions_file, 'w') as f:
            f.write(instructions)
        
        print(instructions)
        print(f"\n📄 Talimatlar kaydedildi: {instructions_file}")

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║             Docker Secrets Otomatik Kurulum                      ║
║           .env → Docker Secrets Geçiş Asistanı                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    setup = DockerSecretsSetup()
    
    # 1. Secrets dizini oluştur
    setup.create_secrets_directory()
    
    # 2. .env'yi yedekle
    setup.create_backup()
    
    # 3. .env'den secrets'ları yükle
    secrets = setup.load_env_variables()
    
    if not secrets:
        print("\n✗ Hiçbir secret bulunamadı! .env dosyasını kontrol edin.")
        return
    
    # 4. Secret dosyalarını oluştur
    setup.create_secret_files(secrets)
    
    # 5. Docker Compose konfigürasyonu oluştur
    setup.generate_docker_compose_secrets(secrets)
    
    # 6. Python secret loader oluştur
    setup.generate_secret_loader()
    
    # 7. Talimatları göster
    setup.generate_instructions()
    
    print("\n🎉 Kurulum başarıyla tamamlandı!")
    print("\n💡 Şimdi SECRETS_SETUP_INSTRUCTIONS.md dosyasını okuyun.\n")

if __name__ == "__main__":
    main()