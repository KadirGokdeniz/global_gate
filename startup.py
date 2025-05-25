#!/usr/bin/env python3
import subprocess
import time
import sys
import os

print("🚀 Turkish Airlines API Container başlatılıyor...")

# Database bağlantısını bekle
print("📊 Database bağlantısı kontrol ediliyor...")
max_retries = 30
for i in range(max_retries):
    try:
        result = subprocess.run([
            'pg_isready', 
            '-h', os.getenv('DB_HOST', 'db'),
            '-p', '5432',
            '-U', os.getenv('DB_USER', 'postgres'),
            '-d', os.getenv('DB_DATABASE', 'global_gate')
        ], capture_output=True)
        
        if result.returncode == 0:
            print("✅ Database hazır!")
            break
    except:
        pass
    
    print(f"⏳ Database bekleniyor... ({i+1}/{max_retries})")
    time.sleep(2)

# Web scraping çalıştır
print("🌐 Turkish Airlines web scraping başlatılıyor...")
try:
    result = subprocess.run(['python', 'web_scrapper.py'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Web scraping tamamlandı!")
    else:
        print("⚠️ Web scraping hatası - API yine de başlatılıyor...")
        print(f"Hata: {result.stderr}")
except Exception as e:
    print(f"⚠️ Web scraping exception: {e}")

# API'yi başlat
print("🚀 FastAPI server başlatılıyor...")
try:
    subprocess.run(['uvicorn', 'myapp:app', '--host', '0.0.0.0', '--port', '8000'])
except KeyboardInterrupt:
    print("🛑 API server kapatılıyor...")
    sys.exit(0)