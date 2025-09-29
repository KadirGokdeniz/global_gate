#!/usr/bin/env python3
"""
.gitignore Otomatik Güncelleyici
Eksik secret-related entries'leri ekler
"""

import os
from pathlib import Path

def update_gitignore():
    gitignore_path = Path('.gitignore')
    
    # Eklenecek satırlar
    required_entries = """
# ============================================
# SECRETS & ENVIRONMENT VARIABLES
# ============================================
.env
.env.local
.env.*.local
*.env
.env.backup
.env.production
.env.development
.env.staging

# Vault secrets
.vault-token
.vault-data/
secrets/
vault-config/

# AWS credentials
.aws/credentials
aws-credentials.json

# API Keys
*_key.txt
*_secret.txt
api_keys/

# ============================================
# GENERATED REPORTS
# ============================================
secrets_audit_report.json
security_scan_*.json

# ============================================
# DOCKER SECRETS (if using)
# ============================================
docker-secrets/
*.secret

# ============================================
# BACKUP FILES
# ============================================
*.bak
*.backup
*_backup
"""
    
    # Mevcut .gitignore'u oku
    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            current_content = f.read()
        
        print("✓ Mevcut .gitignore bulundu")
        
        # Yeni entries'leri ekle
        if '# SECRETS & ENVIRONMENT VARIABLES' not in current_content:
            with open(gitignore_path, 'a', encoding='utf-8') as f:
                f.write('\n' + required_entries)
            print("✓ Eksik entries eklendi")
        else:
            print("ℹ .gitignore zaten güncel görünüyor")
    else:
        # .gitignore yoksa oluştur
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(required_entries)
        print("✓ Yeni .gitignore oluşturuldu")
    
    # Doğrulama
    print("\n" + "="*60)
    print("DOĞRULAMA")
    print("="*60)
    
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    checks = {
        '.env': '.env' in content,
        '.env.local': '.env.local' in content,
        'secrets/': 'secrets/' in content,
        '.vault-token': '.vault-token' in content,
    }
    
    all_good = True
    for entry, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"{status} {entry}")
        if not exists:
            all_good = False
    
    if all_good:
        print("\n🎉 .gitignore başarıyla güncellendi!")
    else:
        print("\n⚠️ Bazı entries eksik kalabilir")
    
    return all_good

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════╗
║      .gitignore Otomatik Güncelleyici    ║
╚══════════════════════════════════════════╝
    """)
    update_gitignore()