#!/usr/bin/env python3
"""
Secrets Security Audit Script
Projedeki tüm secrets'ları tespit eder ve risk analizi yapar
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

class SecretsAuditor:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.findings = defaultdict(list)
        self.secrets_patterns = {
            'openai_key': r'(OPENAI_API_KEY|openai[_-]?key)',
            'claude_key': r'(ANTHROPIC_API_KEY|claude[_-]?key)',
            'aws_key': r'(AWS_ACCESS_KEY_ID|AWS_SECRET_ACCESS_KEY|aws[_-]?key)',
            'assemblyai_key': r'(ASSEMBLYAI_API_KEY|assembly[_-]?key)',
            'postgres_pass': r'(POSTGRES_PASSWORD|DB_PASSWORD|database[_-]?pass)',
            'jwt_secret': r'(JWT_SECRET|SECRET_KEY|secret[_-]?key)',
        }
        
    def scan_env_files(self):
        """Tüm .env dosyalarını tara"""
        print("\n[1] .env Dosyaları Taranıyor...")
        env_files = list(self.project_root.rglob('.env*'))
        
        for env_file in env_files:
            if env_file.name == '.env.example':
                continue
                
            print(f"  ✓ Bulundu: {env_file}")
            try:
                with open(env_file, 'r') as f:
                    content = f.read()
                    
                # Secrets bul
                for secret_type, pattern in self.secrets_patterns.items():
                    matches = re.finditer(f'{pattern}\\s*=\\s*(.+)', content, re.IGNORECASE)
                    for match in matches:
                        value = match.group(2).strip().strip('"\'')
                        if value and value != 'your_key_here' and len(value) > 10:
                            self.findings['env_files'].append({
                                'file': str(env_file),
                                'type': secret_type,
                                'value_preview': value[:10] + '...',
                                'risk': 'YÜKSEK'
                            })
            except Exception as e:
                print(f"  ✗ Hata: {env_file} - {e}")
    
    def scan_docker_compose(self):
        """Docker Compose dosyalarını tara"""
        print("\n[2] Docker Compose Dosyaları Taranıyor...")
        compose_files = list(self.project_root.glob('*docker-compose*.yml')) + \
                       list(self.project_root.glob('*docker-compose*.yaml'))
        
        for compose_file in compose_files:
            print(f"  ✓ Bulundu: {compose_file}")
            try:
                with open(compose_file, 'r') as f:
                    content = f.read()
                    
                # Environment variables kontrol et
                if 'environment:' in content:
                    for secret_type, pattern in self.secrets_patterns.items():
                        if re.search(pattern, content, re.IGNORECASE):
                            self.findings['docker_compose'].append({
                                'file': str(compose_file),
                                'type': secret_type,
                                'risk': 'ORTA' if '${' in content else 'YÜKSEK'
                            })
            except Exception as e:
                print(f"  ✗ Hata: {compose_file} - {e}")
    
    def scan_source_code(self):
        """Kaynak kodda hardcoded secrets ara"""
        print("\n[3] Kaynak Kod Taranıyor...")
        code_files = list(self.project_root.rglob('*.py')) + \
                    list(self.project_root.rglob('*.js')) + \
                    list(self.project_root.rglob('*.ts'))
        
        dangerous_patterns = [
            (r'["\']sk-[a-zA-Z0-9]{48}["\']', 'OpenAI Key'),
            (r'["\']sk-ant-[a-zA-Z0-9-]{95}["\']', 'Anthropic Key'),
            (r'AKIA[0-9A-Z]{16}', 'AWS Access Key'),
            (r'["\'][a-f0-9]{40}["\']', 'Possible Secret'),
        ]
        
        for code_file in code_files:
            if 'node_modules' in str(code_file) or 'venv' in str(code_file):
                continue
                
            try:
                with open(code_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, secret_type in dangerous_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        self.findings['hardcoded'].append({
                            'file': str(code_file),
                            'type': secret_type,
                            'line': content[:match.start()].count('\n') + 1,
                            'risk': 'KRİTİK'
                        })
            except:
                continue
    
    def check_gitignore(self):
        """gitignore kontrolü"""
        print("\n[4] .gitignore Kontrolü...")
        gitignore_path = self.project_root / '.gitignore'
        
        required_entries = ['.env', '.env.local', '*.env', 'secrets/', '.vault-token']
        
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                content = f.read()
            
            missing = [entry for entry in required_entries if entry not in content]
            
            if missing:
                self.findings['gitignore'] = {
                    'status': 'EKSİK',
                    'missing_entries': missing,
                    'risk': 'YÜKSEK'
                }
            else:
                self.findings['gitignore'] = {
                    'status': 'TAMAM',
                    'risk': 'DÜŞÜK'
                }
        else:
            self.findings['gitignore'] = {
                'status': 'BULUNAMADI',
                'risk': 'KRİTİK'
            }
    
    def check_git_history(self):
        """Git history'de sızma kontrolü (basit)"""
        print("\n[5] Git History Kontrolü...")
        
        if (self.project_root / '.git').exists():
            print("  ℹ Git repository bulundu")
            print("  ⚠ Manuel kontrol gerekli:")
            print("     git log --all --full-history --source -- .env")
            print("     git log -S 'OPENAI_API_KEY' --all")
            
            self.findings['git_history'] = {
                'status': 'MANUEL KONTROL GEREKLİ',
                'risk': 'BİLİNMİYOR'
            }
        else:
            print("  ℹ Git repository bulunamadı")
    
    def generate_report(self):
        """Rapor oluştur"""
        print("\n" + "="*70)
        print("SECRETS GÜVENLİK RAPORU")
        print("="*70)
        
        total_issues = sum(len(v) if isinstance(v, list) else 1 
                          for v in self.findings.values())
        
        print(f"\n📊 Toplam Tespit: {total_issues} sorun bulundu\n")
        
        # Risk seviyelerine göre sırala
        risk_colors = {
            'KRİTİK': '🔴',
            'YÜKSEK': '🟠',
            'ORTA': '🟡',
            'DÜŞÜK': '🟢',
            'BİLİNMİYOR': '⚪'
        }
        
        for category, items in self.findings.items():
            print(f"\n{'─'*70}")
            print(f"📁 {category.upper().replace('_', ' ')}")
            print(f"{'─'*70}")
            
            if isinstance(items, list):
                for item in items:
                    risk = item.get('risk', 'BİLİNMİYOR')
                    print(f"\n{risk_colors[risk]} {risk} Risk:")
                    for key, value in item.items():
                        if key != 'risk':
                            print(f"   • {key}: {value}")
            else:
                risk = items.get('risk', 'BİLİNMİYOR')
                print(f"\n{risk_colors[risk]} {risk} Risk:")
                for key, value in items.items():
                    if key != 'risk':
                        print(f"   • {key}: {value}")
        
        # Öneriler
        print(f"\n{'='*70}")
        print("🔧 ÖNCELİKLİ AKSIYONLAR")
        print(f"{'='*70}\n")
        
        action_plan = []
        
        if self.findings.get('hardcoded'):
            action_plan.append("1. [KRİTİK] Hardcoded secrets'ları HEMEN kaldırın ve API anahtarlarını yenileyin!")
        
        if self.findings.get('env_files'):
            action_plan.append("2. [YÜKSEK] .env dosyalarını .gitignore'a ekleyin ve git history'den temizleyin")
        
        if self.findings.get('gitignore', {}).get('status') != 'TAMAM':
            action_plan.append("3. [YÜKSEK] .gitignore dosyasını düzenleyin")
        
        action_plan.append("4. [ORTA] HashiCorp Vault veya AWS Secrets Manager kurulumuna başlayın")
        action_plan.append("5. [ORTA] Tüm API anahtarlarını rotate edin (yenilenip yenilenmediğinden emin olun)")
        
        for i, action in enumerate(action_plan, 1):
            print(f"{action}")
        
        # JSON export
        report_file = self.project_root / 'secrets_audit_report.json'
        with open(report_file, 'w') as f:
            json.dump(dict(self.findings), f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Detaylı rapor kaydedildi: {report_file}")
        print("\n" + "="*70 + "\n")
        
        return self.findings

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           SECRETS SECURITY AUDIT TOOL                            ║
║        Multi-Airline RAG System - Güvenlik Analizi               ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    project_path = input("Proje dizinini girin (varsayılan: mevcut dizin): ").strip() or "."
    
    auditor = SecretsAuditor(project_path)
    
    auditor.scan_env_files()
    auditor.scan_docker_compose()
    auditor.scan_source_code()
    auditor.check_gitignore()
    auditor.check_git_history()
    
    auditor.generate_report()

if __name__ == "__main__":
    main()