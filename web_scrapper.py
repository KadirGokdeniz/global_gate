from bs4 import BeautifulSoup
import requests
import time
import random
import hashlib
import re
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import asyncio
from vector_operations import VectorOperations
import asyncpg

# PostgreSQL Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'db'),  # 'localhost' yerine 'db'
    'database': os.getenv('DB_DATABASE', 'global_gate'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'qeqe')
}

def get_db_connection():
    """PostgreSQL bağlantısı kur"""
    try:
        print(f"🔗 Database'e bağlanıyor: {DB_CONFIG['host']}:{DB_CONFIG['database']}")
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ PostgreSQL bağlantısı başarılı")
        return conn
    except Exception as e:
        print(f"❌ PostgreSQL bağlantı hatası: {e}")
        return None

def setup_database():
    """Database ve tabloları oluştur"""
    
    conn = get_db_connection()
    if not conn:
        print("❌ Database bağlantısı kurulamadı")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Ana policies tablosu
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS baggage_policies (
                id SERIAL PRIMARY KEY,
                source VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                content_hash VARCHAR(32) UNIQUE NOT NULL,
                quality_score REAL DEFAULT 0.0,
                extraction_type VARCHAR(50) DEFAULT 'unknown',
                url TEXT,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Performance için indexler
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON baggage_policies(source);
            CREATE INDEX IF NOT EXISTS idx_content_hash ON baggage_policies(content_hash);
            CREATE INDEX IF NOT EXISTS idx_quality_score ON baggage_policies(quality_score);
            CREATE INDEX IF NOT EXISTS idx_created_at ON baggage_policies(created_at);
        """)
        
        # Full-text search için
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_fts 
            ON baggage_policies 
            USING GIN (to_tsvector('english', content));
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ PostgreSQL database ve tablolar hazırlandı")
        return True
        
    except Exception as e:
        print(f"❌ Database setup hatası: {e}")
        return False

def clear_old_data():
    """Eski verileri temizle (fresh start için)"""
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM baggage_policies")
        deleted_count = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"🗑️ Eski {deleted_count} policy temizlendi (fresh start)")
        return True
        
    except Exception as e:
        print(f"❌ Veri temizleme hatası: {e}")
        return False

def save_policies_to_postgresql(policies):
    """Temizlenmiş policies'leri PostgreSQL'e kaydet"""
    
    conn = get_db_connection()
    if not conn:
        return 0
    
    try:
        cursor = conn.cursor()
        saved_count = 0
        
        insert_sql = """
            INSERT INTO baggage_policies 
            (source, content, content_hash, quality_score, extraction_type, url, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) 
            DO UPDATE SET 
                updated_at = CURRENT_TIMESTAMP,
                quality_score = EXCLUDED.quality_score
            RETURNING id;
        """
        
        for policy in policies:
            try:
                cursor.execute(insert_sql, (
                    policy['source'],
                    policy['content'],
                    policy['content_hash'],
                    policy.get('quality_score', 0.0),
                    policy.get('type', 'unknown'),
                    policy.get('url', ''),
                    psycopg2.extras.Json(policy.get('metadata', {}))
                ))
                
                result = cursor.fetchone()
                if result:
                    saved_count += 1
                    
            except Exception as e:
                print(f"  ⚠️ Policy kaydetme hatası: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"✅ {saved_count} policy PostgreSQL'e kaydedildi")
        return saved_count
        
    except Exception as e:
        print(f"❌ PostgreSQL kaydetme hatası: {e}")
        return 0

# Scraping fonksiyonları (önceki kodla aynı)
def scrape_turkish_airlines_page(url, page_name):
    """Turkish Airlines belirli sayfasından baggage bilgilerini çek"""
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    try:
        time.sleep(random.uniform(1, 2))
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        baggage_data = []
        
        # Strategy 1: List-based content
        li_elements = soup.select('#page_wrapper .container .row li')
        if li_elements:
            print(f"  └── Liste yapısı: {len(li_elements)} li element bulundu")
            for li in li_elements:
                text = li.get_text(strip=True)
                if text and len(text) > 10 and not is_navigation_item(text):
                    baggage_data.append({
                        'source': page_name,
                        'content': text,
                        'url': url,
                        'type': 'list_item'
                    })
        
        # Strategy 2: Header-paragraph content
        if not baggage_data:
            content_elements = soup.select('#page_wrapper .container h2, #page_wrapper .container h3, #page_wrapper .container p')
            if content_elements:
                print(f"  └── Header-paragraph yapısı: {len(content_elements)} element bulundu")
                for element in content_elements:
                    text = element.get_text(strip=True)
                    if text and len(text) > 15 and not is_navigation_item(text):
                        baggage_data.append({
                            'source': page_name,
                            'content': text,
                            'url': url,
                            'type': f'header_{element.name}'
                        })
        
        # Strategy 3: Table-based content
        tables = soup.select('#page_wrapper table')
        if tables:
            print(f"  └── Tablo yapısı: {len(tables)} table bulundu")
            
            for table_idx, table in enumerate(tables):
                table_context = extract_table_with_context(table, page_name, url, table_idx)
                baggage_data.extend(table_context)
        
        return baggage_data
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Request error: {e}")
        return []
    except Exception as e:
        print(f"  ❌ Parsing error: {e}")
        return []

def extract_table_with_context(table, page_name, url, table_idx):
    """Tablo verilerini bağlamını koruyarak çıkar"""
    
    table_data = []
    rows = table.find_all('tr')
    headers = []
    
    if rows:
        first_row = rows[0]
        header_cells = first_row.find_all(['th', 'td'])
        headers = [cell.get_text(strip=True) for cell in header_cells if cell.get_text(strip=True)]
    
    for row_idx, row in enumerate(rows[1:], 1):
        cells = row.find_all(['td', 'th'])
        cell_texts = [cell.get_text(strip=True) for cell in cells if cell.get_text(strip=True)]
        
        if cell_texts:
            if headers and len(cell_texts) >= len(headers):
                row_content = []
                for i, cell_text in enumerate(cell_texts[:len(headers)]):
                    if i < len(headers) and headers[i]:
                        row_content.append(f"{headers[i]}: {cell_text}")
                    else:
                        row_content.append(cell_text)
                
                formatted_content = " | ".join(row_content)
            else:
                formatted_content = " | ".join(cell_texts)
            
            table_data.append({
                'source': page_name,
                'content': formatted_content,
                'url': url,
                'type': 'table_row',
                'metadata': {
                    'table_index': table_idx,
                    'row_index': row_idx,
                    'headers': headers,
                    'cell_count': len(cell_texts)
                }
            })
    
    return table_data

def is_navigation_item(text):
    """Navigation/menu itemlerini filtrele"""
    nav_keywords = [
        'home', 'menu', 'search', 'login', 'contact',
        'about', 'services', 'help', 'support', 'more information'
    ]
    return any(keyword in text.lower() for keyword in nav_keywords)

def calculate_quality_score(item):
    """Content quality score hesapla"""
    content = item['content']
    score = len(content) * 0.01  # Length bonus
    
    # Structured content bonuses
    if '|' in content: score += 0.5      # Structured content
    if ':' in content: score += 0.3      # Key-value pairs
    if re.search(r'\d+\s*(kg|cm|ml|liter|%)', content.lower()): score += 0.4  # Numeric info
    if '✓' in content or 'X' in content: score += 0.25  # Table symbols
    if item.get('type') == 'table_row': score += 0.6  # Table content
    if content.strip().endswith(('.', '!', '?')): score += 0.1  # Complete sentences
    
    return max(score, 0.1)

def remove_duplicates_in_scraping(data_list):
    """Scraping aşamasında duplicate removal"""
    
    print("\n🧠 SCRAPING AŞAMASINDA DUPLICATE REMOVAL")
    print("=" * 50)
    
    if not data_list:
        return []
    
    # Content hash ile grupla
    hash_groups = {}
    
    for item in data_list:
        content_normalized = item['content'].lower().strip()
        content_hash = hashlib.md5(content_normalized.encode()).hexdigest()
        
        # Quality score hesapla ve item'a ekle
        item['quality_score'] = calculate_quality_score(item)
        item['content_hash'] = content_hash
        
        if content_hash not in hash_groups:
            hash_groups[content_hash] = []
        
        hash_groups[content_hash].append(item)
    
    # Her grup için en iyisini seç
    unique_data = []
    total_duplicates = 0
    
    for content_hash, items in hash_groups.items():
        if len(items) == 1:
            unique_data.append(items[0])
        else:
            # En yüksek quality score'lu olanı seç
            best_item = max(items, key=lambda x: x['quality_score'])
            unique_data.append(best_item)
            
            duplicates_count = len(items) - 1
            total_duplicates += duplicates_count
            
            print(f"🔄 {len(items)} duplicate bulundu:")
            print(f"  ✅ Korunan: {best_item['content'][:70]}... (Score: {best_item['quality_score']:.2f})")
            for item in items:
                if item != best_item:
                    print(f"  ❌ Atılan: {item['content'][:70]}... (Score: {item['quality_score']:.2f})")
    
    print(f"\n✅ Duplicate Removal Tamamlandı:")
    print(f"  📊 {len(data_list)} → {len(unique_data)} item")
    print(f"  🗑️ {total_duplicates} duplicate kaldırıldı")
    print(f"  📈 Veri kalitesi: %{((len(unique_data)/len(data_list))*100):.1f} retention")
    
    return unique_data

def scrape_all_turkish_airlines():
    """Tüm Turkish Airlines sayfalarını çek ve PostgreSQL'e kaydet"""
    
    pages = {
        'checked_baggage': 'https://www.turkishairlines.com/en-int/any-questions/checked-baggage/',
        'carry_on_baggage': 'https://www.turkishairlines.com/en-int/any-questions/carry-on-baggage/',
        'sports_equipment': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/',
        'musical_instruments': 'https://www.turkishairlines.com/en-int/any-questions/musical-instruments/',
        'pets': 'https://www.turkishairlines.com/en-int/any-questions/traveling-with-pets/',
        'excess_baggage': 'https://www.turkishairlines.com/en-int/any-questions/excess-baggage/',
        'restrictions': 'https://www.turkishairlines.com/en-int/any-questions/restrictions/'
    }
    
    all_data = []
    success_count = 0
    error_count = 0
    
    print(f"🚀 {len(pages)} sayfa scraping başlatılıyor...\n")
    
    # 1. Ham veriyi topla
    for page_name, url in pages.items():
        print(f"📡 {page_name.replace('_', ' ').title()} çekiliyor...")
        
        page_data = scrape_turkish_airlines_page(url, page_name)
        
        if page_data:
            all_data.extend(page_data)
            success_count += 1
            print(f"  ✅ {len(page_data)} ham veri alındı")
        else:
            error_count += 1
            print(f"  ❌ Veri alınamadı")
        
        print()
    
    print(f"📊 HAM VERİ TOPLAMA RAPORU:")
    print(f"  ✅ Başarılı: {success_count}/{len(pages)} sayfa")
    print(f"  ❌ Hatalı: {error_count}/{len(pages)} sayfa")
    print(f"  📋 Toplam ham veri: {len(all_data)} item")
    
    # 2. Duplicate removal uygula
    clean_data = remove_duplicates_in_scraping(all_data)
    
    # 3. PostgreSQL'e kaydet
    if clean_data:
        print(f"\n💾 PostgreSQL'e kayıt başlıyor...")
        saved_count = save_policies_to_postgresql(clean_data)
        
        if saved_count > 0:
            print(f"🎉 Başarılı! {saved_count} policy PostgreSQL'de hazır")
            return saved_count
        else:
            print(f"❌ PostgreSQL kayıt başarısız")
            return 0
    else:
        print(f"❌ Temizlenecek veri bulunamadı")
        return 0

def get_database_stats():
    """PostgreSQL'den istatistikleri getir"""
    
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Genel istatistikler
        cursor.execute("""
            SELECT 
                COUNT(*) as total_policies,
                COUNT(DISTINCT source) as total_sources,
                AVG(quality_score) as avg_quality_score,
                MAX(updated_at) as last_update
            FROM baggage_policies
        """)
        
        stats = dict(cursor.fetchone())
        
        # Source breakdown
        cursor.execute("""
            SELECT source, COUNT(*) as count, AVG(quality_score) as avg_quality
            FROM baggage_policies 
            GROUP BY source 
            ORDER BY count DESC
        """)
        
        source_breakdown = {}
        for row in cursor.fetchall():
            source_breakdown[row['source']] = {
                'count': row['count'],
                'avg_quality': round(float(row['avg_quality']), 2)
            }
        
        stats['source_breakdown'] = source_breakdown
        
        cursor.close()
        conn.close()
        
        return stats
        
    except Exception as e:
        print(f"❌ Stats alma hatası: {e}")
        return {}

# Ana execution (dosyanın sonunda)
if __name__ == "__main__":
    print("=" * 60)
    print("TURKISH AIRLINES SCRAPER - CONTAINER STARTUP")
    print("=" * 60)
    
    # 1. Database setup
    print("🔧 Database setup kontrol ediliyor...")
    if not setup_database():
        print("❌ Database setup başarısız - çıkılıyor")
        exit(1)
    
    # 2. Mevcut veri kontrolü
    existing_stats = get_database_stats()
    if existing_stats.get('total_policies', 0) > 0:
        print(f"📊 Database'de zaten {existing_stats['total_policies']} policy var")
        print("🤔 Yeniden scraping yapmak istiyor musunuz? (Container startup)")
        # Container'da otomatik evet diyoruz
        print("✅ Container startup - fresh scraping yapılıyor")
        clear_old_data()
    
    # 3. Scraping ve PostgreSQL'e kayıt
    saved_count = scrape_all_turkish_airlines()
    
    if saved_count > 0:
        print(f"\n📊 FINAL İSTATİSTİKLERİ:")
        stats = get_database_stats()
        
        if stats:
            print(f"  📋 Toplam policy: {stats.get('total_policies', 0)}")
            print(f"  📂 Toplam kaynak: {stats.get('total_sources', 0)}")
            print(f"  ⭐ Ortalama kalite: {stats.get('avg_quality_score', 0):.2f}")
            
            print(f"\n📊 KAYNAK DAĞILIMI:")
            for source, info in stats.get('source_breakdown', {}).items():
                print(f"  - {source}: {info['count']} policy (kalite: {info['avg_quality']})")
        
        print(f"\n🎯 CONTAINER STARTUP BAŞARILI:")
        print(f"✅ {saved_count} policy yüklendi")
        print(f"✅ Database hazır")
        print(f"🚀 FastAPI başlatılabilir!")
        
    else:
        print("⚠️ Hiç veri yüklenemedi - API yine de başlatılacak")
        exit(0)  # API'nin başlamasına izin ver
    # web_scrapper.py sonuna ekleyin

    # run_embedding fonksiyonunu aşağıdaki gibi güncelleyin
    async def run_embedding():
        """Doğrudan embedding işlemini çalıştır"""
        print("🚀 Doğrudan embedding işlemi başlatılıyor...")
        
        pool = None
        try:
            # Havuz oluştur
            pool = await asyncpg.create_pool(
                host=DB_CONFIG['host'],
                database=DB_CONFIG['database'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                min_size=1,
                max_size=5
            )
            
            print("🔗 Database havuzu oluşturuldu")
            
            # VectorOperations için havuzu doğrudan kullan
            vector_ops = VectorOperations(pool)
            count = await vector_ops.embed_existing_policies()
            print(f"✅ {count} politika embed edildi")
            
        except Exception as e:
            print(f"❌ Embedding hatası: {e}")
        finally:
            if pool:
                await pool.close()
                print("🔌 Database havuzu kapatıldı")

    # Ana kısımda, scraping başarılı olduktan sonra çağırın
    if saved_count > 0:
        asyncio.run(run_embedding())