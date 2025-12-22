"""
Updated web_scrapper.py - Multi-airline wrapper
Bu dosya artık base_scraper.py'ı kullanır ve backward compatibility sağlar
"""

import os
import logging
from scraper.core.base_scraper import (
    MultiAirlineScraper,
    scrape_all_airlines as base_scrape_all
)
from scraper.configs.airline_configs import get_all_airlines, get_airline_config

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backward compatibility için eski fonksiyonlar
def get_db_connection():
    """Backward compatibility wrapper"""
    scraper = MultiAirlineScraper()
    return scraper.get_db_connection()

def setup_database():
    """Backward compatibility wrapper"""
    scraper = MultiAirlineScraper()
    return scraper.setup_database()

def clear_old_data():
    """Backward compatibility - tüm veriyi temizle"""
    scraper = MultiAirlineScraper()
    return scraper.clear_all_data()

def save_policies_to_postgresql(policies):
    """Backward compatibility wrapper"""
    scraper = MultiAirlineScraper()
    return scraper.save_policies_to_database(policies)

def get_database_stats():
    """Enhanced database stats"""
    scraper = MultiAirlineScraper()
    return scraper.get_database_stats()

def scrape_all_turkish_airlines():
    """Backward compatibility - sadece THY scrape et"""
    logger.info("⚠️ UYARI: scrape_all_turkish_airlines() deprecated!")
    logger.info("✅ Yeni fonksiyon: scrape_specific_airline('turkish_airlines')")
    
    scraper = MultiAirlineScraper()
    return scraper.scrape_airline('turkish_airlines')

# === YENİ MULTI-AIRLINE FONKSİYONLARI ===

def scrape_specific_airline(airline_id: str) -> int:
    """Belirli bir havayolunu scrape et"""
    scraper = MultiAirlineScraper()
    return scraper.scrape_airline(airline_id)

def scrape_all_airlines() -> dict:
    """Tüm havayollarını scrape et"""
    return base_scrape_all()

def scrape_turkish_airlines() -> int:
    """Sadece Turkish Airlines'ı scrape et"""
    return scrape_specific_airline('turkish_airlines')

def scrape_pegasus() -> int:
    """Sadece Pegasus'u scrape et"""
    return scrape_specific_airline('pegasus')

def clear_airline_data(airline_id: str) -> bool:
    """Belirli havayolunun verisini temizle"""
    scraper = MultiAirlineScraper()
    return scraper.clear_airline_data(airline_id)

def get_supported_airlines() -> list:
    """Desteklenen havayolları listesi"""
    return get_all_airlines()

def validate_airline(airline_id: str) -> bool:
    """Havayolu ID'si geçerli mi kontrol et"""
    return airline_id in get_all_airlines()

def get_airline_info(airline_id: str) -> dict:
    """Havayolu bilgilerini getir"""
    config = get_airline_config(airline_id)
    if not config:
        return {}
    
    return {
        'airline_id': config['airline_id'],
        'airline_name': config['airline_name'],
        'base_url': config['base_url'],
        'total_pages': len(config['pages']),
        'pages': list(config['pages'].keys())
    }

def print_airline_summary():
    """Desteklenen havayolları özeti"""
    print("\n" + "=" * 60)
    print("📋 DESTEKLENEN HAVAYOLLARI")
    print("=" * 60)
    
    airlines = get_all_airlines()
    
    for airline_id in airlines:
        info = get_airline_info(airline_id)
        print(f"\n✈️ {info['airline_name']} ({airline_id})")
        print(f"   🌐 Base URL: {info['base_url']}")
        print(f"   📄 Pages: {info['total_pages']}")
        print(f"   📋 Sources: {', '.join(info['pages'])}")
    
    print(f"\n📊 Toplam {len(airlines)} havayolu destekleniyor")
    print("=" * 60)

# === ENHANCED REPORTING ===

def get_detailed_stats():
    """Detaylı istatistikler"""
    scraper = MultiAirlineScraper()
    stats = scraper.get_database_stats()
    
    if not stats:
        return "❌ Database bağlantısı kurulamadı"
    
    report = []
    report.append("\n" + "=" * 60)
    report.append("📊 MULTI-AIRLINE DATABASE STATS")
    report.append("=" * 60)
    
    # Genel istatistikler
    report.append(f"📋 Toplam Policy: {stats.get('total_policies', 0)}")
    report.append(f"✈️ Toplam Airline: {stats.get('total_airlines', 0)}")
    report.append(f"📂 Toplam Source: {stats.get('total_sources', 0)}")
    report.append(f"⭐ Ortalama Kalite: {stats.get('avg_quality_score', 0):.2f}")
    
    # Airline breakdown
    if 'airline_breakdown' in stats:
        report.append(f"\n📊 AIRLINE BREAKDOWN:")
        for airline, info in stats['airline_breakdown'].items():
            airline_info = get_airline_info(airline)
            airline_name = airline_info.get('airline_name', airline)
            
            report.append(f"  ✈️ {airline_name}:")
            report.append(f"     📄 Policies: {info['count']}")
            report.append(f"     📂 Sources: {info['sources_count']}")
            report.append(f"     ⭐ Avg Quality: {info['avg_quality']}")
    
    report.append("=" * 60)
    
    return "\n".join(report)

# === MAIN EXECUTION LOGIC ===

def main_scraping_process(selected_airlines: list = None):
    """Ana scraping süreci - airline seçimi ile"""
    
    print("=" * 70)
    print("🚀 MULTI-AIRLINE SCRAPER - ENHANCED")
    print("=" * 70)
    
    # Airline seçimi
    if not selected_airlines:
        print("📋 Mevcut airline'lar:")
        print_airline_summary()
        
        print("\n🎯 TÜM AIRLINE'LAR SCRAPE EDİLECEK")
        selected_airlines = get_all_airlines()
    else:
        print(f"🎯 Seçilen airline'lar: {selected_airlines}")
    
    # Validation
    invalid_airlines = [a for a in selected_airlines if not validate_airline(a)]
    if invalid_airlines:
        print(f"❌ Geçersiz airline'lar: {invalid_airlines}")
        print(f"✅ Geçerli airline'lar: {get_all_airlines()}")
        return False
    
    # Database setup
    print("\n🔧 Database setup kontrol ediliyor...")
    if not setup_database():
        print("❌ Database setup başarısız")
        return False
    
    # Mevcut veri kontrolü
    existing_stats = get_database_stats()
    if existing_stats.get('total_policies', 0) > 0:
        print(f"\n📊 Database'de mevcut veri:")
        print(get_detailed_stats())
        
        print("\n🧹 Eski veri temizlensin mi? (Container startup - otomatik evet)")
        clear_old_data()
    
    # Scraping process
    results = {}
    
    for airline_id in selected_airlines:
        print(f"\n🚀 {airline_id.upper()} SCRAPING BAŞLANIYOR")
        print("-" * 50)
        
        try:
            count = scrape_specific_airline(airline_id)
            results[airline_id] = count
            
            if count > 0:
                print(f"✅ {airline_id}: {count} policy başarıyla scrape edildi")
            else:
                print(f"⚠️ {airline_id}: Hiç veri alınamadı")
                
        except Exception as e:
            print(f"❌ {airline_id} scraping hatası: {e}")
            results[airline_id] = 0
    
    # Final summary
    total_scraped = sum(results.values())
    successful_airlines = len([k for k, v in results.items() if v > 0])
    
    print(f"\n🎯 SCRAPING SÜRECI TAMAMLANDI!")
    print("=" * 70)
    print(f"📊 Toplam Policy: {total_scraped}")
    print(f"✅ Başarılı Airline: {successful_airlines}/{len(selected_airlines)}")
    print(f"📋 Detay: {results}")
    
    # Final stats
    if total_scraped > 0:
        print(get_detailed_stats())
        return True
    else:
        print("⚠️ Hiç veri scrape edilemedi")
        return False

# Entry point için backward compatibility
if __name__ == "__main__":
    
    import sys
    
    if len(sys.argv) > 1:
        # Command line arguments
        if sys.argv[1] == "--thy-only":
            print("🎯 Sadece Turkish Airlines scraping")
            main_scraping_process(['turkish_airlines'])
        elif sys.argv[1] == "--pegasus-only":
            print("🎯 Sadece Pegasus scraping")
            main_scraping_process(['pegasus'])
        elif sys.argv[1] == "--list-airlines":
            print_airline_summary()
        else:
            print("❌ Geçersiz argument")
            print("✅ Kullanım: python web_scrapper.py [--thy-only|--pegasus-only|--list-airlines]")
    else:
        # Default: tüm airline'lar
        print("🌍 Tüm airline'lar scrape edilecek")
        main_scraping_process()