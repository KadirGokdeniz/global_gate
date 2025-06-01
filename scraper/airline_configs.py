"""
airline_configs.py - Centralized airline configurations
Bu dosya tüm airline'ların scraping konfigürasyonlarını içerir
"""

# Base configuration template
BASE_CONFIG = {
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    },
    'request_delay': (1, 2),  # Random delay range
    'timeout': 30,
    'retry_attempts': 3
}

# Turkish Airlines Configuration
TURKISH_AIRLINES_CONFIG = {
    **BASE_CONFIG,
    'airline_id': 'turkish_airlines',
    'airline_name': 'Turkish Airlines',
    'base_url': 'https://www.turkishairlines.com',
    'pages': {
        'checked_baggage': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/checked-baggage/',
            'parsing_strategy': 'thy_standard'
        },
        'carry_on_baggage': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/carry-on-baggage/',
            'parsing_strategy': 'thy_standard'
        },
        'sports_equipment': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/',
            'parsing_strategy': 'thy_standard'
        },
        'musical_instruments': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/musical-instruments/',
            'parsing_strategy': 'thy_standard'
        },
        'pets': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/traveling-with-pets/',
            'parsing_strategy': 'thy_standard'
        },
        'excess_baggage': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/excess-baggage/',
            'parsing_strategy': 'thy_standard'
        },
        'restrictions': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/restrictions/',
            'parsing_strategy': 'thy_standard'
        }
    },
    'parsing_strategies': {
        'thy_standard': {
            'selectors': {
                'content_containers': [
                    '#page_wrapper .container .row li',
                    '#page_wrapper .container h2, #page_wrapper .container h3, #page_wrapper .container p',
                    '#page_wrapper table'
                ],
                'main_content': '#page_wrapper .container',
                'tables': '#page_wrapper table'
            },
            'filters': {
                'min_text_length': 10,
                'exclude_keywords': [
                    'home', 'menu', 'search', 'login', 'contact', 
                    'about', 'services', 'help', 'support', 'more information'
                ]
            }
        }
    }
}

# Pegasus Airlines Configuration - COMPLETE & FIXED
PEGASUS_CONFIG = {
    **BASE_CONFIG,
    'airline_id': 'pegasus',
    'airline_name': 'Pegasus Airlines',
    'base_url': 'https://www.flypgs.com',
    'pages': {
        'baggage_allowance': {
            'url': 'https://www.flypgs.com/en/pegasus-baggage-allowance',
            'parsing_strategy': 'pegasus_accordion_v1'
        },
        'general_rules': {
            'url': 'https://www.flypgs.com/en/useful-info/info-about-flights/general-rules',
            'parsing_strategy': 'pegasus_accordion_v2'
        },
        'extra_services_pricing': {
            'url': 'https://www.flypgs.com/en/useful-info/other-info/extra-services-price-table',
            'parsing_strategy': 'pegasus_price_table'
        },
        'travelling_with_pets': {
            'url': 'https://www.flypgs.com/en/travelling-with-pets',
            'parsing_strategy': 'pegasus_cms_content'
        }
    },
    'parsing_strategies': {
        'pegasus_accordion_v1': {
            'selectors': {
                # Version 1: n-faq-acc prefix (baggage-allowance style)
                'main_container': '.n-faq-acc__container',
                'faq_items': '.n-faq-acc__item',
                'faq_headers': '.n-faq-acc__header',
                'faq_content': '.n-faq-acc__content',
                
                # İçerik alanları
                'content_containers': [
                    '.cms-content-default p',
                    '.cms-content-default li', 
                    '.cms-content-default h1, .cms-content-default h2, .cms-content-default h3',
                    '.cms-content-default table'
                ],
                'main_content': '.cms-content-default, .n-faq-acc__content',
                'tables': '.cms-content-default table'
            },
            'filters': {
                'min_text_length': 15,
                'exclude_keywords': [
                    'anasayfa', 'menü', 'arama', 'giriş', 'iletişim', 
                    'hakkında', 'hizmetler', 'yardım', 'destek',
                    'home', 'menu', 'search', 'login', 'contact',
                    'target here'
                ],
                'skip_empty_accordions': True,
                'combine_header_content': True
            }
        },
        
        'pegasus_accordion_v2': {
            'selectors': {
                # Version 2: faq-acc prefix (general-rules style)
                'main_container': '.faq-acc__container',
                'faq_items': '.faq-acc__item',
                'faq_headers': '.faq-acc__header',
                'faq_content': '.faq-acc__content',
                
                # İçerik alanları (aynı)
                'content_containers': [
                    '.cms-content-default p',
                    '.cms-content-default li', 
                    '.cms-content-default h1, .cms-content-default h2, .cms-content-default h3',
                    '.cms-content-default table'
                ],
                'main_content': '.cms-content-default, .faq-acc__content',
                'tables': '.cms-content-default table'
            },
            'filters': {
                'min_text_length': 15,
                'exclude_keywords': [
                    'anasayfa', 'menü', 'arama', 'giriş', 'iletişim', 
                    'hakkında', 'hizmetler', 'yardım', 'destek',
                    'home', 'menu', 'search', 'login', 'contact',
                    'target here'
                ],
                'skip_empty_accordions': True,
                'combine_header_content': True
            }
        },
        
        'pegasus_price_table': {
            'selectors': {
                # Ana tablo container
                'main_container': '.c-table-ft',
                'table_body': '.pgs-table-ft__body',
                
                # Hizmet türü başlıkları (Domestic, International, KKTC)
                'service_titles': '.pgs-table-ft__title',
                
                # Her hizmet türü için liste container
                'service_lists': '.pgs-table-ft__list',
                
                # Tablo satırları ve hücreleri
                'table_rows': '.pgs-table-ft__tr',
                'table_cells': '.pgs-table-ft__td',
                
                # Mobil başlıklar ve içerik
                'mobile_headers': '.pgs-table-ft-mobil__th',
                'cell_titles': '.pgs-table-ft__td__title',
                
                # Fallback selectors
                'content_containers': [
                    '.pgs-table-ft__td',
                    '.pgs-table-ft__title',
                    '.c-table-ft table'
                ],
                'tables': '.c-table-ft table, .pgs-table-ft__list'
            },
            'filters': {
                'min_text_length': 5,  # Shorter for price/code data
                'exclude_keywords': [
                    'target here', 'placeholder',
                    'anasayfa', 'menü', 'arama', 'giriş', 'iletişim'
                ],
                'price_indicators': ['TL', '€', '$', 'USD', 'EUR'],
                'service_categories': ['domestic', 'international', 'kktc', 'flights']
            }
        },
        
        'pegasus_cms_content': {
            'selectors': {
                # Direct CMS content (pets page style)
                'main_container': '.cms-content-default',
                
                # Content elements
                'paragraphs': '.cms-content-default p',
                'strong_elements': '.cms-content-default strong',
                'headers': '.cms-content-default h1, .cms-content-default h2, .cms-content-default h3',
                'list_items': '.cms-content-default li',
                
                # Fallback selectors
                'content_containers': [
                    '.cms-content-default p',
                    '.cms-content-default strong',
                    '.cms-content-default h1, .cms-content-default h2, .cms-content-default h3',
                    '.cms-content-default li'
                ],
                'main_content': '.cms-content-default, .zone-section__container',
                'tables': '.cms-content-default table'
            },
            'filters': {
                'min_text_length': 10,
                'exclude_keywords': [
                    'target here', 'placeholder',
                    'anasayfa', 'menü', 'arama', 'giriş', 'iletişim',
                    'home', 'menu', 'search', 'login', 'contact'
                ],
                'content_indicators': ['pet', 'animal', 'dog', 'cat', 'travel', 'cabin', 'cargo'],
                'extract_strong_as_headers': True  # Strong elements treated as section headers
            }
        },
        
        'pegasus_universal': {
            'selectors': {
                # Try both versions
                'faq_items': '.n-faq-acc__item, .faq-acc__item',
                'faq_headers': '.n-faq-acc__header, .faq-acc__header',
                'faq_content': '.n-faq-acc__content, .faq-acc__content',
                
                'content_containers': [
                    '.main-wrapper p',
                    '.main-wrapper li',
                    '.main-wrapper h2, .main-wrapper h3',
                    '.main-wrapper table'
                ],
                'main_content': '.main-wrapper, .container',
                'tables': '.main-wrapper table'
            },
            'filters': {
                'min_text_length': 15,
                'exclude_keywords': [
                    'anasayfa', 'menü', 'arama', 'giriş', 'iletişim', 
                    'hakkında', 'hizmetler', 'yardım', 'destek'
                ]
            }
        }
    }
}

# Master airline registry
AIRLINE_CONFIGS = {
    'turkish_airlines': TURKISH_AIRLINES_CONFIG,
    'pegasus': PEGASUS_CONFIG
}

# Helper functions
def get_airline_config(airline_id: str):
    """Get specific airline configuration"""
    return AIRLINE_CONFIGS.get(airline_id)

def get_all_airlines():
    """Get list of all supported airlines"""
    return list(AIRLINE_CONFIGS.keys())

def get_all_pages_for_airline(airline_id: str):
    """Get all pages for specific airline"""
    config = get_airline_config(airline_id)
    return config['pages'] if config else {}

def validate_airline_config(airline_id: str):
    """Validate airline configuration"""
    config = get_airline_config(airline_id)
    if not config:
        return False, f"Airline {airline_id} not found"
    
    required_fields = ['airline_id', 'airline_name', 'base_url', 'pages', 'parsing_strategies']
    for field in required_fields:
        if field not in config:
            return False, f"Missing required field: {field}"
    
    # Check if all pages have valid strategies
    for page_name, page_config in config['pages'].items():
        strategy_name = page_config.get('parsing_strategy')
        if strategy_name not in config['parsing_strategies']:
            return False, f"Page '{page_name}' references undefined strategy '{strategy_name}'"
    
    return True, "Valid configuration"

def get_airline_summary():
    """Get summary of all configured airlines"""
    summary = {}
    
    for airline_id in get_all_airlines():
        config = get_airline_config(airline_id)
        summary[airline_id] = {
            'name': config['airline_name'],
            'base_url': config['base_url'],
            'total_pages': len(config['pages']),
            'pages': list(config['pages'].keys()),
            'strategies': list(config['parsing_strategies'].keys())
        }
    
    return summary

# Usage example and testing
if __name__ == "__main__":
    print("🔍 AIRLINE CONFIGURATIONS TEST")
    print("=" * 50)
    
    # Test all configurations
    for airline_id in get_all_airlines():
        valid, message = validate_airline_config(airline_id)
        status = "✅" if valid else "❌"
        print(f"{status} {airline_id}: {message}")
        
        if valid:
            config = get_airline_config(airline_id)
            print(f"   📄 Pages: {list(config['pages'].keys())}")
            print(f"   🎯 Strategies: {list(config['parsing_strategies'].keys())}")
        print()
    
    # Summary
    summary = get_airline_summary()
    print(f"📊 SUMMARY:")
    print(f"   Total Airlines: {len(summary)}")
    for airline_id, info in summary.items():
        print(f"   - {info['name']}: {info['total_pages']} pages")
    
    print("\n🎉 Configuration test completed!")