"""
airline_configs.py - Hybrid Configuration
THY: Page-specific selectors
Pegasus: Parsing strategies (sophisticated logic korunuyor)
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

# Turkish Airlines Configuration - PAGE-SPECIFIC SELECTORS
TURKISH_AIRLINES_CONFIG = {
    **BASE_CONFIG,
    'airline_id': 'turkish_airlines',
    'airline_name': 'Turkish Airlines',
    'base_url': 'https://www.turkishairlines.com',
    'pages': {
        # Main pages with page-specific selectors
        'checked_baggage': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/checked-baggage/',
            'selectors': ['#page_wrapper .container .row li']
        },
        'carry_on_baggage': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/carry-on-baggage/',
            'selectors': [
                '#page_wrapper div[style*="text-align: justify; font-size: 14pt;"]',
                '#page_wrapper .col-xs-height.col-middle',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_equipment': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/',
            'selectors': ['#page_wrapper .container .row li']
        },
        'musical_instruments': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/musical-instruments/',
            'selectors': [
                '#page_wrapper .container .row .col-md-9 p',
                '#page_wrapper .container .row li'
            ]
        },
        'pets': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/traveling-with-pets/',
            'selectors': ['#tcm508-364188 .row p']
        },
        'excess_baggage': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/excess-baggage/',
            'selectors': [
                '#page_wrapper .container .row li',
                '#tcm508-275552 .container .row .col-md-12 .row .col-md-6 div'
            ]
        },
        'restrictions': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/restrictions/',
            'selectors': ['#page_wrapper table']
        },
        
        # Sports equipment detailed pages
        'sports_bicycle': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/bicycle/',
            'selectors': [
                '#page_wrapper .container .row li',
                '#tcm508-361154 .middle-wrapper .container .row .col-12 .card-body .dflex .col-xs-height p'
            ]
        },
        'sports_mountaineering': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/mountaineering/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_golf': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/golf/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li',
                '#tcm508-365094 .middle-wrapper .container .row .col-12 .card-body .dflex .col-xs-height'
            ]
        },
        'sports_canoeing': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/canoeing/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_skiing': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/skiing-snowboard/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_archery': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/archery/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_parachuting': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/parachuting-paragliding/',
            'selectors': ['#page_wrapper .container .row li']
        },
        'sports_rafting': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/rafting-inflatable-boat/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_surfing': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/surfing/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li',
                '.middle-wrapper .container .row .col-12 .card-body .dflex .col-xs-height'
            ]
        },
        'sports_windsurfing': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/windsurfing/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li',
                '.middle-wrapper .container .row .col-12 .card-body .dflex .col-xs-height'
            ]
        },
        'sports_water_skiing': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/water-skiing/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_diving': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/diving/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li',
                '.middle-wrapper .container .row .col-12 .card-body .dflex .col-xs-height'
            ]
        },
        'sports_hockey': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/hockey-lacrosse/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_bowling': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/bowling/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_tenting': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/tenting/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_fishing': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/fishing/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        'sports_hunting': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/sports-equipment/hunting/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        },
        
        # Pets detailed pages  
        'pets_cargo': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/traveling-with-pets/transport-in-the-cargo-compartment/',
            'selectors': [
                '#page_wrapper .container .row li',
                '.middle-wrapper .container .row .col-12 .card-body .dflex .col-xs-height'
            ]
        },
        'pets_country_rules': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/traveling-with-pets/country-based-situations/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li',
                '.middle-wrapper .container .row .col-12 .card-body .dflex .col-xs-height',
                '#tcm508-364158 .container .row div .TypographyPresentation'
            ]
        },
        'pets_terms': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/traveling-with-pets/all-terms-and-conditions/',
            'selectors': ['#page_wrapper .container .row li']
        },
        'pets_onboard': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/traveling-with-pets/pets-allowed-on-board/',
            'selectors': ['#page_wrapper .container .row li']
        },
        'pets_service_animals': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/traveling-with-pets/service-animals/',
            'selectors': ['#page_wrapper .container .row li']
        },
        'pets_cabin': {
            'url': 'https://www.turkishairlines.com/en-int/any-questions/traveling-with-pets/transport-in-the-cabin/',
            'selectors': [
                '#page_wrapper .container .row p',
                '#page_wrapper .container .row li'
            ]
        }
    }
    # THY'de parsing_strategies yok - sadece page-specific selectors
}

# Pegasus Airlines Configuration - PARSING STRATEGIES (SOPHISTICATED LOGIC KORUNUYOR)
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
            'parsing_strategy': 'pegasus_cms_content'
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
    
    required_fields = ['airline_id', 'airline_name', 'base_url', 'pages']
    for field in required_fields:
        if field not in config:
            return False, f"Missing required field: {field}"
    
    # Check pages have either selectors or parsing_strategy
    for page_name, page_config in config['pages'].items():
        has_selectors = 'selectors' in page_config
        has_strategy = 'parsing_strategy' in page_config
        
        if not has_selectors and not has_strategy:
            return False, f"Page '{page_name}' has neither selectors nor parsing_strategy"
        
        # If has parsing_strategy, check if it exists
        if has_strategy:
            strategy_name = page_config['parsing_strategy']
            if 'parsing_strategies' not in config or strategy_name not in config['parsing_strategies']:
                return False, f"Page '{page_name}' references undefined strategy '{strategy_name}'"
    
    return True, "Valid hybrid configuration"

def get_airline_summary():
    """Get summary of all configured airlines"""
    summary = {}
    
    for airline_id in get_all_airlines():
        config = get_airline_config(airline_id)
        
        # Count page types
        page_specific_count = 0
        strategy_count = 0
        
        for page_config in config['pages'].values():
            if 'selectors' in page_config:
                page_specific_count += 1
            elif 'parsing_strategy' in page_config:
                strategy_count += 1
        
        summary[airline_id] = {
            'name': config['airline_name'],
            'base_url': config['base_url'],
            'total_pages': len(config['pages']),
            'page_specific_selectors': page_specific_count,
            'parsing_strategies': strategy_count,
            'strategy_names': list(config.get('parsing_strategies', {}).keys())
        }
    
    return summary

# Usage example and testing
if __name__ == "__main__":
    print("🔍 HYBRID AIRLINE CONFIGURATIONS TEST")
    print("=" * 50)
    
    # Test all configurations
    for airline_id in get_all_airlines():
        valid, message = validate_airline_config(airline_id)
        status = "✅" if valid else "❌"
        print(f"{status} {airline_id}: {message}")
        
        if valid:
            config = get_airline_config(airline_id)
            print(f"   📄 Pages: {list(config['pages'].keys())}")
            if 'parsing_strategies' in config:
                print(f"   🎯 Strategies: {list(config['parsing_strategies'].keys())}")
            else:
                print(f"   🎯 Uses: Page-specific selectors only")
        print()
    
    # Summary
    summary = get_airline_summary()
    print(f"📊 HYBRID SUMMARY:")
    print(f"   Total Airlines: {len(summary)}")
    for airline_id, info in summary.items():
        print(f"   - {info['name']}:")
        print(f"     • Total pages: {info['total_pages']}")
        print(f"     • Page-specific: {info['page_specific_selectors']}")
        print(f"     • Strategy-based: {info['parsing_strategies']}")
    
    print("\n🎉 Hybrid configuration test completed!")