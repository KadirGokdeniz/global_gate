"""
base_scraper.py - Hybrid scraper for all airlines
THY: Page-specific selectors
Pegasus: Parsing strategies (sophisticated logic)
"""

from bs4 import BeautifulSoup
import requests
import time
import random
import hashlib
import re
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import List, Dict, Optional, Tuple
import logging
from airline_configs import get_airline_config, get_all_airlines

logger = logging.getLogger(__name__)

# PostgreSQL Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'db'),
    'database': os.getenv('DB_DATABASE', 'global_gate'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'qeqe')
}

class MultiAirlineScraper:
    """Hybrid scraper: THY page-specific selectors, Pegasus parsing strategies"""
    
    def __init__(self):
        self.connection = None
        self.setup_database()
    
    def get_db_connection(self):
        """PostgreSQL bağlantısı kur"""
        try:
            if not self.connection or self.connection.closed:
                logger.info(f"🔗 Database'e bağlanıyor: {DB_CONFIG['host']}:{DB_CONFIG['database']}")
                self.connection = psycopg2.connect(**DB_CONFIG)
                logger.info("✅ PostgreSQL bağlantısı başarılı")
            return self.connection
        except Exception as e:
            logger.error(f"❌ PostgreSQL bağlantı hatası: {e}")
            return None
    
    def setup_database(self):
        """Database ve tabloları oluştur"""
        conn = self.get_db_connection()
        if not conn:
            logger.error("❌ Database bağlantısı kurulamadı")
            return False
        
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'policy'
            """)
            
            columns = [row[0] for row in cursor.fetchall()]
            
            if 'airline' not in columns:
                logger.error("❌ Database schema güncel değil! init.sql güncellenmeli!")
                return False
            
            logger.info("✅ Multi-airline database schema hazır")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup hatası: {e}")
            return False
    
    def scrape_page(self, airline_id: str, page_name: str, page_config: dict) -> List[Dict]:
        """Hybrid page scraping - page-specific selectors OR parsing strategy"""
        
        print(f"DEBUG: Page config keys: {page_config.keys()}")
        if 'selectors' in page_config:
            print(f"DEBUG: Using page-specific selectors: {len(page_config['selectors'])}")
        elif 'parsing_strategy' in page_config:
            print(f"DEBUG: Using parsing strategy: {page_config['parsing_strategy']}")
        
        config = get_airline_config(airline_id)
        if not config:
            logger.error(f"❌ Airline config bulunamadı: {airline_id}")
            return []
        
        url = page_config['url']
        logger.info(f"📡 {airline_id} - {page_name} scraping: {url}")
        
        try:
            # Request delay
            delay_range = config.get('request_delay', (1, 2))
            time.sleep(random.uniform(*delay_range))
            
            # Make request
            headers = config.get('headers', {})
            timeout = config.get('timeout', 30)
            
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # HYBRID APPROACH: Route to appropriate parsing method
            if 'selectors' in page_config:
                # THY approach - page-specific selectors
                custom_strategy = {
                    'selectors': {
                        'content_containers': page_config['selectors']
                    },
                    'filters': page_config.get('filters', {
                        'min_text_length': 15,
                        'exclude_keywords': ['menu', 'home', 'target here', 'search', 'login', 'contact']
                    })
                }
                scraped_data = self._apply_page_specific_parsing(
                    soup, airline_id, page_name, url, custom_strategy
                )
            
            elif 'parsing_strategy' in page_config:
                # Pegasus approach - sophisticated parsing strategies
                strategy_name = page_config['parsing_strategy']
                strategy = config['parsing_strategies'][strategy_name]
                scraped_data = self._apply_strategy_parsing(
                    soup, airline_id, page_name, url, strategy
                )
            
            else:
                logger.error(f"❌ {page_name}: Ne selectors ne de parsing_strategy var!")
                return []
            
            logger.info(f"  ✅ {len(scraped_data)} item çekildi")
            return scraped_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"  ❌ Request error: {e}")
            return []
        except Exception as e:
            logger.error(f"  ❌ Parsing error: {e}")
            return []
    
    def _apply_page_specific_parsing(self, soup: BeautifulSoup, airline_id: str, 
                          page_name: str, url: str, strategy: dict) -> List[Dict]:
        """Simple page-specific selector parsing (THY style)"""
        
        selectors = strategy['selectors']
        filters = strategy['filters']
        scraped_data = []
        
        logger.info(f"DEBUG: Page-specific parsing with {len(selectors.get('content_containers', []))} selectors")
        
        # Process ALL content_containers selectors
        list_selectors = selectors.get('content_containers', [])
        if isinstance(list_selectors, list) and list_selectors:
            
            for selector_idx, list_selector in enumerate(list_selectors):
                elements = soup.select(list_selector)
                
                if elements:
                    logger.info(f"  └── Selector {selector_idx} ({list_selector}): {len(elements)} element bulundu")
                    for element in elements:
                        text = element.get_text(strip=True)
                        if self._is_valid_content(text, filters):
                            scraped_data.append({
                                'airline': airline_id,
                                'source': page_name,
                                'content': text,
                                'url': url,
                                'type': f'page_specific_{selector_idx}'
                            })
                else:
                    logger.info(f"  └── Selector {selector_idx} ({list_selector}): 0 element bulundu")
        
        # Table extraction
        tables = soup.select('table')
        if tables:
            logger.info(f"  └── Tablo yapısı: {len(tables)} table bulundu")
            for table_idx, table in enumerate(tables):
                table_data = self._extract_table_data(
                    table, airline_id, page_name, url, table_idx
                )
                scraped_data.extend(table_data)
        
        logger.info(f"  └── PAGE-SPECIFIC TOTAL: {len(scraped_data)} item")
        return scraped_data
    
    def _apply_strategy_parsing(self, soup: BeautifulSoup, airline_id: str, 
                          page_name: str, url: str, strategy: dict) -> List[Dict]:
        """Sophisticated parsing strategy (Pegasus style)"""
        
        selectors = strategy['selectors']
        filters = strategy['filters']
        scraped_data = []
        
        logger.info(f"DEBUG: Strategy parsing for sophisticated extraction")
        
        # PEGASUS SPECIAL: CMS Content parsing (pets page style)
        if 'paragraphs' in selectors and 'strong_elements' in selectors:
            logger.info(f"  └── CMS Content parsing strategy")
            cms_data = self._parse_cms_content(
                soup, airline_id, page_name, url, selectors, filters
            )
            if cms_data:
                scraped_data.extend(cms_data)
        
        # PEGASUS SPECIAL: Price table parsing
        if 'service_titles' in selectors and 'table_rows' in selectors:
            logger.info(f"  └── Price table parsing strategy")
            price_table_data = self._parse_price_table_content(
                soup, airline_id, page_name, url, selectors, filters
            )
            if price_table_data:
                scraped_data.extend(price_table_data)
        
        # PEGASUS SPECIAL: Accordion-based parsing
        if 'faq_items' in selectors:
            logger.info(f"  └── Accordion parsing strategy")
            accordion_data = self._parse_accordion_content(
                soup, airline_id, page_name, url, selectors, filters
            )
            if accordion_data:
                scraped_data.extend(accordion_data)
        
        # Fallback: Basic content extraction
        if not scraped_data:
            logger.info(f"  └── Fallback: Basic content extraction")
            list_selectors = selectors.get('content_containers', [])
            if isinstance(list_selectors, list) and list_selectors:
                
                for selector_idx, list_selector in enumerate(list_selectors):
                    elements = soup.select(list_selector)
                    
                    if elements:
                        logger.info(f"    └── Fallback selector {selector_idx}: {len(elements)} element")
                        for element in elements:
                            text = element.get_text(strip=True)
                            if self._is_valid_content(text, filters):
                                scraped_data.append({
                                    'airline': airline_id,
                                    'source': page_name,
                                    'content': text,
                                    'url': url,
                                    'type': f'fallback_{selector_idx}'
                                })
        
        logger.info(f"  └── STRATEGY TOTAL: {len(scraped_data)} item")
        return scraped_data
    
    def _parse_cms_content(self, soup: BeautifulSoup, airline_id: str, 
                          page_name: str, url: str, selectors: dict, filters: dict) -> List[Dict]:
        """Parse direct CMS content (pets page style) - structured content extraction"""
        
        cms_data = []
        
        # Find main CMS container
        main_container = soup.select_one(selectors['main_container'])
        if not main_container:
            logger.warning(f"    ❌ No CMS content container found")
            return []
        
        logger.info(f"    🎯 Processing CMS content container")
        
        # Strategy 1: Extract structured content (strong headers + paragraphs)
        current_section = None
        section_content = []
        
        # Get all content elements in order
        all_elements = main_container.find_all(['p', 'strong', 'h1', 'h2', 'h3', 'li'])
        
        for elem_idx, element in enumerate(all_elements):
            try:
                text = element.get_text(strip=True)
                text = text.replace("Target here", "").strip()
                text = ' '.join(text.split())  # Clean extra spaces
                
                if not text or len(text) < 5:
                    continue
                
                # Check if this is a header/section title
                is_header = (
                    element.name in ['h1', 'h2', 'h3'] or 
                    element.name == 'strong' or
                    (element.name == 'p' and element.find('strong'))
                )
                
                if is_header and filters.get('extract_strong_as_headers', False):
                    # Save previous section if exists
                    if current_section and section_content:
                        combined_content = f"Section: {current_section}\n" + "\n".join(section_content)
                        
                        if self._is_valid_content(combined_content, filters):
                            cms_data.append({
                                'airline': airline_id,
                                'source': page_name,
                                'content': combined_content,
                                'url': url,
                                'type': 'cms_section',
                                'metadata': {
                                    'section_title': current_section,
                                    'content_parts': len(section_content),
                                    'element_index': elem_idx
                                }
                            })
                    
                    # Start new section
                    current_section = text
                    section_content = []
                    
                    # Also save standalone header
                    cms_data.append({
                        'airline': airline_id,
                        'source': page_name,
                        'content': f"Topic: {text}",
                        'url': url,
                        'type': 'cms_header',
                        'metadata': {
                            'header_text': text,
                            'element_type': element.name,
                            'element_index': elem_idx
                        }
                    })
                    
                else:
                    # This is content under current section
                    if current_section:
                        section_content.append(text)
                    
                    # Also save as standalone content
                    if self._is_valid_content(text, filters):
                        content_type = 'cms_paragraph'
                        if element.name == 'li':
                            content_type = 'cms_list_item'
                        elif current_section:
                            content_type = 'cms_section_content'
                        
                        cms_data.append({
                            'airline': airline_id,
                            'source': page_name,
                            'content': text,
                            'url': url,
                            'type': content_type,
                            'metadata': {
                                'parent_section': current_section,
                                'element_type': element.name,
                                'element_index': elem_idx
                            }
                        })
                        
            except Exception as e:
                logger.warning(f"    ⚠️ CMS element {elem_idx} parsing error: {e}")
                continue
        
        # Save final section if exists
        if current_section and section_content:
            combined_content = f"Section: {current_section}\n" + "\n".join(section_content)
            
            if self._is_valid_content(combined_content, filters):
                cms_data.append({
                    'airline': airline_id,
                    'source': page_name,
                    'content': combined_content,
                    'url': url,
                    'type': 'cms_section',
                    'metadata': {
                        'section_title': current_section,
                        'content_parts': len(section_content),
                        'is_final_section': True
                    }
                })
        
        logger.info(f"    ✅ Extracted {len(cms_data)} CMS content items")
        return cms_data
    
    def _parse_price_table_content(self, soup: BeautifulSoup, airline_id: str, 
                                  page_name: str, url: str, selectors: dict, filters: dict) -> List[Dict]:
        """Parse Pegasus price table content - Domestic/International/KKTC tables"""
        
        price_data = []
        
        # Find main table container
        main_container = soup.select_one(selectors['main_container'])
        if not main_container:
            logger.warning(f"    ❌ No main price table container found")
            return []
        
        # Find service titles (Domestic, International, KKTC)
        service_sections = self._extract_service_sections(main_container, selectors)
        
        logger.info(f"    🎯 Found {len(service_sections)} service sections")
        
        for section_idx, (service_title, section_data) in enumerate(service_sections.items()):
            logger.info(f"    📊 Processing {service_title} section...")
            
            # Process each row in this service section
            for row_idx, row_data in enumerate(section_data):
                try:
                    # Create structured price entry
                    if len(row_data) >= 2:  # At least code and description
                        short_code = row_data[0] if row_data[0] else f"Item-{row_idx}"
                        description = row_data[1] if len(row_data) > 1 else ""
                        price = row_data[2] if len(row_data) > 2 else "Price not specified"
                        
                        # Create formatted content
                        price_entry = f"Service: {service_title} | Code: {short_code} | Description: {description} | Price: {price}"
                        
                        if self._is_valid_content(price_entry, filters):
                            price_data.append({
                                'airline': airline_id,
                                'source': page_name,
                                'content': price_entry,
                                'url': url,
                                'type': 'price_table_entry',
                                'metadata': {
                                    'service_category': service_title.lower(),
                                    'short_code': short_code,
                                    'description': description,
                                    'price': price,
                                    'section_index': section_idx,
                                    'row_index': row_idx,
                                    'columns_count': len(row_data)
                                }
                            })
                            
                        # Also create service-specific summary
                        if row_idx == 0:  # First row of each service
                            service_summary = f"{service_title} Services: Available extra services for {service_title.lower()} flights including {description}"
                            
                            price_data.append({
                                'airline': airline_id,
                                'source': page_name,
                                'content': service_summary,
                                'url': url,
                                'type': 'service_category_summary',
                                'metadata': {
                                    'service_category': service_title.lower(),
                                    'is_summary': True,
                                    'section_index': section_idx
                                }
                            })
                
                except Exception as e:
                    logger.warning(f"    ⚠️ Row {row_idx} in {service_title} parsing error: {e}")
                    continue
        
        logger.info(f"    ✅ Extracted {len(price_data)} price table entries")
        return price_data
    
    def _extract_service_sections(self, container, selectors: dict) -> Dict[str, List[List[str]]]:
        """Extract service sections (Domestic, International, KKTC) with their data"""
        
        service_sections = {}
        
        # Method 1: Try to find title + list pairs
        service_titles = container.select(selectors['service_titles'])
        
        for title_element in service_titles:
            # Get service title
            title_text = title_element.get_text(strip=True)
            title_text = title_text.replace("Target here", "").strip()
            
            if not title_text or len(title_text) < 3:
                continue
            
            # Find corresponding table/list after this title
            service_list = title_element.find_next_sibling() or title_element.parent.find_next_sibling()
            
            if service_list:
                service_rows = service_list.select(selectors['table_rows'])
                
                table_data = []
                for row in service_rows:
                    row_cells = row.select(selectors['table_cells'])
                    cell_data = []
                    
                    for cell in row_cells:
                        # Try to get cell title first, then text content
                        cell_title_elem = cell.select_one(selectors.get('cell_titles', ''))
                        if cell_title_elem:
                            cell_text = cell_title_elem.get_text(strip=True)
                        else:
                            cell_text = cell.get_text(strip=True)
                        
                        cell_text = cell_text.replace("Target here", "").strip()
                        cell_text = ' '.join(cell_text.split())  # Clean spaces
                        
                        if cell_text:
                            cell_data.append(cell_text)
                    
                    if cell_data:
                        table_data.append(cell_data)
                
                if table_data:
                    service_sections[title_text] = table_data
                    logger.debug(f"      📋 {title_text}: {len(table_data)} rows")
        
        # Method 2: Fallback - try to extract from any table structure
        if not service_sections:
            logger.info(f"    🔄 Using fallback table extraction...")
            
            all_rows = container.select(selectors['table_rows'])
            if all_rows:
                fallback_data = []
                
                for row in all_rows:
                    cells = row.select(selectors['table_cells'])
                    row_data = []
                    
                    for cell in cells:
                        text = cell.get_text(strip=True)
                        text = text.replace("Target here", "").strip()
                        if text:
                            row_data.append(text)
                    
                    if row_data:
                        fallback_data.append(row_data)
                
                if fallback_data:
                    service_sections["General Services"] = fallback_data
        
        return service_sections
    
    def _parse_accordion_content(self, soup: BeautifulSoup, airline_id: str, 
                                page_name: str, url: str, selectors: dict, filters: dict) -> List[Dict]:
        """Parse Pegasus accordion/FAQ content - Enhanced with multiple style support"""
        
        accordion_data = []
        
        # Try to find FAQ items with the configured selectors
        faq_selector = selectors['faq_items']
        faq_items = soup.select(faq_selector)
        
        # If no items found with primary selector, try alternative selectors
        if not faq_items:
            alternative_selectors = [
                '.n-faq-acc__item',  # Style 1
                '.faq-acc__item',    # Style 2
                '.faq-item',         # Generic
                '.accordion-item'    # Generic
            ]
            
            for alt_selector in alternative_selectors:
                if alt_selector != faq_selector:  # Don't retry the same selector
                    faq_items = soup.select(alt_selector)
                    if faq_items:
                        logger.info(f"    🔄 Found items with alternative selector: {alt_selector}")
                        # Update selectors for this page
                        selectors = self._adapt_selectors_to_alternative(selectors, alt_selector)
                        break
        
        if not faq_items:
            logger.warning(f"    ❌ No FAQ items found with any selector")
            return []
        
        logger.info(f"    🎯 Found {len(faq_items)} accordion items")
        
        for item_idx, faq_item in enumerate(faq_items):
            try:
                # Get header (SORU) - try multiple selectors
                question = self._extract_question(faq_item, selectors)
                
                # Get content (CEVAP) - try multiple selectors  
                answer_parts = self._extract_answer_parts(faq_item, selectors, filters)
                
                if question and answer_parts:
                    full_answer = " ".join(answer_parts)
                    
                    # Create Q&A format
                    qa_content = f"Q: {question}\nA: {full_answer}"
                    
                    # Validate content quality
                    if self._is_valid_content(qa_content, filters):
                        accordion_data.append({
                            'airline': airline_id,
                            'source': page_name,
                            'content': qa_content,
                            'url': url,
                            'type': 'qa_pair',
                            'metadata': {
                                'accordion_index': item_idx,
                                'question': question,
                                'answer': full_answer,
                                'answer_parts_count': len(answer_parts)
                            }
                        })
                        
                        logger.debug(f"    📝 Q&A pair {item_idx}: {question[:50]}...")
                    
                    # Create detailed entries for complex answers
                    if len(answer_parts) > 3:
                        for part_idx, answer_part in enumerate(answer_parts):
                            if len(answer_part) >= 30:
                                detailed_content = f"Regarding '{question}': {answer_part}"
                                
                                accordion_data.append({
                                    'airline': airline_id,
                                    'source': page_name,
                                    'content': detailed_content,
                                    'url': url,
                                    'type': 'qa_detail',
                                    'metadata': {
                                        'accordion_index': item_idx,
                                        'part_index': part_idx,
                                        'parent_question': question,
                                        'detail_type': self._classify_content_type(answer_part)
                                    }
                                })
                
                # Handle question-only cases
                elif question and len(question) >= 20:
                    accordion_data.append({
                        'airline': airline_id,
                        'source': page_name,
                        'content': f"Question: {question}",
                        'url': url,
                        'type': 'question_only',
                        'metadata': {
                            'accordion_index': item_idx,
                            'question': question,
                            'has_answer': False
                        }
                    })
                    
            except Exception as e:
                logger.warning(f"    ⚠️ Accordion item {item_idx} parsing error: {e}")
                continue
        
        logger.info(f"    ✅ Extracted {len(accordion_data)} Q&A items from accordions")
        return accordion_data
    
    def _adapt_selectors_to_alternative(self, original_selectors: dict, alt_item_selector: str) -> dict:
        """Adapt selectors when alternative accordion structure is found"""
        
        new_selectors = original_selectors.copy()
        
        if alt_item_selector == '.n-faq-acc__item':
            new_selectors.update({
                'faq_items': '.n-faq-acc__item',
                'faq_headers': '.n-faq-acc__header',
                'faq_content': '.n-faq-acc__content'
            })
        elif alt_item_selector == '.faq-acc__item':
            new_selectors.update({
                'faq_items': '.faq-acc__item',
                'faq_headers': '.faq-acc__header',
                'faq_content': '.faq-acc__content'
            })
        
        return new_selectors
    
    def _extract_question(self, faq_item, selectors: dict) -> str:
        """Extract question text with multiple selector fallbacks"""
        
        header_selectors = [
            selectors.get('faq_headers'),
            '.n-faq-acc__header',
            '.faq-acc__header', 
            'h3',
            '.header',
            '.title'
        ]
        
        for selector in header_selectors:
            if selector:
                header_element = faq_item.select_one(selector)
                if header_element:
                    question = header_element.get_text(strip=True)
                    if question:
                        # Clean question text
                        question = question.replace("Target here.", "").strip()
                        question = ' '.join(question.split())  # Remove extra spaces
                        if len(question) >= 5:
                            return question
        
        return ""
    
    def _extract_answer_parts(self, faq_item, selectors: dict, filters: dict) -> List[str]:
        """Extract answer parts with multiple selector fallbacks"""
        
        content_selectors = [
            selectors.get('faq_content'),
            '.n-faq-acc__content',
            '.faq-acc__content',
            '.content',
            '.answer'
        ]
        
        answer_parts = []
        
        for selector in content_selectors:
            if selector:
                content_element = faq_item.select_one(selector)
                if content_element:
                    # Extract content from various elements
                    elements_to_check = [
                        ('.cms-content-default p', ''),
                        ('.cms-content-default li', '• '),
                        ('.cms-content-default h1, .cms-content-default h2, .cms-content-default h3', '## '),
                        ('p', ''),
                        ('li', '• '),
                        ('div', '')
                    ]
                    
                    for element_selector, prefix in elements_to_check:
                        elements = content_element.select(element_selector)
                        for elem in elements:
                            text = elem.get_text(strip=True)
                            text = text.replace("Target here.", "").strip()
                            text = ' '.join(text.split())
                            
                            if text and len(text) >= 10:
                                answer_parts.append(f"{prefix}{text}")
                    
                    if answer_parts:
                        break
        
        return answer_parts
    
    def _classify_content_type(self, content: str) -> str:
        """Classify content type for metadata"""
        if content.startswith('• '):
            return 'list_item'
        elif content.startswith('## '):
            return 'sub_header'
        else:
            return 'paragraph'
    
    def _is_valid_content(self, text: str, filters: dict) -> bool:
        """Content validation based on filters"""
        
        if not text:
            return False
        
        # Length check
        min_length = filters.get('min_text_length', 10)
        if len(text) < min_length:
            return False
        
        # Keyword exclusion
        exclude_keywords = filters.get('exclude_keywords', [])
        text_lower = text.lower()
        
        for keyword in exclude_keywords:
            if keyword in text_lower:
                return False
        
        return True
    
    def _extract_table_data(self, table, airline_id: str, page_name: str, 
                           url: str, table_idx: int) -> List[Dict]:
        """Extract table data with context"""
        
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
                    'airline': airline_id,
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
    
    def calculate_quality_score(self, item: dict) -> float:
        """Enhanced quality score calculation"""
        content = item['content']
        item_type = item.get('type', 'unknown')
        score = len(content) * 0.01  # Base length bonus
        
        # Q&A specific bonuses
        if item_type == 'qa_pair':
            score += 1.0  # High bonus for complete Q&A pairs
            if 'Q:' in content and 'A:' in content:
                score += 0.5  # Well-formatted Q&A
        
        if item_type == 'qa_detail':
            score += 0.7  # Good bonus for detailed answers
        
        # Price table specific bonuses
        if item_type == 'price_table_entry':
            score += 0.9  # High value for structured price data
            if 'Code:' in content and 'Price:' in content:
                score += 0.6  # Complete price entry
        
        if item_type == 'service_category_summary':
            score += 0.8  # Good value for service summaries
        
        # CMS content specific bonuses
        if item_type == 'cms_section':
            score += 0.8  # High value for structured sections
            if 'Section:' in content:
                score += 0.4  # Well-formatted section
        
        if item_type == 'cms_header':
            score += 0.6  # Good value for topic headers
        
        # Structured content bonuses
        if '|' in content: score += 0.5      # Structured content
        if ':' in content: score += 0.3      # Key-value pairs
        if re.search(r'\d+\s*(kg|cm|ml|liter|%|TL|€|\$|USD|EUR)', content.lower()): score += 0.6  # Numeric/price info
        if '✓' in content or 'X' in content: score += 0.25  # Table symbols
        if item.get('type') == 'table_row': score += 0.6  # Table content
        if content.strip().endswith(('.', '!', '?')): score += 0.1  # Complete sentences
        
        # Language quality indicators
        if re.search(r'\b(allowed|prohibited|maximum|minimum|limit|fee|cost|price|service|section)\b', content.lower()): 
            score += 0.4  # Policy-relevant keywords
        
        # Quality penalties
        if len(content) < 15: score *= 0.5  # Very short content penalty
        if 'target here' in content.lower(): score *= 0.1  # Placeholder content penalty
        
        # Metadata-based bonuses
        metadata = item.get('metadata', {})
        if metadata.get('price') and metadata['price'] != 'Price not specified':
            score += 0.3  # Has actual price info
        if metadata.get('short_code'):
            score += 0.2  # Has service code
        if metadata.get('section_title'):
            score += 0.3  # Has section context
        if metadata.get('parent_section'):
            score += 0.2  # Has parent context
        
        return max(score, 0.1)
    
    def remove_duplicates(self, data_list: List[Dict]) -> List[Dict]:
        """Enhanced duplicate removal - airline aware"""
        
        logger.info(f"\n🧹 MULTI-AIRLINE DUPLICATE REMOVAL")
        logger.info("=" * 50)
        
        if not data_list:
            return []
        
        # Group by airline + content hash
        hash_groups = {}
        
        for item in data_list:
            content_normalized = item['content'].lower().strip()
            content_hash = hashlib.md5(content_normalized.encode()).hexdigest()
            
            # Quality score hesapla
            item['quality_score'] = self.calculate_quality_score(item)
            item['content_hash'] = content_hash
            
            # Key: airline + content_hash
            key = f"{item['airline']}:{content_hash}"
            
            if key not in hash_groups:
                hash_groups[key] = []
            
            hash_groups[key].append(item)
        
        # Her grup için en iyisini seç
        unique_data = []
        total_duplicates = 0
        
        for key, items in hash_groups.items():
            if len(items) == 1:
                unique_data.append(items[0])
            else:
                # En yüksek quality score'lu olanı seç
                best_item = max(items, key=lambda x: x['quality_score'])
                unique_data.append(best_item)
                
                duplicates_count = len(items) - 1
                total_duplicates += duplicates_count
                
                logger.info(f"🔄 {best_item['airline']} - {len(items)} duplicate:")
                logger.info(f"  ✅ Korunan: {best_item['content'][:70]}... (Score: {best_item['quality_score']:.2f})")
        
        logger.info(f"\n✅ Multi-Airline Duplicate Removal:")
        logger.info(f"  📊 {len(data_list)} → {len(unique_data)} item")
        logger.info(f"  🗑️ {total_duplicates} duplicate kaldırıldı")
        
        return unique_data
    
    def save_policies_to_database(self, policies: List[Dict]) -> int:
        """Save policies to database"""
        
        conn = self.get_db_connection()
        if not conn:
            return 0
        
        try:
            cursor = conn.cursor()
            saved_count = 0
            
            insert_sql = """
                INSERT INTO policy 
                (airline, source, content, content_hash, quality_score, extraction_type, url, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (airline, source, content_hash) 
                DO UPDATE SET 
                    updated_at = CURRENT_TIMESTAMP,
                    quality_score = EXCLUDED.quality_score
                RETURNING id;
            """
            
            for policy in policies:
                try:
                    cursor.execute(insert_sql, (
                        policy['airline'],
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
                    logger.error(f"  ⚠️ Policy kaydetme hatası: {e}")
                    conn.rollback()
                    break
            
            conn.commit()
            cursor.close()
            
            logger.info(f"✅ {saved_count} policy PostgreSQL'e kaydedildi")
            return saved_count
            
        except Exception as e:
            logger.error(f"❌ PostgreSQL kaydetme hatası: {e}")
            conn.rollback()
            return 0
    
    def scrape_airline(self, airline_id: str) -> int:
        """Scrape all pages for specific airline"""
        
        config = get_airline_config(airline_id)
        if not config:
            logger.error(f"❌ Airline config bulunamadı: {airline_id}")
            return 0
        
        logger.info(f"\n🚀 {config['airline_name']} scraping başlatılıyor...")
        logger.info("=" * 60)
        
        all_data = []
        success_count = 0
        error_count = 0
        
        # Scrape all pages
        for page_name, page_config in config['pages'].items():
            logger.info(f"📡 {page_name.replace('_', ' ').title()} çekiliyor...")
            
            page_data = self.scrape_page(airline_id, page_name, page_config)
            
            if page_data:
                all_data.extend(page_data)
                success_count += 1
                logger.info(f"  ✅ {len(page_data)} ham veri alındı")
            else:
                error_count += 1
                logger.info(f"  ❌ Veri alınamadı")
        
        logger.info(f"\n📊 {config['airline_name']} HAM VERİ RAPORU:")
        logger.info(f"  ✅ Başarılı: {success_count}/{len(config['pages'])} sayfa")
        logger.info(f"  ❌ Hatalı: {error_count}/{len(config['pages'])} sayfa")
        logger.info(f"  📋 Toplam ham veri: {len(all_data)} item")
        
        if not all_data:
            logger.warning(f"⚠️ {airline_id} için veri alınamadı")
            return 0
        
        # Remove duplicates
        clean_data = self.remove_duplicates(all_data)
        
        # Save to database
        if clean_data:
            saved_count = self.save_policies_to_database(clean_data)
            logger.info(f"🎉 {config['airline_name']}: {saved_count} policy kaydedildi!")
            return saved_count
        else:
            logger.error(f"❌ {airline_id} için temiz veri bulunamadı")
            return 0
    
    def scrape_all_airlines(self) -> Dict[str, int]:
        """Scrape all configured airlines"""
        
        airlines = get_all_airlines()
        results = {}
        
        logger.info(f"\n🌍 HYBRID AIRLINE SCRAPING BAŞLANIYOR")
        logger.info(f"📋 Airline sayısı: {len(airlines)}")
        logger.info("=" * 70)
        
        for airline_id in airlines:
            try:
                count = self.scrape_airline(airline_id)
                results[airline_id] = count
                
                if count > 0:
                    logger.info(f"✅ {airline_id}: {count} policy başarılı")
                else:
                    logger.warning(f"⚠️ {airline_id}: Veri alınamadı")
                    
            except Exception as e:
                logger.error(f"❌ {airline_id} scraping hatası: {e}")
                results[airline_id] = 0
        
        # Final summary
        total_scraped = sum(results.values())
        successful_airlines = len([k for k, v in results.items() if v > 0])
        
        logger.info(f"\n🎯 HYBRID SCRAPING RAPORU:")
        logger.info(f"  📊 Toplam policy: {total_scraped}")
        logger.info(f"  ✅ Başarılı airline: {successful_airlines}/{len(airlines)}")
        logger.info(f"  📋 Detay: {results}")
        
        return results
    
    def get_database_stats(self) -> Dict:
        """Get enhanced database statistics"""
        
        conn = self.get_db_connection()
        if not conn:
            return {}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Overall stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_policies,
                    COUNT(DISTINCT airline) as total_airlines,
                    COUNT(DISTINCT source) as total_sources,
                    AVG(quality_score) as avg_quality_score
                FROM policy
            """)
            
            stats = dict(cursor.fetchone())
            
            # Airline breakdown
            cursor.execute("""
                SELECT 
                    airline,
                    COUNT(*) as count, 
                    AVG(quality_score) as avg_quality,
                    COUNT(DISTINCT source) as sources_count
                FROM policy 
                GROUP BY airline 
                ORDER BY count DESC
            """)
            
            airline_breakdown = {}
            for row in cursor.fetchall():
                airline_breakdown[row['airline']] = {
                    'count': row['count'],
                    'avg_quality': round(float(row['avg_quality']), 2),
                    'sources_count': row['sources_count']
                }
            
            stats['airline_breakdown'] = airline_breakdown
            
            cursor.close()
            return stats
            
        except Exception as e:
            logger.error(f"❌ Stats alma hatası: {e}")
            return {}
    
    def clear_airline_data(self, airline_id: str) -> bool:
        """Clear data for specific airline"""
        
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM policy WHERE airline = %s", (airline_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            
            logger.info(f"🗑️ {airline_id}: {deleted_count} policy silindi")
            return True
            
        except Exception as e:
            logger.error(f"❌ {airline_id} veri silme hatası: {e}")
            return False
    
    def clear_all_data(self) -> bool:
        """Clear all airline data"""
        
        conn = self.get_db_connection()
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM policy")
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            
            logger.info(f"🗑️ Tüm data temizlendi: {deleted_count} policy")
            return True
            
        except Exception as e:
            logger.error(f"❌ Tüm veri silme hatası: {e}")
            return False

# Usage helper functions
def scrape_specific_airline(airline_id: str) -> int:
    """Scrape only specific airline"""
    scraper = MultiAirlineScraper()
    return scraper.scrape_airline(airline_id)

def scrape_all_airlines() -> Dict[str, int]:
    """Scrape all airlines"""
    scraper = MultiAirlineScraper()
    return scraper.scrape_all_airlines()