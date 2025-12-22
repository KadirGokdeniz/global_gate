# test_enhanced_scraping.py
import sys
from pathlib import Path

# Add 'src' directory to sys.path to resolve 'api' and 'scraper' packages
root_dir = Path(__file__).resolve().parent.parent # global_gate/
src_dir = root_dir / "src"
sys.path.append(str(src_dir))

from base_scraper import MultiAirlineScraper
from scraper.configs.airline_configs import get_airline_config

# Test without database - just extraction
scraper = MultiAirlineScraper()
scraper.connection = None  # Disable database

# Pegasus test
page_data = scraper.scrape_page(
    'pegasus', 
    'extra_services_pricing', 
    get_airline_config('pegasus')['pages']['extra_services_pricing']
)

print(f"Total extracted items: {len(page_data)}")
for item in page_data[:]:  # First 5 items
    print(f"- {item['content'][:]}...")