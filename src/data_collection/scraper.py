"""
Web scraper for collecting news headlines from Fox News and NBC News.
Uses Beautiful Soup to extract headlines from provided URLs.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
from typing import List, Dict, Optional
import csv
from urllib.parse import urlparse
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsHeadlineScraper:
    """Scraper for collecting news headlines from Fox News and NBC News websites."""
    
    def __init__(self, delay: float = 1.0, timeout: int = 10):
        """
        Initialize the scraper.
        
        Args:
            delay: Delay between requests in seconds
            timeout: Request timeout in seconds
        """
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set user agent to avoid blocking
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def extract_headline(self, url: str) -> Optional[str]:
        """
        Extract headline from a single URL.
        
        Args:
            url: URL to scrape
            
        Returns:
            Extracted headline or None if failed
        """
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try different selectors for headlines
            headline_selectors = [
                'h1',  # Most common headline tag
                '.headline',
                '.title',
                'title',
                '[data-testid="headline"]',
                '.article-title',
                '.story-title'
            ]
            
            for selector in headline_selectors:
                element = soup.select_one(selector)
                if element:
                    headline = element.get_text().strip()
                    if headline:
                        return headline
                        
            logger.warning(f"No headline found for URL: {url}")
            return None
            
        except requests.RequestException as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            return None
    
    def determine_source(self, url: str) -> str:
        """
        Determine news source from URL.
        
        Args:
            url: URL to analyze
            
        Returns:
            'fox' or 'nbc'
        """
        domain = urlparse(url).netloc.lower()
        if 'fox' in domain:
            return 'fox'
        elif 'nbc' in domain:
            return 'nbc'
        else:
            logger.warning(f"Unknown domain: {domain}")
            return 'unknown'
    
    def scrape_from_csv(self, csv_file: str, output_file: str) -> pd.DataFrame:
        """
        Scrape headlines from URLs listed in a CSV file.
        
        Args:
            csv_file: Path to CSV file containing URLs
            output_file: Path to save scraped data
            
        Returns:
            DataFrame with scraped headlines
        """
        # Read URLs from CSV
        try:
            urls_df = pd.read_csv(csv_file)
            urls = urls_df['url'].tolist() if 'url' in urls_df.columns else urls_df.iloc[:, 0].tolist()
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return pd.DataFrame()
        
        headlines_data = []
        
        logger.info(f"Starting to scrape {len(urls)} URLs...")
        
        for i, url in enumerate(urls):
            if pd.isna(url):
                continue
                
            logger.info(f"Scraping {i+1}/{len(urls)}: {url}")
            
            headline = self.extract_headline(url)
            source = self.determine_source(url)
            
            if headline:
                headlines_data.append({
                    'url': url,
                    'headline': headline,
                    'source': source
                })
            
            # Respect rate limiting
            time.sleep(self.delay)
        
        # Create DataFrame and save
        df = pd.DataFrame(headlines_data)
        
        if not df.empty:
            df.to_csv(output_file, index=False)
            logger.info(f"Scraped {len(df)} headlines saved to {output_file}")
        else:
            logger.warning("No headlines were successfully scraped")
        
        return df
    
    def scrape_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Scrape headlines from a list of URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of dictionaries with headline data
        """
        headlines_data = []
        
        for url in urls:
            headline = self.extract_headline(url)
            source = self.determine_source(url)
            
            if headline:
                headlines_data.append({
                    'url': url,
                    'headline': headline,
                    'source': source
                })
            
            time.sleep(self.delay)
        
        return headlines_data


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape news headlines')
    parser.add_argument('--input', required=True, help='Input CSV file with URLs')
    parser.add_argument('--output', required=True, help='Output CSV file for headlines')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests')
    parser.add_argument('--timeout', type=int, default=10, help='Request timeout')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    scraper = NewsHeadlineScraper(delay=args.delay, timeout=args.timeout)
    df = scraper.scrape_from_csv(args.input, args.output)
    
    print(f"Scraping completed. {len(df)} headlines collected.")
    print(f"Fox News headlines: {len(df[df['source'] == 'fox'])}")
    print(f"NBC headlines: {len(df[df['source'] == 'nbc'])}")


if __name__ == "__main__":
    main()
