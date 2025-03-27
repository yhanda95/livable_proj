import praw
import pandas as pd
import os
import time
import concurrent.futures
import logging
import requests
import feedparser
import newspaper
from newspaper import Article
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("data_collection.log"),
                              logging.StreamHandler()])

# Reddit API Credentials
REDDIT_CLIENT_ID = 'C2o7qWzp2-dQnnH2Wyducw'
REDDIT_CLIENT_SECRET = 'Pp6_MAL7EpBZ45EgSAzUezxni0JEMA'
USER_AGENT = 'LivabilityDashboard/1.0 (by u/your_username)'


# File paths
WORLD_DATA_PATH = r"C:\Users\NAVYA\OneDrive\Desktop\projects\mini project\data\world_data_with_scores (1).csv"
OUTPUT_DIR = "collected_data"
CACHE_DIR = "cache"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# News sources and forums configuration
NEWS_SOURCES = {
    # Format: 'source_name': {'url': 'base_url', 'comment_selector': 'css_selector_for_comments'}
    'guardian': {'url': 'https://www.theguardian.com', 'comment_selector': '.d-comment__content'},
    'reuters': {'url': 'https://www.reuters.com', 'comment_selector': '.comment-content'},
    'bbc': {'url': 'https://www.bbc.com', 'comment_selector': '.comments__comment'},
    # Add more news sources as needed
}

# RSS Feed sources by category
RSS_FEEDS = {
    'general': [
        'http://feeds.bbci.co.uk/news/world/rss.xml',
        'http://rss.cnn.com/rss/cnn_world.rss',
        'https://www.aljazeera.com/xml/rss/all.xml'
    ],
    'health': [
        'https://www.who.int/rss-feeds/news-english.xml',
        'https://rss.medicalnewstoday.com/newsfeeds/Medical.xml'
    ],
    'economy': [
        'https://www.economist.com/finance-and-economics/rss.xml',
        'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664'
    ],
    'environment': [
        'https://www.ecowatch.com/feeds/latest.rss',
        'https://www.wired.com/feed/category/science/latest/rss'
    ],
    'social': [
        'https://feeds.feedburner.com/TEDTalks_video',
        'https://www.hrw.org/taxonomy/term/10175/feed'
    ]
}

# Additional public forums by category
PUBLIC_FORUMS = {
    'health': ['patient.info', 'healthunlocked.com', 'medhelp.org'],
    'economy': ['tradingview.com/ideas', 'bogleheads.org/forum'],
    'environment': ['environmentalscience.org/forum', 'myforum.laserfiche.com/environment'],
    'social': ['sociologyforums.com', 'city-data.com/forum']
}

class EnhancedDataCollector:
    def __init__(self):
        self.reddit = None
        self.world_data = None
        
    def initialize_reddit(self):
        """Initialize Reddit API client"""
        if not self.reddit:
            self.reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                                     client_secret=REDDIT_CLIENT_SECRET,
                                     user_agent=USER_AGENT)
        return self.reddit
    
    def load_world_data(self):
        """Load country list from world data"""
        try:
            self.world_data = pd.read_csv(WORLD_DATA_PATH)
            all_countries = self.world_data["Country"].unique().tolist()
            logging.info(f"Loaded {len(all_countries)} countries from world data")
            return all_countries
        except Exception as e:
            logging.error(f"Failed to load world data: {e}")
            return []
    
    def fetch_reddit_data(self, country, max_posts=30):
        """Fetch Reddit data for a given country"""
        country_cache_file = os.path.join(CACHE_DIR, f"reddit_{country.replace(' ', '_')}.csv")
        
        # Check cache
        if os.path.exists(country_cache_file) and os.path.getmtime(country_cache_file) > time.time() - 86400:
            try:
                return pd.read_csv(country_cache_file)
            except Exception as e:
                logging.warning(f"Could not read Reddit cache for {country}: {e}")
        
        # Initialize Reddit if not already done
        reddit = self.initialize_reddit()
        
        data = []
        retries = 0
        max_retries = 3
        
        while retries < max_retries:
            try:
                # Optimize query to target key policy areas
                queries = [
                    f'"{country}" (healthcare OR health OR hospital OR medicine OR doctor)',
                    f'"{country}" (economy OR economic OR jobs OR unemployment OR inflation OR cost of living)',
                    f'"{country}" (environment OR climate OR pollution OR sustainable OR green)',
                    f'"{country}" (education OR schools OR university OR learning)',
                    f'"{country}" (crime OR safety OR police OR security)',
                    f'"{country}" (infrastructure OR transport OR roads OR housing)',
                    f'"{country}" (government OR policy OR politics OR democracy)'
                ]
                
                for query in queries:
                    for post in reddit.subreddit("all").search(query, limit=max_posts//len(queries), sort="relevance"):
                        # Skip posts with no content
                        if not post.selftext and len(post.title) < 20:
                            continue
                            
                        # Get top level comments for better analysis
                        post.comments.replace_more(limit=0)
                        comments = []
                        for comment in post.comments[:5]:  # Get top 5 comments
                            if hasattr(comment, 'body') and len(comment.body) > 20:
                                comments.append(comment.body[:300])  # Limit comment length
                        
                        # Combine title, preview of selftext, and top comments
                        text = post.title
                        if post.selftext:
                            text += " " + post.selftext[:500]  # Limit text length
                        
                        comment_text = " ".join(comments)
                        
                        category = self._determine_category(query)
                        
                        data.append({
                            'country': country,
                            'source': 'reddit',
                            'post_id': post.id,
                            'title': post.title,
                            'text': text,
                            'comments': comment_text,
                            'category': category,
                            'sentiment_score': None,  # Will be filled later
                            'upvotes': post.score,
                            'created_utc': post.created_utc,
                            'url': f"https://www.reddit.com{post.permalink}"
                        })
                
                # Cache the results for this country
                country_df = pd.DataFrame(data)
                if not country_df.empty:
                    country_df.to_csv(country_cache_file, index=False)
                
                return country_df
                
            except Exception as e:
                retries += 1
                logging.warning(f"Error fetching Reddit data for {country}, retry {retries}/{max_retries}: {e}")
                time.sleep(2)  # Wait before retrying
        
        logging.error(f"Failed to fetch Reddit data for {country} after {max_retries} retries")
        return pd.DataFrame()  # Return empty DataFrame on failure
    
    def fetch_news_comments(self, country, max_articles=10):
        """Fetch comments from news articles related to a country"""
        country_cache_file = os.path.join(CACHE_DIR, f"news_{country.replace(' ', '_')}.csv")
        
        # Check cache
        if os.path.exists(country_cache_file) and os.path.getmtime(country_cache_file) > time.time() - 86400:
            try:
                return pd.read_csv(country_cache_file)
            except Exception as e:
                logging.warning(f"Could not read news cache for {country}: {e}")
        
        data = []
        
        for source_name, source_info in NEWS_SOURCES.items():
            try:
                # Use Google News API or search directly for articles
                search_url = f"https://news.google.com/rss/search?q={country}+when:7d&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(search_url)
                
                articles_processed = 0
                for entry in feed.entries[:max_articles]:
                    if articles_processed >= max_articles:
                        break
                    
                    try:
                        # Extract article using newspaper3k
                        article = Article(entry.link)
                        article.download()
                        article.parse()
                        
                        # If we can access the original source, try to get comments
                        comments = []
                        try:
                            response = requests.get(article.url, headers={'User-Agent': USER_AGENT}, timeout=10)
                            if response.status_code == 200:
                                soup = BeautifulSoup(response.text, 'html.parser')
                                # Look for comments section
                                comment_elements = soup.select(source_info['comment_selector'])
                                for comment in comment_elements[:10]:  # Get up to 10 comments
                                    comments.append(comment.get_text(strip=True)[:300])  # Limit length
                        except Exception as e:
                            logging.debug(f"Could not extract comments from {article.url}: {e}")
                        
                        # Determine category from article content
                        category = self._categorize_text(article.title + " " + article.text[:500])
                        
                        data.append({
                            'country': country,
                            'source': f"news_{source_name}",
                            'post_id': entry.id,
                            'title': entry.title,
                            'text': article.text[:1000],  # Limit text length
                            'comments': " ".join(comments),
                            'category': category,
                            'sentiment_score': None,  # Will be filled later
                            'upvotes': None,
                            'created_utc': time.mktime(entry.published_parsed) if hasattr(entry, 'published_parsed') else time.time(),
                            'url': article.url
                        })
                        
                        articles_processed += 1
                        
                    except Exception as e:
                        logging.debug(f"Error processing article {entry.link}: {e}")
            
            except Exception as e:
                logging.warning(f"Error fetching news data from {source_name} for {country}: {e}")
        
        # Cache the results for this country
        country_df = pd.DataFrame(data)
        if not country_df.empty:
            country_df.to_csv(country_cache_file, index=False)
        
        return country_df
    
    def fetch_rss_feeds(self, country):
        """Fetch data from RSS feeds related to a country"""
        country_cache_file = os.path.join(CACHE_DIR, f"rss_{country.replace(' ', '_')}.csv")
        
        # Check cache
        if os.path.exists(country_cache_file) and os.path.getmtime(country_cache_file) > time.time() - 86400:
            try:
                return pd.read_csv(country_cache_file)
            except Exception as e:
                logging.warning(f"Could not read RSS cache for {country}: {e}")
        
        data = []
        
        for category, feeds in RSS_FEEDS.items():
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries:
                        # Check if entry is related to this country
                        if country.lower() in (entry.title + entry.description).lower():
                            # Extract article text if possible
                            article_text = entry.description
                            try:
                                article = Article(entry.link)
                                article.download()
                                article.parse()
                                article_text = article.text[:1000]  # Limit text length
                            except:
                                pass  # Use entry description if article extraction fails
                            
                            data.append({
                                'country': country,
                                'source': f"rss_{feed_url.split('/')[2]}",
                                'post_id': entry.id if hasattr(entry, 'id') else entry.link,
                                'title': entry.title,
                                'text': article_text,
                                'comments': "",  # RSS usually doesn't have comments
                                'category': category,
                                'sentiment_score': None,  # Will be filled later
                                'upvotes': None,
                                'created_utc': time.mktime(entry.published_parsed) if hasattr(entry, 'published_parsed') else time.time(),
                                'url': entry.link
                            })
                
                except Exception as e:
                    logging.warning(f"Error fetching RSS feed {feed_url} for {country}: {e}")
        
        # Cache the results for this country
        country_df = pd.DataFrame(data)
        if not country_df.empty:
            country_df.to_csv(country_cache_file, index=False)
        
        return country_df
    
    def fetch_public_forums(self, country):
        """Fetch data from public forums related to a country"""
        country_cache_file = os.path.join(CACHE_DIR, f"forum_{country.replace(' ', '_')}.csv")
        
        # Check cache
        if os.path.exists(country_cache_file) and os.path.getmtime(country_cache_file) > time.time() - 86400:
            try:
                return pd.read_csv(country_cache_file)
            except Exception as e:
                logging.warning(f"Could not read forum cache for {country}: {e}")
        
        data = []
        
        for category, forums in PUBLIC_FORUMS.items():
            for forum in forums:
                try:
                    # Search for threads related to the country
                    search_url = f"https://www.google.com/search?q=site:{forum}+{country}&tbm=dsc"
                    
                    response = requests.get(search_url, headers={'User-Agent': USER_AGENT}, timeout=10)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extract forum posts from search results
                        for result in soup.select('.g')[:5]:  # Limit to first 5 results
                            try:
                                title_elem = result.select_one('h3')
                                link_elem = result.select_one('a')
                                snippet_elem = result.select_one('.st')
                                
                                if title_elem and link_elem and snippet_elem:
                                    title = title_elem.get_text(strip=True)
                                    link = link_elem['href']
                                    snippet = snippet_elem.get_text(strip=True)
                                    
                                    data.append({
                                        'country': country,
                                        'source': f"forum_{forum}",
                                        'post_id': link,
                                        'title': title,
                                        'text': snippet,
                                        'comments': "",  # Would need to follow the link to get comments
                                        'category': category,
                                        'sentiment_score': None,  # Will be filled later
                                        'upvotes': None,
                                        'created_utc': time.time(),  # Use current time as proxy
                                        'url': link
                                    })
                            except Exception as e:
                                logging.debug(f"Error parsing forum result: {e}")
                
                except Exception as e:
                    logging.warning(f"Error fetching forum data from {forum} for {country}: {e}")
        
        # Cache the results for this country
        country_df = pd.DataFrame(data)
        if not country_df.empty:
            country_df.to_csv(country_cache_file, index=False)
        
        return country_df
    
    def fetch_all_sources(self, country):
        """Fetch data from all sources for a given country"""
        reddit_data = self.fetch_reddit_data(country)
        news_data = self.fetch_news_comments(country)
        rss_data = self.fetch_rss_feeds(country)
        forum_data = self.fetch_public_forums(country)
        
        # Combine all data sources
        all_data = pd.concat([reddit_data, news_data, rss_data, forum_data], ignore_index=True)
        
        return all_data
    
    def collect_all_countries(self, max_workers=4):
        """Collect data for all countries with parallel processing"""
        start_time = time.time()
        
        # Load country list
        all_countries = self.load_world_data()
        if not all_countries:
            return
        
        # Process countries in parallel
        all_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Create a dictionary of future to country for tracking
            future_to_country = {
                executor.submit(self.fetch_all_sources, country): country 
                for country in all_countries
            }
            
            # Process completed futures with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_country), 
                              total=len(future_to_country),
                              desc="Fetching country data from all sources"):
                country = future_to_country[future]
                try:
                    country_df = future.result()
                    if not country_df.empty:
                        all_data.append(country_df)
                        logging.info(f"Fetched {len(country_df)} posts for {country} from all sources")
                except Exception as e:
                    logging.error(f"Error processing results for {country}: {e}")
        
        if all_data:
            # Combine all data
            df = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV with timestamp
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = os.path.join(OUTPUT_DIR, f"all_sources_data_{timestamp}.csv")
            df.to_csv(output_path, index=False)
            
            # Also save by category
            for category in df['category'].unique():
                if pd.notna(category):
                    category_df = df[df['category'] == category]
                    category_path = os.path.join(OUTPUT_DIR, f"{category.lower()}_data_{timestamp}.csv")
                    category_df.to_csv(category_path, index=False)
            
            logging.info(f"✅ All data saved successfully! Total records: {len(df)}")
            logging.info(f"Total time: {time.time() - start_time:.2f} seconds")
            
            return df
        else:
            logging.warning("No data collected!")
            return pd.DataFrame()
    
    def _determine_category(self, query):
        """Determine category based on search query"""
        query = query.lower()
        if any(term in query for term in ['healthcare', 'health', 'hospital', 'medicine', 'doctor']):
            return 'Health'
        elif any(term in query for term in ['economy', 'economic', 'jobs', 'unemployment', 'inflation']):
            return 'Economy'
        elif any(term in query for term in ['environment', 'climate', 'pollution', 'sustainable']):
            return 'Environment'
        elif any(term in query for term in ['education', 'schools', 'university']):
            return 'Education'
        elif any(term in query for term in ['crime', 'safety', 'police', 'security']):
            return 'Safety'
        elif any(term in query for term in ['infrastructure', 'transport', 'roads', 'housing']):
            return 'Infrastructure'
        elif any(term in query for term in ['government', 'policy', 'politics']):
            return 'Government'
        return 'Social'  # Default category
    
    def _categorize_text(self, text):
        """Simple rule-based categorization of text"""
        text = text.lower()
        categories = {
            'Health': ['health', 'healthcare', 'hospital', 'medical', 'doctor', 'disease', 'pandemic', 'virus', 'medicine'],
            'Economy': ['economy', 'economic', 'job', 'unemployment', 'inflation', 'finance', 'business', 'market', 'trade', 'tax'],
            'Environment': ['environment', 'climate', 'pollution', 'sustainable', 'renewable', 'emission', 'green', 'energy', 'waste'],
            'Education': ['education', 'school', 'university', 'college', 'student', 'teacher', 'learning', 'academic'],
            'Safety': ['crime', 'safety', 'police', 'security', 'violence', 'law', 'justice', 'prison'],
            'Infrastructure': ['infrastructure', 'transport', 'road', 'housing', 'construction', 'public transport', 'railway', 'airport'],
            'Government': ['government', 'policy', 'politics', 'election', 'democracy', 'minister', 'president', 'parliament'],
            'Social': ['social', 'community', 'welfare', 'equality', 'diversity', 'inclusion', 'culture', 'religion', 'family']
        }
        
        category_scores = {category: 0 for category in categories}
        
        for category, keywords in categories.items():
            for keyword in keywords:
                category_scores[category] += text.count(keyword)
        
        # Get the category with the highest score
        max_category = max(category_scores.items(), key=lambda x: x[1])
        
        # If no keywords found, default to Social
        if max_category[1] == 0:
            return 'Social'
        return max_category[0]

# Example usage
if __name__ == "__main__":
    collector = EnhancedDataCollector()
    all_data = collector.collect_all_countries(max_workers=4)
