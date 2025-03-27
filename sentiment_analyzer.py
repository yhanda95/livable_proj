#[file name]: sentiment-analyzer.py
#[file content begin]
import pandas as pd
import numpy as np
import os
import time
import logging
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("sentiment_analysis.log"),
                              logging.StreamHandler()])

# Directories
DATA_DIR = "collected_data"
OUTPUT_DIR = "analysis_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

class SentimentAnalyzer:
    def __init__(self, use_gpu=False):
        self.device = 0 if use_gpu and torch.cuda.is_available() else -1
        self.models = {}
        self.nlp = None
        self.sia = None
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
    def initialize_models(self):
        """Initialize all NLP models"""
        start_time = time.time()
        logging.info("Initializing NLP models...")
        
        # Simple rule-based sentiment analyzer (fast)
        self.sia = SentimentIntensityAnalyzer()
        
        # Load spaCy for text processing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load Hugging Face models
        self.models['sentiment'] = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device
        )
        
        self.models['emotion'] = pipeline(
            "text-classification", 
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            device=self.device
        )
        
        self.models['topic'] = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )
        
        logging.info(f"Models initialized in {time.time() - start_time:.2f} seconds")
    
    def _preprocess_text(self, text):
        """Clean and preprocess text for analysis"""
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_text(self, text, full_analysis=False):
        """Analyze sentiment of a single text"""
        if not text or not isinstance(text, str) or len(text.strip()) < 10:
            return {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'emotion': None,
                'key_phrases': [],
                'entities': []
            }
        
        # Preprocess text
        clean_text = self._preprocess_text(text)
        if not clean_text:
            return {
                'sentiment_score': 0,
                'sentiment_label': 'neutral',
                'emotion': None,
                'key_phrases': [],
                'entities': []
            }
        
        # Quick VADER sentiment analysis
        vader_scores = self.sia.polarity_scores(clean_text)
        sentiment_score = vader_scores['compound']
        
        # Determine sentiment label
        if sentiment_score >= 0.05:
            sentiment_label = 'positive'
        elif sentiment_score <= -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        # Return minimal analysis if full analysis not requested
        if not full_analysis:
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'emotion': None,
                'key_phrases': [],
                'entities': []
            }
        
        # For full analysis, use more advanced models
        try:
            # Transformer-based sentiment (more nuanced)
            transformer_sentiment = self.models['sentiment'](clean_text[:512])[0]
            
            # Emotion analysis
            emotion_result = self.models['emotion'](clean_text[:512])[0]
            emotion = emotion_result['label']
            
            # Extract key phrases and entities
            doc = self.nlp(clean_text[:1000])  # Limit for performance
            
            # Get named entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_
                })
            
            # Extract key noun phrases
            key_phrases = []
            for chunk in doc.noun_chunks:
                # Filter out stopwords and short phrases
                phrase_text = ' '.join([token.text for token in chunk if token.text.lower() not in self.stopwords])
                if len(phrase_text) > 3:
                    key_phrases.append(phrase_text)
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'transformer_sentiment': transformer_sentiment,
                'emotion': emotion,
                'key_phrases': key_phrases[:5],  # Limit to top 5
                'entities': entities[:5]  # Limit to top 5
            }
        
        except Exception as e:
            logging.warning(f"Error in full text analysis: {e}")
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'emotion': None,
                'key_phrases': [],
                'entities': []
            }
    
    def analyze_dataframe(self, df, batch_size=32):
        """Analyze all texts in the dataframe"""
        if not self.sia:
            self.initialize_models()
        
        start_time = time.time()
        logging.info(f"Analyzing {len(df)} records...")
        
        # Create results columns
        df['sentiment_score'] = 0.0
        df['sentiment_label'] = 'neutral'
        df['emotion'] = None
        df['key_phrases'] = None
        df['entities'] = None
        
        # Process in batches for efficiency
        for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
            batch = df.iloc[i:i+batch_size].copy()
            
            # Combine title and text for analysis
            batch['combined_text'] = batch.apply(
                lambda row: f"{row.get('title', '')} {row.get('text', '')} {row.get('comments', '')}", 
                axis=1
            )
            
            # Analyze each text in the batch
            for idx, row in batch.iterrows():
                try:
                    # Determine if we should do full analysis (for a sample of records)
                    do_full_analysis = np.random.random() < 0.1  # 10% of records get full analysis
                    
                    # Analyze the text
                    results = self.analyze_text(row['combined_text'], full_analysis=do_full_analysis)
                    
                    # Update batch with results
                    batch.at[idx, 'sentiment_score'] = results['sentiment_score']
                    batch.at[idx, 'sentiment_label'] = results['sentiment_label']
                    
                    if do_full_analysis:
                        batch.at[idx, 'emotion'] = results['emotion']
                        batch.at[idx, 'key_phrases'] = results['key_phrases']
                        batch.at[idx, 'entities'] = results['entities']
                
                except Exception as e:
                    logging.error(f"Error analyzing row {idx}: {e}")
            
            # Update main dataframe with batch results
            df.update(batch)
        
        # Save analyzed data
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"analyzed_data_{timestamp}.csv")
        df.to_csv(output_path, index=False)
        logging.info(f"Saved analyzed data to {output_path}")
        
        # Generate visualizations
        self.generate_visualizations(df)
        
        logging.info(f"Analysis completed in {time.time() - start_time:.2f} seconds")
        return df
    
    def generate_visualizations(self, df):
        """Generate visualizations from analyzed data"""
        logging.info("Generating visualizations...")
        
        # Sentiment distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='sentiment_label', data=df)
        plt.title("Sentiment Distribution")
        plt.savefig(os.path.join(PLOTS_DIR, "sentiment_distribution.png"))
        plt.close()
        
        # Emotion distribution (for records with full analysis)
        if 'emotion' in df.columns:
            plt.figure(figsize=(12, 6))
            df['emotion'].value_counts().plot(kind='bar')
            plt.title("Emotion Distribution")
            plt.savefig(os.path.join(PLOTS_DIR, "emotion_distribution.png"))
            plt.close()
        
        # Sentiment by category
        if 'category' in df.columns:
            plt.figure(figsize=(14, 8))
            sns.countplot(x='category', hue='sentiment_label', data=df)
            plt.title("Sentiment Distribution by Category")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, "sentiment_by_category.png"))
            plt.close()
        
        # Word cloud for key phrases
        all_phrases = [phrase for sublist in df['key_phrases'].dropna() for phrase in sublist]
        if all_phrases:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_phrases))
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title("Most Frequent Key Phrases")
            plt.savefig(os.path.join(PLOTS_DIR, "key_phrases_wordcloud.png"))
            plt.close()
        
        logging.info("Visualizations saved to plots directory")

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer(use_gpu=False)
    
    # Load collected data (use most recent file)
    data_files = [f for f in os.listdir(DATA_DIR) if f.startswith("all_sources_data")]
    if data_files:
        latest_file = sorted(data_files)[-1]
        df = pd.read_csv(os.path.join(DATA_DIR, latest_file))
        
        # Analyze dataframe
        analyzed_df = analyzer.analyze_dataframe(df)
    else:
        logging.warning("No data files found in collected_data directory")
#[file content end]