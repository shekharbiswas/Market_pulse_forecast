import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re

# Download necessary NLTK data (only needed once)
nltk.download('vader_lexicon')
nltk.download('stopwords')

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to clean text and remove stop words
def clean_text(text):
    # Check if text is a string and not None
    if isinstance(text, str):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)

        # Convert to lowercase
        text = text.lower()
        
        # Remove stop words
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [w for w in words if not w in stop_words]
        text = " ".join(words)

        return text
    else:
        # Handle None or non-string values by returning an empty string or a placeholder
        return "" # or any suitable placeholder

# Apply the cleaning function to the 'text' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Create a function to assign sentiment labels based on sentiment score
def assign_sentiment_label(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


# Function to calculate sentiment score
def get_sentiment_score(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']

# Calculate sentiment scores for cleaned text and store it in the 'sentiment' column
df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment_score)

# Apply the function to create the 'sentiment_label' column
df['sentiment_label'] = df['sentiment_score'].apply(assign_sentiment_label)

# Calculate news_count in the last 24 hours
df['news_count_last_24h'] = list(df.groupby('symbol').rolling('24H', on='publishedDate')["title"].count().reset_index(0,drop=True))

# Calculate the average sentiment score in the last 24 hours
df['average_sentiment_last_24h'] = list(df.groupby('symbol').rolling('24H', on='publishedDate')['sentiment_score'].mean().reset_index(0,drop=True))

# Calculate the sentiment volatility (standard deviation) in the last 24 hours
df['news_volatility_impact'] = list(df.groupby('symbol').rolling('24H', on='publishedDate')['sentiment_score'].std().reset_index(0,drop=True))

df.to_csv('msft_stock_news_sentiment.csv', index=False) 
