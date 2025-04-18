import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import matplotlib.font_manager as fm
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
import logging
import emoji
import traceback
import contractions
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('vader_lexicon')
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
    nltk.download('stopwords')

# Ensure spaCy model is properly loaded
try:
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        st.warning("Downloading spaCy model... This may take a moment.")
        from spacy.cli import download
        download('en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')
except Exception as e:
    st.warning(f"Error with spaCy: {str(e)}. Some features will be disabled.")
    nlp = None

# Page Configuration
st.set_page_config(
    page_title="Call Transcript Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("📞 Call Transcript Sentiment Analysis")
st.markdown("""
This application analyzes call transcripts to extract sentiment, key topics, and emotional patterns.
Upload your transcript in text format to get started.
""")

# Custom emotion lexicons with more nuanced terms
EMOTION_LEXICONS = {
    'joy': [
        'happy', 'glad', 'delighted', 'pleased', 'satisfied', 'enjoy', 'exciting', 'excited', 
        'thank', 'thanks', 'appreciate', 'wonderful', 'great', 'perfect', 'excellent', 'thrilled',
        'ecstatic', 'joyful', 'elated', 'cheerful', 'content', 'blessed', 'gratitude', 'grateful',
        'love', 'positive', 'fantastic', 'awesome', 'good', 'pleasure', 'delight', 'thrilled'
    ],
    'anger': [
        'angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'upset', 'outraged', 
        'hate', 'resent', 'livid', 'enraged', 'hostile', 'incensed', 'aggravated', 'pissed',
        'infuriated', 'rage', 'temper', 'fed up', 'exasperated', 'disgusted', 'offended', 'irked',
        'displeased', 'indignant', 'fuming', 'heated', 'agitated', 'annoying', 'anger', 'irate'
    ],
    'sadness': [
        'sad', 'unhappy', 'disappointed', 'regret', 'sorry', 'unfortunate', 'depressed', 'miserable',
        'grief', 'heartbroken', 'despair', 'dejected', 'sorrow', 'gloomy', 'melancholy', 'tearful',
        'hurt', 'disheartened', 'blue', 'down', 'upset', 'distressed', 'misery', 'mourning', 'painful',
        'somber', 'hopeless', 'discouraged', 'broken', 'crushed', 'devastated', 'troubled'
    ],
    'fear': [
        'afraid', 'scared', 'worried', 'anxious', 'concerned', 'nervous', 'terrified', 'panic',
        'frightened', 'fearful', 'dread', 'alarmed', 'terror', 'horror', 'apprehensive', 'uneasy',
        'phobia', 'intimidated', 'petrified', 'paranoid', 'suspicious', 'threat', 'unsafe', 'risk',
        'danger', 'vulnerable', 'insecure', 'hesitant', 'desperate', 'distrust', 'doubt', 'uncomfortable'
    ],
    'surprise': [
        'surprised', 'amazed', 'astonished', 'shocked', 'unexpected', 'wow', 'oh', 'stunning',
        'startled', 'bewildered', 'dumbfounded', 'speechless', 'awestruck', 'flabbergasted', 'astounded',
        'incredible', 'unbelievable', 'sudden', 'unpredictable', 'extraordinary', 'remarkable', 'wonder',
        'disbelief', 'unexpected', 'staggering', 'striking', 'eye-opening', 'jaw-dropping'
    ],
    'confusion': [
        'confused', 'unsure', 'unclear', 'perplexed', 'misunderstood', 'don\'t understand', 'what do you mean',
        'puzzled', 'baffled', 'disoriented', 'lost', 'uncertain', 'ambiguous', 'doubtful', 'bewildered',
        'mixed up', 'muddled', 'confusing', 'disorganized', 'vague', 'complicated', 'complex', 'cryptic',
        'mistaken', 'misconception', 'misinterpretation', 'bewilderment', 'mystified', 'stumped'
    ],
    'trust': [
        'trust', 'believe', 'confidence', 'faith', 'sure', 'certain', 'assured', 'reliable',
        'dependable', 'honest', 'truth', 'loyal', 'faithful', 'credible', 'authentic', 'legitimate',
        'support', 'certainty', 'count on', 'rely on', 'help', 'assurance', 'security', 'safety'
    ],
    'anticipation': [
        'anticipate', 'expect', 'look forward', 'await', 'hope', 'eager', 'excited', 'soon',
        'prepare', 'ready', 'anticipation', 'prospect', 'waiting', 'upcoming', 'future', 'plan',
        'predict', 'foresee', 'forecast', 'looking ahead', 'pending', 'approaching', 'imminent'
    ]
}

# Industry-specific sentiment modifiers
CUSTOMER_SERVICE_MODIFIERS = {
    'positive': [
        'resolved', 'solution', 'solved', 'fixed', 'working', 'helped', 'support', 'assist',
        'quick', 'fast', 'immediate', 'refund', 'credit', 'discount', 'upgrade', 'apologize',
        'helpful', 'understand', 'responsive', 'efficient', 'prompt', 'clear', 'professional'
    ],
    'negative': [
        'error', 'issue', 'problem', 'bug', 'glitch', 'broken', 'not working', 'fail', 'failed',
        'wrong', 'mistake', 'delay', 'slow', 'wait', 'waiting', 'disconnected', 'drop', 'lost', 
        'charge', 'fee', 'expensive', 'cost', 'waste', 'frustrate', 'difficult', 'hard', 'useless',
        'unhelpful', 'rude', 'disrespectful', 'cancel', 'complaint', 'escalate', 'manager', 'supervisor'
    ]
}

# Functions for preprocessing
def clean_text(text):
    """Clean text with improved preprocessing"""
    if not text:
        return ""
    
    # Replace common contractions
    text = contractions.fix(text)
    
    # Handle emojis - convert to text
    text = emoji.demojize(text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Handle punctuation more carefully to keep sentence structure
    text = re.sub(r'([^\w\s\.\?\!])', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_speaker_content(text):
    """Enhanced extraction of content by speaker with multiple patterns"""
    # Multiple patterns for different transcript formats
    patterns = [
        # Format: "Speaker: Text"
        r'(Agent|Customer|Representative|Rep|Support|Client|Customer Service|CSR|User|Caller|A:|C:):\s*(.*?)(?=\n\s*(?:Agent|Customer|Representative|Rep|Support|Client|Customer Service|CSR|User|Caller|A:|C:):|$)',
        
        # Format: "[Speaker] Text"
        r'\[(Agent|Customer|Representative|Rep|Support|Client|Customer Service|CSR|User|Caller)\]\s*(.*?)(?=\n\s*\[(?:Agent|Customer|Representative|Rep|Support|Client|Customer Service|CSR|User|Caller)\]|$)',
        
        # Format: "Speaker - Text"
        r'(Agent|Customer|Representative|Rep|Support|Client|Customer Service|CSR|User|Caller)\s*-\s*(.*?)(?=\n\s*(?:Agent|Customer|Representative|Rep|Support|Client|Customer Service|CSR|User|Caller)\s*-|$)',
        
        # Format: "Speaker Name: Text" (e.g., "John: Hello")
        r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?):\s*(.*?)(?=\n\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)?:|$)',
        
        # Format: "(Speaker) Text"
        r'\(([A-Za-z\s]+)\)\s*(.*?)(?=\n\s*\([A-Za-z\s]+\)|$)',
        
        # Format: Time stamp with speaker "HH:MM:SS Speaker: Text"
        r'\d{1,2}:\d{2}(?::\d{2})?\s+([A-Za-z\s]+):\s*(.*?)(?=\n\s*\d{1,2}:\d{2}(?::\d{2})?\s+[A-Za-z\s]+:|$)'
    ]
    
    speakers = []
    contents = []
    
    # Try each pattern
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            for match in matches:
                speakers.append(match[0].strip())
                contents.append(match[1].strip())
            logger.info(f"Found {len(matches)} speaker segments using pattern: {pattern[:50]}...")
            break  # Use first successful pattern
    
    # If no patterns match, try to detect turn-taking based on line breaks
    if not speakers:
        logger.info("No structured speaker format found, trying to detect turn-taking")
        paragraphs = re.split(r'\n\s*\n', text)
        if len(paragraphs) > 1:
            for i, para in enumerate(paragraphs):
                if para.strip():  # Skip empty paragraphs
                    # Alternate between "Speaker 1" and "Speaker 2"
                    speaker = "Speaker 1" if i % 2 == 0 else "Speaker 2"
                    speakers.append(speaker)
                    contents.append(para.strip())
    
    # If still no speakers found but text exists, use "Unknown" for whole text
    if not speakers and text.strip():
        logger.info("No clear speaker structure found, using single speaker")
        speakers = ["Unknown"]
        contents = [text.strip()]
    
    return speakers, contents

def split_into_sentences(text, use_spacy=False):
    """Split text into sentences with improved sentence boundary detection"""
    if not text or not text.strip():
        return []
    
    # Use spaCy if available for better sentence segmentation
    if use_spacy and nlp:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # Fallback to NLTK
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

def blend_sentiment_scores(vader_scores, textblob_scores, context_factor=0.2):
    """Blend different sentiment analysis methods with context awareness"""
    # Base sentiment from VADER (known for handling social media text well)
    compound = vader_scores['compound']
    
    # Use TextBlob's polarity to adjust VADER score
    polarity_adjustment = textblob_scores['polarity'] * 0.3
    
    # Blend the scores
    blended_score = compound + polarity_adjustment
    
    # Normalize to the range [-1, 1]
    if blended_score > 1:
        blended_score = 1
    elif blended_score < -1:
        blended_score = -1
        
    return blended_score

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER with industry-specific adjustments"""
    if not text or not text.strip():
        return {'compound': 0, 'pos': 0, 'neg': 0, 'neu': 1}
    
    sia = SentimentIntensityAnalyzer()
    
    # Apply sentiment modifiers for customer service context
    cleaned_text = text.lower()
    for positive_term in CUSTOMER_SERVICE_MODIFIERS['positive']:
        if positive_term in cleaned_text:
            # Adjust text to emphasize positive terms
            text = text + " " + positive_term
    
    for negative_term in CUSTOMER_SERVICE_MODIFIERS['negative']:
        if negative_term in cleaned_text:
            # Add negative terms only if not preceded by "no", "not", "resolved", etc.
            if not re.search(r'\b(no|not|resolved|fixed|solved)\s+' + re.escape(negative_term), cleaned_text):
                text = text + " " + negative_term
    
    # Apply VADER sentiment analysis
    sentiment = sia.polarity_scores(text)
    return sentiment

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    if not text or not text.strip():
        return {'polarity': 0, 'subjectivity': 0}
    
    analysis = TextBlob(text)
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

def get_emotion_scores(text):
    """Get more granular emotion scores with enhanced lexicons"""
    if not text or not text.strip():
        return {emotion: 0 for emotion in EMOTION_LEXICONS}
    
    # Tokenize and clean words
    words = word_tokenize(clean_text(text.lower()))
    words = [word for word in words if word.isalpha()]  # Keep only alphabetic words
    
    # Calculate emotion scores with lexical analysis
    emotion_scores = {}
    word_count = max(1, len(words))  # Avoid division by zero
    
    for emotion, lexicon in EMOTION_LEXICONS.items():
        # Count words that match the emotion lexicon
        emotion_words = [word for word in words if word in lexicon]
        count = len(emotion_words)
        
        # Calculate percentage
        emotion_scores[emotion] = count / word_count * 100
        
    return emotion_scores

def extract_context_aware_topics(texts, num_topics=3):
    """Extract key topics from texts using LDA"""
    if not texts or len(texts) == 0 or all(not text.strip() for text in texts):
        return [], []
    
    try:
        # Create bag of words
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            max_features=50
        )
        
        # Feature extraction
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Check if we have enough features
        if X.shape[1] < num_topics + 1:
            num_topics = max(1, X.shape[1] - 1)
            if num_topics < 1:
                return [], []
        
        # Apply LDA
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            max_iter=10,
            learning_method='online',
            random_state=42
        )
        
        lda.fit(X)
        
        # Extract topics
        topics = []
        topic_keywords = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-6:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(f"Topic {topic_idx+1}")
            topic_keywords.append(", ".join(top_words))
        
        return topics, topic_keywords
    
    except Exception as e:
        st.error(f"Topic extraction error: {str(e)}")
        st.error(traceback.format_exc())
        return [], []

def detect_intent(text):
    """Detect customer intent from text"""
    # Dictionary of intents and their keywords
    intent_keywords = {
        'Technical Problem': ['error', 'issue', 'broken', 'not working', 'problem', 'fix', 'bug', 'glitch', 'crash'],
        'Billing Question': ['bill', 'charge', 'payment', 'credit', 'refund', 'price', 'cost', 'fee', 'discount', 'subscription'],
        'Account Management': ['account', 'password', 'login', 'sign in', 'access', 'profile', 'settings', 'update', 'change'],
        'Product Information': ['how to', 'feature', 'works', 'information', 'details', 'explain', 'tell me about', 'learn'],
        'Complaint': ['unhappy', 'dissatisfied', 'disappointed', 'complaint', 'complain', 'bad', 'worse', 'terrible', 'poor'],
        'Cancellation': ['cancel', 'terminate', 'end', 'stop', 'discontinue', 'close account'],
        'Service Request': ['help', 'support', 'assistance', 'service', 'schedule', 'book', 'appointment'],
        'Feedback': ['feedback', 'suggestion', 'idea', 'improve', 'better', 'review', 'rating']
    }
    
    # Check for intent matches
    text_lower = text.lower()
    intent_scores = {}
    
    for intent, keywords in intent_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        intent_scores[intent] = score
    
    # Get top intent
    if all(score == 0 for score in intent_scores.values()):
        return "General Inquiry"
    
    top_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
    return top_intent

def plot_sentiment_flow(sentences_df):
    """Plot the flow of sentiment throughout the conversation with context awareness"""
    if len(sentences_df) < 2:
        return None
    
    # Add a moving average for trend visualization
    window_size = min(3, len(sentences_df))
    sentences_df['compound_ma'] = sentences_df['compound'].rolling(window=window_size, min_periods=1).mean()
    
    # Create the figure
    fig = px.line(
        sentences_df, 
        x=sentences_df.index, 
        y=['compound', 'compound_ma'],
        title="Sentiment Flow Throughout the Call",
        labels={'index': 'Sentence Number', 'value': 'Sentiment Score'},
        color_discrete_map={
            'compound': 'lightblue',
            'compound_ma': 'black'
        },
        height=400
    )
    
    # Rename the traces for better legend
    fig.data[0].name = "Raw Sentiment"
    fig.data[1].name = "Trend (Moving Avg)"
    
    # Add threshold lines for sentiment categories
    fig.add_shape(
        type="line", 
        x0=0, y0=0.05, x1=len(sentences_df), y1=0.05,
        line=dict(color="green", width=1, dash="dash"),
    )
    
    fig.add_shape(
        type="line", 
        x0=0, y0=-0.05, x1=len(sentences_df), y1=-0.05,
        line=dict(color="red", width=1, dash="dash"),
    )
    
    # Add annotations for significant sentiment points
    sentiment_peaks = []
    sentiment_valleys = []
    
    # Find significant peaks and valleys in the compound score
    for i in range(1, len(sentences_df) - 1):
        if sentences_df.iloc[i]['compound'] > sentences_df.iloc[i-1]['compound'] and sentences_df.iloc[i]['compound'] > sentences_df.iloc[i+1]['compound']:
            if sentences_df.iloc[i]['compound'] >= 0.5:
                sentiment_peaks.append(i)
        elif sentences_df.iloc[i]['compound'] < sentences_df.iloc[i-1]['compound'] and sentences_df.iloc[i]['compound'] < sentences_df.iloc[i+1]['compound']:
            if sentences_df.iloc[i]['compound'] <= -0.5:
                sentiment_valleys.append(i)
    
    # Add annotations for the peaks and valleys
    for peak in sentiment_peaks:
        fig.add_annotation(
            x=peak, y=sentences_df.iloc[peak]['compound'],
            text="Positive Peak",
            showarrow=True,
            arrowhead=1
        )
    
    for valley in sentiment_valleys:
        fig.add_annotation(
            x=valley, y=sentences_df.iloc[valley]['compound'],
            text="Negative Valley",
            showarrow=True,
            arrowhead=1
        )
    
    fig.update_layout(
        legend_title_text='Sentiment Type',
        hovermode="x unified",
        xaxis=dict(title='Sentence Number'),
        yaxis=dict(title='Sentiment Score')
    )
    
    return fig

def plot_speaker_sentiment(speakers, contents, sentiments):
    """Plot sentiment by speaker with enhanced analysis"""
    if not speakers or len(speakers) == 0:
        return None
    
    # Create dataframe with speaker and sentiment data
    data = []
    for speaker, content, sentiment in zip(speakers, contents, sentiments):
        # Add basic sentiment scores
        row = {
            'Speaker': speaker,
            'Content Length': len(content.split()),
            'Compound': sentiment['compound'],
            'Positive': sentiment['pos'],
            'Negative': sentiment['neg'],
            'Neutral': sentiment['neu']
        }
        
        # Add emotion scores
        emotion_scores = get_emotion_scores(content)
        for emotion, score in emotion_scores.items():
            row[emotion] = score
        
        # Detect intent
        row['Intent'] = detect_intent(content)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Group by speaker and calculate means
    speaker_sentiment = df.groupby('Speaker').mean().reset_index()
    
    # Create a more complex visualization
    fig = go.Figure()
    
    # Add sentiment bars for each speaker
    for speaker in speaker_sentiment['Speaker'].unique():
        speaker_data = speaker_sentiment[speaker_sentiment['Speaker'] == speaker]
        
        fig.add_trace(go.Bar(
            x=[speaker],
            y=[speaker_data['Positive'].values[0]],
            name=f"{speaker} - Positive",
            marker_color='green',
            width=0.2,
            offset=-0.2
        ))
        
        fig.add_trace(go.Bar(
            x=[speaker],
            y=[speaker_data['Negative'].values[0]],
            name=f"{speaker} - Negative",
            marker_color='red',
            width=0.2,
            offset=0
        ))
        
        fig.add_trace(go.Bar(
            x=[speaker],
            y=[speaker_data['Neutral'].values[0]],
            name=f"{speaker} - Neutral",
            marker_color='gray',
            width=0.2,
            offset=0.2
        ))
    
    # Add a horizontal line for the compound sentiment
    for speaker in speaker_sentiment['Speaker'].unique():
        speaker_data = speaker_sentiment[speaker_sentiment['Speaker'] == speaker]
        
        fig.add_trace(go.Scatter(
            x=[speaker, speaker],
            y=[0, speaker_data['Compound'].values[0]],
            mode='lines',
            name=f"{speaker} - Compound",
            line=dict(color='black', width=3)
        ))
    
    fig.update_layout(
        title="Sentiment Analysis by Speaker",
        xaxis_title="Speaker",
        yaxis_title="Sentiment Score",
        barmode='group',
        height=500
    )
    
    return fig

def generate_wordcloud(text, title="Word Cloud", mask_color=None):
    """Generate an improved wordcloud with better preprocessing"""
    if not text or not text.strip():
        return None
    
    # Try to locate a common font
    common_fonts = ['Arial', 'DejaVuSans', 'Tahoma', 'Verdana', 'Helvetica', 'Times New Roman']
    font_path = None
    
    for font in common_fonts:
        font_files = fm.findSystemFonts()
        for file in font_files:
            if font.lower() in os.path.basename(file).lower():
                font_path = file
                break
        if font_path:
            break
    
    # Enhanced text preprocessing
    # Remove stopwords and common filler words
    stopwords_list = set(stopwords.words('english'))
    # Add common filler words in customer service
    additional_stopwords = {'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally',
                           'so', 'just', 'okay', 'ok', 'right', 'well', 'anyway', 'hmm'}
    stopwords_list.update(additional_stopwords)
    
    # Tokenize, clean, and filter words
    words = word_tokenize(clean_text(text.lower()))
    filtered_words = [word for word in words if word.lower() not in stopwords_list and len(word) > 2]
    filtered_text = " ".join(filtered_words)
    
    try:
        # Create WordCloud with custom parameters
        if font_path:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                font_path=font_path,
                colormap=mask_color,
                collocations=True,  # Include bigrams
                min_word_length=3,
                prefer_horizontal=0.9
            ).generate(filtered_text)
        else:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                colormap=mask_color,
                collocations=True,  # Include bigrams
                min_word_length=3,
                prefer_horizontal=0.9
            ).generate(filtered_text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        return fig
    except ValueError as e:
        logger.error(f"WordCloud generation error: {str(e)}")
        # Fallback to bar chart
        st.warning("WordCloud generation failed. Displaying top words instead.")
        words = filtered_text.lower().split()
        word_count = {}
        for word in words:
            if len(word) > 3:  # Ignore very short words
                word_count[word] = word_count.get(word, 0) + 1
        
        # Sort and get top words
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:20]
        words = [w[0] for w in sorted_words]
        counts = [w[1] for w in sorted_words]
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(words, counts)
        ax.set_title(f"Top Words in {title}")
        ax.set_xlabel("Word")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig

def apply_contextual_sentiment_analysis(sentences_df, context_window=3):
    """Apply contextual sentiment analysis considering surrounding sentences"""
    if len(sentences_df) <= 1:
        return sentences_df
    
    # Create a copy of the DataFrame
    context_df = sentences_df.copy()
    
    # Add context-aware sentiment
    context_df['context_compound'] = context_df['compound'].copy()
    
    # Apply context window to adjust sentiment
    for i in range(len(context_df)):
        # Get surrounding sentences indices within the context window
        start_idx = max(0, i - context_window)
        end_idx = min(len(context_df), i + context_window + 1)
        
        # Get current sentence sentiment
        current_sentiment = context_df.iloc[i]['compound']
        
        # Get surrounding sentiments and calculate weighted average
        surrounding_sentiments = context_df.iloc[start_idx:end_idx]['compound'].values
        
        # Create weights (higher weight for current sentence)
        weights = [0.5 if j == i else (1 / (abs(j - i) + 1)) / (2 * context_window) 
                  for j in range(start_idx, end_idx)]
        
        # Calculate weighted average
        if len(surrounding_sentiments) == len(weights):
            context_sentiment = np.average(surrounding_sentiments, weights=weights)
            context_df.loc[i, 'context_compound'] = context_sentiment
    
    return context_df

def plot_emotions(df):
    """Plot emotion distribution in the conversation"""
    if len(df) == 0 or 'joy' not in df.columns:
        return None
    
    # Get emotion columns
    emotion_cols = [col for col in df.columns if col in EMOTION_LEXICONS.keys()]
    
    if not emotion_cols:
        return None
    
    # Calculate average emotion scores
    emotion_avgs = {emotion: df[emotion].mean() for emotion in emotion_cols}
    
    # Create radar chart using plotly
    emotions = list(emotion_avgs.keys())
    values = list(emotion_avgs.values())
    
    # Complete the loop for radar chart
    values.append(values[0])
    emotions.append(emotions[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        name='Emotions'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2]
            )
        ),
        title="Emotion Distribution in Conversation",
        showlegend=False
    )
    
    return fig

def identify_key_moments(sentences_df, threshold=0.5):
    """Identify key moments in the conversation based on sentiment changes"""
    if len(sentences_df) < 5:  # Need enough sentences for meaningful analysis
        return pd.DataFrame()
    
    key_moments = []
    
    # Look for significant sentiment shifts
    for i in range(1, len(sentences_df)):
        prev_sentiment = sentences_df.iloc[i-1]['compound']
        current_sentiment = sentences_df.iloc[i]['compound']
        
        # Calculate sentiment shift
        shift = abs(current_sentiment - prev_sentiment)
        
        # If significant shift detected
        if shift >= threshold:
            direction = "positive" if current_sentiment > prev_sentiment else "negative"
            key_moments.append({
                'Sentence_Number': i,
                'Text': sentences_df.iloc[i]['text'],
                'Speaker': sentences_df.iloc[i]['speaker'],
                'Sentiment_Before': prev_sentiment,
                'Sentiment_After': current_sentiment,
                'Shift_Magnitude': shift,
                'Direction': direction
            })
    
    # If no key moments found with the threshold, try with a lower threshold
    if not key_moments and threshold > 0.3:
        return identify_key_moments(sentences_df, threshold=0.3)
    
    return pd.DataFrame(key_moments)

def extract_customer_pain_points(text, speaker):
    """Extract potential customer pain points from the conversation"""
    if 'customer' not in speaker.lower() or not text:
        return []
    
    pain_point_indicators = [
        r'(not happy|unhappy|disappointed|frustrated|annoyed) with',
        r'(issue|problem|trouble|difficull?t|challenge) with',
        r'(doesn\'?t|did not|doesn\'?t|won\'?t|can\'?t|unable to) work',
        r'(failed|failing|fails) to',
        r'(error|warning|alert)',
        r'too (slow|fast|complicated|difficult|hard|confusing)',
        r'(waste|wasting) (?:of )?(?:my )?time',
        r'(ridiculous|absurd|unacceptable)',
        r'(refund|cancel|return|money back)',
        r'(spoke|talking|talked) to (?:multiple|several|many) (?:people|representatives)',
        r'(waiting|waited) (?:for|on hold)',
        r'(escalate|supervisor|manager|complaint)'
    ]
    
    pain_points = []
    for pattern in pain_point_indicators:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            # Get context around the pain point (up to 100 chars)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            pain_point_context = text[start:end]
            
            # Add to the list if not already included
            if pain_point_context not in [p['context'] for p in pain_points]:
                pain_points.append({
                    'indicator': match.group(0),
                    'context': pain_point_context,
                    'pattern': pattern
                })
    
    return pain_points

def calculate_conversation_metrics(df):
    """Calculate various conversation metrics"""
    if len(df) == 0:
        return {}
    
    # Speaker distribution
    speaker_counts = df['speaker'].value_counts().to_dict()
    total_utterances = len(df)
    
    # Sentiment metrics
    avg_sentiment = df['compound'].mean()
    sentiment_volatility = df['compound'].std()
    
    # Content metrics
    avg_response_length = df.groupby('speaker')['text'].apply(lambda x: sum(len(s.split()) for s in x) / len(x)).to_dict()
    
    # Turn-taking dynamics
    turns = df['speaker'].tolist()
    speaker_transitions = [1 if i > 0 and turns[i] != turns[i-1] else 0 for i in range(len(turns))]
    turn_taking_rate = sum(speaker_transitions) / (len(turns) - 1) if len(turns) > 1 else 0
    
    # Intent distribution
    if 'intent' in df.columns:
        intent_counts = df['intent'].value_counts().to_dict()
    else:
        intent_counts = {}
    
    # Emotion metrics
    emotion_cols = [col for col in df.columns if col in EMOTION_LEXICONS.keys()]
    emotion_metrics = {}
    if emotion_cols:
        for emotion in emotion_cols:
            emotion_metrics[emotion] = df[emotion].mean()
    
    metrics = {
        'total_utterances': total_utterances,
        'speaker_distribution': speaker_counts,
        'avg_sentiment': avg_sentiment,
        'sentiment_volatility': sentiment_volatility,
        'avg_response_length': avg_response_length,
        'turn_taking_rate': turn_taking_rate,
        'intent_distribution': intent_counts,
        'emotion_metrics': emotion_metrics
    }
    
    return metrics

def main():
    """Main application with enhanced UI/UX"""
    # Create separate tabs for different analysis sections
    tabs = st.tabs(["Upload & Basic Analysis", "Sentiment Analysis", "Topic Analysis", "Advanced Insights"])
    
    # Upload & Basic Analysis Tab
    with tabs[0]:
        st.header("Upload Transcript")
        
        # File upload with multiple format options
        uploaded_file = st.file_uploader("Upload your call transcript (TXT, CSV, PDF)", 
                                        type=["txt", "csv", "pdf"])
        
        # Sample data option
        use_sample = st.checkbox("Use sample data instead")
        
        # Process the transcript
        transcript_text = ""
        
        if use_sample:
            transcript_text = """
            Customer: Hi, I've been having trouble with my account. I can't seem to log in.
            Agent: I'm sorry to hear that. I'd be happy to help you with that issue. Can you tell me what happens when you try to log in?
            Customer: It says my password is incorrect, but I'm positive I'm using the right one. I've tried multiple times.
            Agent: I understand how frustrating that can be. Let me check your account status. Could you please provide your email address?
            Customer: Sure, it's johndoe@example.com.
            Agent: Thank you. I'm checking your account now... I can see there were multiple failed login attempts, which triggered our security system to temporarily lock your account. That's why you're having trouble.
            Customer: Oh great, now I'm locked out? This is ridiculous. I need access today to submit an important form.
            Agent: I completely understand your concern, especially with an important deadline. The good news is I can help unlock your account right now and get you back in quickly.
            Customer: Well, that would be helpful at least. How long will it take?
            Agent: I can reset it immediately. You'll receive an email with a temporary password within the next 5 minutes. Would you like me to do that for you now?
            Customer: Yes, please. That would be great.
            Agent: Perfect! I've just processed the reset. Please check your email soon for the temporary password. Once you log in, you'll be prompted to create a new password of your choice.
            Customer: Thank you, I appreciate your help with this.
            Agent: You're very welcome! Is there anything else I can assist you with today?
            Customer: No, that's all. Thanks again.
            Agent: It was my pleasure helping you today. Thank you for contacting us, and have a wonderful day!
            """
        elif uploaded_file is not None:
            # Handle different file types
            file_extension = uploaded_file.name.split(".")[-1].lower()
            
            if file_extension == "txt":
                transcript_text = uploaded_file.getvalue().decode("utf-8")
            elif file_extension == "csv":
                try:
                    df = pd.read_csv(uploaded_file)
                    # Assume first column is speaker, second is text
                    if len(df.columns) >= 2:
                        speakers = df.iloc[:, 0].tolist()
                        contents = df.iloc[:, 1].tolist()
                        transcript_text = "\n".join([f"{s}: {c}" for s, c in zip(speakers, contents)])
                    else:
                        st.error("CSV format not recognized. Please ensure it has at least two columns.")
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
            elif file_extension == "pdf":
                st.error("PDF processing requires additional libraries. Please upload a TXT or CSV file instead.")
        
        if transcript_text:
            st.success("Transcript loaded successfully!")
            
            # Display raw transcript with option to expand/collapse
            with st.expander("View Raw Transcript", expanded=False):
                st.text_area("", transcript_text, height=200)
            
            # Process the transcript
            speakers, contents = extract_speaker_content(transcript_text)
            
            # Basic statistics
            st.header("Basic Transcript Information")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", len(speakers))
            with col2:
                unique_speakers = len(set(speakers))
                st.metric("Number of Speakers", unique_speakers)
            with col3:
                total_words = sum(len(content.split()) for content in contents)
                st.metric("Total Word Count", total_words)
            
            # Display processed conversation
            st.header("Processed Conversation")
            
            # Create a DataFrame for the conversation
            conversation_df = pd.DataFrame({
                'speaker': speakers,
                'text': contents
            })
            
            # Apply sentiment analysis to each message
            compound_scores = []
            text_blob_scores = []
            
            for content in contents:
                vader_sentiment = analyze_sentiment_vader(content)
                textblob_sentiment = analyze_sentiment_textblob(content)
                
                compound_scores.append(vader_sentiment)
                text_blob_scores.append(textblob_sentiment)
            
            # Add sentiments to DataFrame
            for i, (vader, textblob) in enumerate(zip(compound_scores, text_blob_scores)):
                conversation_df.loc[i, 'compound'] = vader['compound']
                conversation_df.loc[i, 'positive'] = vader['pos']
                conversation_df.loc[i, 'negative'] = vader['neg']
                conversation_df.loc[i, 'neutral'] = vader['neu']
                conversation_df.loc[i, 'polarity'] = textblob['polarity']
                conversation_df.loc[i, 'subjectivity'] = textblob['subjectivity']
            
            # Add emotions to DataFrame
            for i, content in enumerate(contents):
                emotion_scores = get_emotion_scores(content)
                for emotion, score in emotion_scores.items():
                    conversation_df.loc[i, emotion] = score
            
            # Add intent detection
            conversation_df['intent'] = conversation_df['text'].apply(detect_intent)
            
            # Display conversation table with sentiment indicators
            st.dataframe(
                conversation_df[['speaker', 'text', 'compound', 'intent']].style.apply(
                    lambda x: ['background-color: #CCFFCC' if x['compound'] > 0.05 
                              else 'background-color: #FFCCCC' if x['compound'] < -0.05 
                              else 'background-color: #F0F0F0' for _ in x],
                    axis=1
                ),
                height=400
            )
    
    # Sentiment Analysis Tab
    with tabs[1]:
        if 'conversation_df' in locals():
            st.header("Sentiment Analysis")
            
            # Overall sentiment summary
            st.subheader("Overall Sentiment Summary")
            
            avg_sentiment = conversation_df['compound'].mean()
            sentiment_volatility = conversation_df['compound'].std()
            
            col1, col2 = st.columns(2)
            with col1:
                # Sentiment score gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=avg_sentiment,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Average Sentiment"},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-1, -0.05], 'color': "red"},
                            {'range': [-0.05, 0.05], 'color': "gray"},
                            {'range': [0.05, 1], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': avg_sentiment
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                # Sentiment breakdown donut chart
                positive = (conversation_df['compound'] > 0.05).sum()
                negative = (conversation_df['compound'] < -0.05).sum()
                neutral = len(conversation_df) - positive - negative
                
                labels = ['Positive', 'Neutral', 'Negative']
                values = [positive, neutral, negative]
                colors = ['green', 'gray', 'red']
                
                fig = go.Figure(data=[go.Pie(
                    labels=labels, 
                    values=values,
                    hole=.4,
                    marker_colors=colors
                )])
                fig.update_layout(title_text="Sentiment Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment flow analysis
            st.subheader("Sentiment Flow Analysis")
            
            # Process all sentences to analyze sentiment flow
            all_sentences = []
            
            for i, row in conversation_df.iterrows():
                sentences = split_into_sentences(row['text'])
                for sentence in sentences:
                    if sentence:  # Skip empty sentences
                        all_sentences.append({
                            'text': sentence,
                            'speaker': row['speaker']
                        })
            
            sentences_df = pd.DataFrame(all_sentences)
            
            # Apply sentiment analysis to each sentence
            for i, sentence in enumerate(sentences_df['text']):
                vader_sentiment = analyze_sentiment_vader(sentence)
                textblob_sentiment = analyze_sentiment_textblob(sentence)
                
                sentences_df.loc[i, 'compound'] = vader_sentiment['compound']
                sentences_df.loc[i, 'positive'] = vader_sentiment['pos']
                sentences_df.loc[i, 'negative'] = vader_sentiment['neg']
                sentences_df.loc[i, 'neutral'] = vader_sentiment['neu']
                sentences_df.loc[i, 'polarity'] = textblob_sentiment['polarity']
                sentences_df.loc[i, 'subjectivity'] = textblob_sentiment['subjectivity']
            
            # Apply contextual sentiment analysis
            sentences_df = apply_contextual_sentiment_analysis(sentences_df)
            
            # Plot sentiment flow
            flow_fig = plot_sentiment_flow(sentences_df)
            if flow_fig:
                st.plotly_chart(flow_fig, use_container_width=True)
            
            # Plot sentiment by speaker
            st.subheader("Sentiment by Speaker")
            speaker_fig = plot_speaker_sentiment(speakers, contents, compound_scores)
            if speaker_fig:
                st.plotly_chart(speaker_fig, use_container_width=True)
            
            # Emotion analysis
            st.subheader("Emotion Analysis")
            emotion_fig = plot_emotions(conversation_df)
            if emotion_fig:
                st.plotly_chart(emotion_fig, use_container_width=True)
            
            # Key moments based on sentiment shifts
            st.subheader("Key Conversation Moments")
            key_moments = identify_key_moments(sentences_df)
            
            if not key_moments.empty:
                st.dataframe(key_moments[['Sentence_Number', 'Speaker', 'Text', 'Direction', 'Shift_Magnitude']])
            else:
                st.info("No significant sentiment shifts detected in this conversation.")
        else:
            st.info("Please upload a transcript in the 'Upload & Basic Analysis' tab to see sentiment analysis.")
    
    # Topic Analysis Tab
    with tabs[2]:
        if 'conversation_df' in locals():
            st.header("Topic Analysis")
            
            # Word clouds
            st.subheader("Word Clouds")
            
            col1, col2 = st.columns(2)
            
            # All text combined
            all_text = " ".join(contents)
            with col1:
                all_cloud = generate_wordcloud(all_text, title="Overall Conversation")
                if all_cloud:
                    st.pyplot(all_cloud)
            
            # Split by customer and agent if those speakers exist
            customer_texts = " ".join([text for speaker, text in zip(speakers, contents) 
                                      if 'customer' in speaker.lower()])
            
            agent_texts = " ".join([text for speaker, text in zip(speakers, contents) 
                                   if any(role in speaker.lower() for role in ['agent', 'rep', 'support'])])
            
            if customer_texts:
                with col2:
                    customer_cloud = generate_wordcloud(customer_texts, title="Customer's Words", mask_color="Blues")
                    if customer_cloud:
                        st.pyplot(customer_cloud)
            
            if agent_texts:
                col3, col4 = st.columns(2)
                with col3:
                    agent_cloud = generate_wordcloud(agent_texts, title="Agent's Words", mask_color="Greens")
                    if agent_cloud:
                        st.pyplot(agent_cloud)
            
            # Topic modeling
            st.subheader("Topic Modeling")
            
            # Extract topics from the conversation
            topics, topic_keywords = extract_context_aware_topics(contents)
            
            if topics and topic_keywords:
                for topic, keywords in zip(topics, topic_keywords):
                    st.write(f"**{topic}:** {keywords}")
                
                # Create horizontal bar chart for topic distribution
                try:
                    # Create vectorizer
                    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
                    X = vectorizer.fit_transform(contents)
                    
                    # Create LDA model
                    lda = LatentDirichletAllocation(n_components=len(topics), random_state=42)
                    lda.fit(X)
                    
                    # Get topic distribution for each document
                    topic_distribution = lda.transform(X)
                    
                    # Calculate overall topic presence
                    topic_presence = topic_distribution.sum(axis=0)
                    topic_presence = topic_presence / topic_presence.sum() * 100
                    
                    # Create horizontal bar chart
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=topics,
                        x=topic_presence,
                        orientation='h',
                        marker=dict(color='skyblue')
                    ))
                    
                    fig.update_layout(
                        title="Topic Distribution",
                        xaxis_title="Percentage (%)",
                        yaxis_title="Topics",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating topic distribution chart: {str(e)}")
            else:
                st.info("Not enough text data for meaningful topic extraction.")
            
            # Common n-grams
            st.subheader("Common Phrases (N-grams)")
            
            try:
                # Extract bigrams and trigrams
                cleaned_text = clean_text(all_text)
                tokens = word_tokenize(cleaned_text.lower())
                stop_words = set(stopwords.words('english'))
                filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words and len(w) > 2]
                
                # Create bigrams
                bigrams = list(zip(filtered_tokens, filtered_tokens[1:]))
                bigram_freq = Counter(bigrams)
                common_bigrams = bigram_freq.most_common(10)
                
                # Create trigrams
                trigrams = list(zip(filtered_tokens, filtered_tokens[1:], filtered_tokens[2:]))
                trigram_freq = Counter(trigrams)
                common_trigrams = trigram_freq.most_common(10)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Display bigrams
                    st.write("**Common Bigrams (Word Pairs)**")
                    if common_bigrams:
                        bigram_labels = [" ".join(bg[0]) for bg in common_bigrams]
                        bigram_values = [bg[1] for bg in common_bigrams]
                        
                        fig = go.Figure(go.Bar(
                            x=bigram_values,
                            y=bigram_labels,
                            orientation='h',
                            marker_color='lightblue'
                        ))
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No common bigrams found.")
                
                with col2:
                    # Display trigrams
                    st.write("**Common Trigrams (Three-Word Phrases)**")
                    if common_trigrams:
                        trigram_labels = [" ".join(tg[0]) for tg in common_trigrams]
                        trigram_values = [tg[1] for tg in common_trigrams]
                        
                        fig = go.Figure(go.Bar(
                            x=trigram_values,
                            y=trigram_labels,
                            orientation='h',
                            marker_color='lightgreen'
                        ))
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No common trigrams found.")
            except Exception as e:
                st.error(f"Error analyzing phrases: {str(e)}")
        else:
            st.info("Please upload a transcript in the 'Upload & Basic Analysis' tab to see topic analysis.")
    
    # Advanced Insights Tab
    with tabs[3]:
        if 'conversation_df' in locals():
            st.header("Advanced Conversation Insights")
            
            # Customer pain points
            st.subheader("Customer Pain Points")
            
            # Extract pain points from customer messages
            all_pain_points = []
            for speaker, content in zip(speakers, contents):
                if 'customer' in speaker.lower():
                    pain_points = extract_customer_pain_points(content, speaker)
                    all_pain_points.extend(pain_points)
            
            if all_pain_points:
                pain_points_df = pd.DataFrame(all_pain_points)
                st.dataframe(pain_points_df[['indicator', 'context']])
            else:
                st.info("No clear customer pain points detected in this conversation.")
            
            # Conversation metrics
            st.subheader("Conversation Metrics")
            
            metrics = calculate_conversation_metrics(conversation_df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Turn-Taking Rate", f"{metrics['turn_taking_rate']:.2f}")
                st.caption("Higher values indicate more back-and-forth conversation")
            
            with col2:
                st.metric("Sentiment Volatility", f"{metrics['sentiment_volatility']:.2f}")
                st.caption("Higher values indicate more emotional ups and downs")
            
            with col3:
                # Find dominant emotion
                if metrics['emotion_metrics']:
                    dominant_emotion = max(metrics['emotion_metrics'].items(), key=lambda x: x[1])
                    st.metric("Dominant Emotion", f"{dominant_emotion[0].capitalize()}")
                    st.caption(f"Score: {dominant_emotion[1]:.2f}")
                else:
                    st.metric("Dominant Emotion", "Neutral")
            
            # Customer intent distribution
            if 'intent_distribution' in metrics and metrics['intent_distribution']:
                st.subheader("Customer Intent Distribution")
                
                # Filter for customer intents only
                customer_intents = {intent: count for intent, count in metrics['intent_distribution'].items()
                                   if any(speaker == 'Customer' for speaker in conversation_df[conversation_df['intent'] == intent]['speaker'])}
                
                if customer_intents:
                    fig = go.Figure(data=[go.Pie(
                        labels=list(customer_intents.keys()),
                        values=list(customer_intents.values()),
                        hole=.3
                    )])
                    
                    fig.update_layout(title_text="Customer Intent Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No customer intents detected.")
            
            # Conversation flow visualization
            st.subheader("Conversation Flow")
            
            try:
                # Create a timeline of speakers
                timeline_data = []
                current_position = 0
                
                for i, (speaker, text) in enumerate(zip(speakers, contents)):
                    # Calculate text length (word count)
                    text_length = len(text.split())
                    
                    # Get sentiment
                    sentiment = conversation_df.iloc[i]['compound']
                    
                    # Determine color based on sentiment
                    if sentiment > 0.05:
                        color = 'rgba(0, 255, 0, 0.3)'  # Green for positive
                    elif sentiment < -0.05:
                        color = 'rgba(255, 0, 0, 0.3)'  # Red for negative
                    else:
                        color = 'rgba(150, 150, 150, 0.3)'  # Gray for neutral
                    
                    # Add to timeline
                    timeline_data.append(dict(
                        Task=speaker,
                        Start=current_position,
                        Finish=current_position + text_length,
                        Resource=speaker,
                        Color=color,
                        Text=text[:50] + "..." if len(text) > 50 else text
                    ))
                    
                    current_position += text_length
                
                # Create timeline chart
                fig = px.timeline(
                    timeline_data, 
                    x_start="Start", 
                    x_end="Finish", 
                    y="Task", 
                    color="Resource",
                    hover_data=["Text"]
                )
                
                # Update colors
                for i, trace in enumerate(fig.data):
                    fig.data[i].marker.color = timeline_data[i]['Color']
                
                fig.update_layout(title="Conversation Flow", height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creating conversation flow: {str(e)}")
            
            # Conversation Summary
            st.subheader("Conversation Summary")
            
            try:
                # Generate summary based on key parts of the conversation
                summary_points = []
                
                # Add opening
                if speakers and contents and len(speakers) > 0:
                    summary_points.append(f"• The conversation begins with {speakers[0]} saying: \"{contents[0][:100]}...\"")
                
                # Add sentiment overview
                avg_sentiment = metrics['avg_sentiment']
                if avg_sentiment > 0.25:
                    sentiment_desc = "overall positive"
                elif avg_sentiment < -0.25:
                    sentiment_desc = "overall negative"
                else:
                    sentiment_desc = "generally neutral"
                    
                summary_points.append(f"• The conversation tone was {sentiment_desc} (score: {avg_sentiment:.2f}).")
                
                # Add customer pain points if any
                if all_pain_points:
                    summary_points.append(f"• {len(all_pain_points)} potential customer pain points were identified.")
                    if len(all_pain_points) > 0:
                        top_pain = all_pain_points[0]['indicator']
                        summary_points.append(f"• Main customer concern: \"{top_pain}\"")
                
                # Add topic information
                if topics and topic_keywords and len(topics) > 0:
                    summary_points.append(f"• Main topic discussed: {topic_keywords[0]}")
                
                # Add closing
                if speakers and contents and len(speakers) > 1:
                    summary_points.append(f"• The conversation ends with {speakers[-1]} saying: \"{contents[-1][:100]}...\"")
                
                # Display summary
                for point in summary_points:
                    st.write(point)
                pass
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.error(traceback.format_exc())
                
        else:
            st.info("Please upload a transcript in the 'Upload & Basic Analysis' tab to see advanced insights.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    execution_time = time.time() - start_time
    st.sidebar.info(f"Analysis completed in {execution_time:.2f} seconds")
