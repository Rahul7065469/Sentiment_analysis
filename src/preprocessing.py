"""Text preprocessing utilities for sentiment analysis."""

import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()


def clean_text(text: str) -> str:
    """
    Remove punctuation from text.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Text without punctuation
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text: str) -> str:
    """
    Remove English stopwords from text.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Text without stopwords
    """
    words = word_tokenize(text)
    cleaned = [word for word in words if word not in stop_words]
    return ' '.join(cleaned)


def remove_noisy_characters(text: str) -> str:
    """
    Remove repeated characters and non-alphanumeric characters.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Cleaned text
    """
    words = word_tokenize(text)
    cleaned = []
    for word in words:
        # Check for repeated characters (3+) and non-alphanumeric
        if not re.search(r'(.)\1{2,}', word) and not re.search(r'\W', word):
            cleaned.append(word)
    return ' '.join(cleaned)


def stem_words(text: str) -> str:
    """
    Apply Porter Stemming to reduce words to root form.
    
    Args:
        text (str): Input text
    
    Returns:
        str: Text with stemmed words
    """
    words = word_tokenize(text)
    stemmed = [ps.stem(word) for word in words]
    return ' '.join(stemmed)


def clean_and_preprocess(text: str) -> str:
    """
    Complete preprocessing pipeline: lowercasing, cleaning, tokenization,
    stopword removal, noise removal, and stemming.
    
    Args:
        text (str): Raw input text
    
    Returns:
        str: Fully preprocessed text
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = clean_text(text)
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    # Remove noisy characters
    text = remove_noisy_characters(text)
    
    # Stem words
    text = stem_words(text)
    
    # Remove digit-only and non-ASCII words
    words = text.split()
    filtered = [w for w in words if not w.isdigit() and w.isascii()]
    
    return ' '.join(filtered)
