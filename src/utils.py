"""Utility functions for sentiment analysis."""

import pickle
from typing import Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to a pickle file.
    
    Args:
        model: Trained model object
        filepath (str): Path to save the model
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from a pickle file.
    
    Args:
        filepath (str): Path to the saved model
    
    Returns:
        Loaded model object
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: str) -> None:
    """
    Save a fitted TF-IDF vectorizer to a pickle file.
    
    Args:
        vectorizer: Fitted TfidfVectorizer object
        filepath (str): Path to save the vectorizer
    """
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {filepath}")


def load_vectorizer(filepath: str) -> TfidfVectorizer:
    """
    Load a fitted TF-IDF vectorizer from a pickle file.
    
    Args:
        filepath (str): Path to the saved vectorizer
    
    Returns:
        Loaded TfidfVectorizer object
    """
    with open(filepath, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"Vectorizer loaded from {filepath}")
    return vectorizer


def predict_sentiment(text: str, model: LogisticRegression, 
                      vectorizer: TfidfVectorizer, 
                      preprocessor) -> Tuple[str, float]:
    """
    Predict sentiment for a given text.
    
    Args:
        text (str): Input text
        model: Trained classifier
        vectorizer: Fitted TfidfVectorizer
        preprocessor: Preprocessing function
    
    Returns:
        Tuple[str, float]: (sentiment_label, confidence)
    """
    # Preprocess
    cleaned = preprocessor(text)
    
    # Vectorize
    vectorized = vectorizer.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    # Map prediction
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = max(probability)
    
    return sentiment, confidence
