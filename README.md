# Sentiment_analysis
# Sentiment Analysis App

## Overview
This project is a Sentiment Analysis application that classifies movie reviews or comments as **positive** or **negative**. It demonstrates hands-on experience with:

- Text preprocessing to clean and prepare raw text data  
- Applying multiple machine learning models and selecting the best one based on evaluation  
- Creating an interactive frontend using [Streamlit](https://streamlit.io/) for user-friendly sentiment prediction  
- Deploying the application with [Hugging Face](https://huggingface.co/) for easy access and sharing  

## Features
- Clean and preprocess input text for accurate analysis  
- Load pre-trained vectorizer and classification model  
- Real-time sentiment prediction displayed via an attractive and interactive UI  

## Getting Started

### Prerequisites
- Python 3.9 or higher  
- pip (Python package installer)  

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-app.git
    cd sentiment-analysis-app
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App Locally

Run the Streamlit app with:

```bash
streamlit run src/streamlit_app.py
