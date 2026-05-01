# Sentiment Analysis Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

---

## 📋 Overview

This project is a **Binary Sentiment Analysis** application that classifies text into **Positive** or **Negative** sentiment using machine learning techniques. The model achieves **93% accuracy** on the test set.

### 🎯 Key Features

- **Text Preprocessing Pipeline**: Punctuation removal, tokenization, stopword removal, stemming
- **TF-IDF Vectorization**: Converts text into numerical features for ML models
- **Logistic Regression Model**: Fast, interpretable classification
- **Model Persistence**: Trained model and vectorizer saved as pickle files
- **Streamlit Web App**: Interactive demo deployed on Hugging Face Spaces
- **Well-Organized Codebase**: Modular structure with notebooks and utilities

---

## 📁 Project Structure

```
Sentiment_analysis/
├── README.md                      # Project documentation
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
│
├── notebooks/                     # Jupyter Notebooks for EDA and training
│   ├── nlp_project.ipynb         # Main exploration and model training
│   └── README.md                 # Notebook descriptions
│
├── src/                           # Core Python modules
│   ├── preprocessing.py           # Text cleaning and preprocessing functions
│   ├── train_model.py             # Model training pipeline
│   └── utils.py                   # Helper utilities
│
├── models/                        # Trained models (not committed)
│   ├── model.pkl                  # Trained Logistic Regression model
│   └── vectorizer.pkl             # Fitted TF-IDF vectorizer
│
└── data/                          # Data files (not committed)
    └── README.md                  # Data directory description
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rahul7065469/Sentiment_analysis.git
   cd Sentiment_analysis
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Running the Jupyter Notebook

```bash
jupyter notebook notebooks/nlp_project.ipynb
```

The notebook includes:
- Data loading and exploration
- Text preprocessing and feature engineering
- Model training and evaluation
- Prediction examples

#### Using the Trained Model

```python
import pickle
import numpy as np
from src.preprocessing import clean_and_preprocess

# Load saved model and vectorizer
model = pickle.load(open('models/model.pkl', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Preprocess text
text = "This product is amazing!"
cleaned_text = clean_and_preprocess(text)

# Vectorize and predict
vectorized = vectorizer.transform([cleaned_text])
prediction = model.predict(vectorized)[0]
sentiment = "Positive" if prediction == 1 else "Negative"

print(f"Text: {text}")
print(f"Sentiment: {sentiment}")
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 93% |
| Training Samples | 16,000 |
| Test Samples | 4,000 |
| Positive Samples | 7,238 (45%) |
| Negative Samples | 8,762 (55%) |

### Data Preprocessing Steps

1. **Sentiment Grouping**: 6 emotions → 2 classes (Positive, Negative)
2. **Lowercasing**: Convert to lowercase for consistency
3. **Punctuation Removal**: Remove special characters
4. **Tokenization**: Split text into words
5. **Stopword Removal**: Remove common English words (the, a, is, etc.)
6. **Noisy Character Removal**: Remove repeated characters and non-alphanumeric
7. **Stemming**: Reduce words to root form using Porter Stemmer

---

## 🔗 Live Demo

👉 **[Try the Sentiment Analysis App](https://huggingface.co/spaces/Rahul9971/Sentiment_analysis)**

The interactive web app is deployed on Hugging Face Spaces using Streamlit.

---

## 🛠️ Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **Scikit-learn** — Machine learning models and utilities
- **NLTK** — Natural Language Processing toolkit
- **Streamlit** — Web app framework
- **Jupyter Notebook** — Interactive notebooks

---

## 📝 Model Details

### Algorithm
- **Classifier**: Logistic Regression
- **Feature Extraction**: TF-IDF Vectorizer
- **Train-Test Split**: 80-20
- **Max Iterations**: 1000

### Why Logistic Regression?
- Fast and efficient for binary classification
- Interpretable results
- Performs well on text data
- Low memory footprint

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Commit: `git commit -m "Add your feature"`
5. Push: `git push origin feature/your-feature`
6. Submit a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Rahul** — [GitHub Profile](https://github.com/Rahul7065469)

---

## ⭐ Acknowledgments

- Inspired by NLP and sentiment analysis tutorials
- Thanks to the scikit-learn and NLTK communities
- Special thanks to Hugging Face for easy deployment

---

## 📞 Contact & Support

If you have questions or suggestions, feel free to open an **Issue** or reach out!

**Last Updated**: May 1, 2026
