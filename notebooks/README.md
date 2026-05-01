# Jupyter Notebooks

This directory contains Jupyter notebooks used for data exploration, model training, and experimentation.

## Notebooks

### `nlp_project.ipynb`
- **Purpose**: Main notebook for sentiment analysis project
- **Contents**:
  - Data loading and exploration
  - Text preprocessing pipeline
  - Sentiment label transformation (6 emotions → 2 classes)
  - Feature extraction using TF-IDF vectorization
  - Model training with Logistic Regression
  - Model evaluation and accuracy metrics
  - Prediction examples
  - Model persistence (pickle saving)

#### Steps in the Notebook

1. **Import Libraries** — Load pandas, scikit-learn, NLTK, etc.
2. **Load Data** — Read training data from CSV
3. **Exploratory Data Analysis** — Check for nulls, class distribution
4. **Data Preprocessing**:
   - Convert 6 emotions to 2 sentiment classes
   - Remove punctuation
   - Tokenize text
   - Remove stopwords
   - Remove noisy characters
   - Apply stemming
5. **Feature Engineering** — TF-IDF Vectorization
6. **Train-Test Split** — 80-20 split
7. **Model Training** — Fit Logistic Regression
8. **Evaluation** — Calculate accuracy (93%)
9. **Save Models** — Pickle model and vectorizer for deployment

---

## 🚀 How to Run

```bash
jupyter notebook notebooks/nlp_project.ipynb
```

Or use JupyterLab:

```bash
jupyter lab
```

---

## 📌 Notes

- Replace the hardcoded data path with your local path or use relative paths
- Ensure you have downloaded NLTK data (punkt, stopwords)
- The notebook generates `model.pkl` and `vectorizer.pkl` in the project root
- It's recommended to modularize reusable code into `src/preprocessing.py`

---

## 📚 Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [Jupyter Documentation](https://jupyter.org/)
