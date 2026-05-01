# Data Directory

This directory is reserved for data files used in the sentiment analysis project.

## ⚠️ Important Notes

1. **Do NOT commit large raw datasets** to GitHub (use `.gitignore`)
2. **Do NOT include private or sensitive data**
3. Use this directory for:
   - Small example datasets for testing
   - Data processing scripts
   - Processed/cleaned data (if small)

## Data Sources

The project uses a sentiment dataset with 16,000 training samples containing:
- Text sentences
- Emotion labels: joy, sadness, anger, fear, love, surprise

**Preprocessing**: 6 emotions are grouped into 2 classes:
- **Positive**: joy, love, surprise
- **Negative**: anger, fear, sadness

## File Format

Original data format:
```
text;sentiment
"sentence here";emotion
```

After preprocessing:
```
text,sentiment
"cleaned sentence",1  # 1 = Positive, 0 = Negative
```

## Usage

When loading data:

```python
import pandas as pd

# Load data
df = pd.read_csv('data/train.csv', delimiter=';', header=None, names=['text', 'sentiment'])
```

## 📊 Data Statistics

- Total samples: 16,000
- Train-Test split: 80-20
- Positive samples: 7,238 (45%)
- Negative samples: 8,762 (55%)
- Languages: English

---

For more details, see the main [README.md](../README.md).
