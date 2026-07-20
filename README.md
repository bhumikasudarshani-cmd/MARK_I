# MARK_I: India-Focused Drug & Health Claim Verification Pipeline

**MARK_I** is a lightweight, accessible, India-focused NLP tool designed to help the general public and healthcare workers quickly verify unverified drug and health claims originating from informal text sources like social media and messaging apps.

## 🚀 Key Features

*   **Robust Text Preprocessing:** Standardized pipeline utilizing NLTK for lowercase normalization, punctuation/number stripping, tokenization, stopword removal, and WordNet lemmatization.
*   **Leakage-Free Training Methodology:** Enforces strict train/test splitting *before* TF-IDF vectorization and feature scaling to ensure evaluation metrics reflect true generalization to unseen data.
*   **Class-Weighted Optimization:** Uses a Linear Support Vector Classifier with `class_weight="balanced"` to effectively handle class imbalances without relying on synthetic data distortion (like SMOTE on sparse text matrices).
*   **Explainable Predictions:** Features keyword influence extraction (`top_keywords_for_prediction`) to surface the specific words driving the model's classification decision.

---

## 📂 Project Structure

```text
MARK_I/
├── backend/
│   ├── app.py                 # Flask entry point — thin, just routes
│   ├── inference.py           # Loads model/vectorizer once, exposes predict()
│   ├── preprocess.py          # Already have this 
│   ├── train_model.py         # Already have this 
│   ├── config.py              # Env-based config (debug, port, model paths)
│   ├── requirements.txt
│   └── tests/
│       └── test_api.py
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
├── data/
│   └── claims.csv
└── frontend/
    ├── index.html
    ├── app.js
    └── style.css
