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
│   ├── preprocess.py     # Centralized text cleaning and tokenization module
│   └── train_model.py    # Leakage-free training and evaluation pipeline
├── data/
│   └── claims.csv        # Compiled master training dataset
├── models/
│   ├── model.pkl         # Serialized Linear SVM classifier
│   ├── vectorizer.pkl    # Fitted TF-IDF vectorizer artifact
│   └── metrics.json      # Evaluation performance report
├── frontend/             # Web interface components (UI)
├── research_paper/       # Academic documentation & reports
├── .gitignore            # Git exclusion rules
├── build_dataset.py      # Dataset ingestion and consolidation script
├── Drug_Alert_March2023.csv                # CDSCO regulatory safety alerts
├── Spurious_Drugs_Puducherry_Nov2025.csv   # Counterfeit/spurious notices
└── README.md             # Project documentation
