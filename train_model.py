import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocess import preprocess_batch

RANDOM_STATE = 42


def load_dataset(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required column(s): {missing}")

    df = df.dropna(subset=[text_col, label_col]).copy()
    
    # IMPROVEMENT: Robust label parsing (handles text labels or numeric strings gracefully)
    if df[label_col].dtype == object:
        mapping = {"fake": 0, "real": 1, "false": 0, "true": 1, "0": 0, "1": 1}
        df[label_col] = df[label_col].astype(str).str.lower().map(mapping)
        df = df.dropna(subset=[label_col])
        
    df[label_col] = df[label_col].astype(int)
    return df


def build_train_test_vectors(df, text_col, label_col, max_features, use_smote):
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df[label_col],
    )

    cleaned_train = preprocess_batch(X_text_train)
    cleaned_test = preprocess_batch(X_text_test)

    # IMPROVEMENT: Added ngram_range=(1, 2) to capture word pairs (e.g., "high blood pressure")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(cleaned_train)
    X_test = vectorizer.transform(cleaned_test)

    if use_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return vectorizer, X_train, X_test, y_train, y_test


def train_and_evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(y_test, y_pred, target_names=["Fake", "Real"], zero_division=0)

    print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
    print(f"Accuracy: {acc * 100:.2f}%   F1: {f1:.3f}")
    print(report)

    return {"model": model, "accuracy": acc, "f1": f1}


def top_keywords_for_prediction(text, vectorizer, model, top_n=5):
    cleaned = preprocess_batch([text])[0]
    vec = vectorizer.transform([cleaned])
    feature_names = np.array(vectorizer.get_feature_names_out())
    nonzero_idx = vec.nonzero()[1]

    if not len(nonzero_idx):
        return []

    if hasattr(model, "coef_"):
        weights = model.coef_[0][nonzero_idx]
    elif hasattr(model, "feature_log_prob_"):
        weights = (model.feature_log_prob_[1] - model.feature_log_prob_[0])[nonzero_idx]
    else:
        return []

    order = np.argsort(-np.abs(weights))[:top_n]
    words = feature_names[nonzero_idx][order]
    scores = weights[order]
    return list(zip(words, scores.tolist()))


def main():
    parser = argparse.ArgumentParser(description="Train the health-claim classifier.")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE on training split only")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    df = load_dataset(args.data, args.text_col, args.label_col)
    
    # IMPROVEMENT: Filter out rows that clean down to empty strings
    df['temp_cleaned'] = preprocess_batch(df[args.text_col])
    df = df[df['temp_cleaned'].str.strip() != ''].copy()
    df = df.drop(columns=['temp_cleaned'])

    print(f"Loaded {len(df)} valid rows after filtering. Class balance:\n{df[args.label_col].value_counts()}")

    vectorizer, X_train, X_test, y_train, y_test = build_train_test_vectors(
        df, args.text_col, args.label_col, args.max_features, args.use_smote
    )

    results = {}
    results["naive_bayes"] = train_and_evaluate(
        "Naive Bayes", MultinomialNB(), X_train, y_train, X_test, y_test
    )
    results["svm"] = train_and_evaluate(
        "Linear SVM",
        LinearSVC(random_state=RANDOM_STATE, class_weight="balanced"),
        X_train, y_train, X_test, y_test,
    )

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.3f})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out_dir / "model.pkl")
    joblib.dump(vectorizer, out_dir / "vectorizer.pkl")

    metrics = {k: {"accuracy": v["accuracy"], "f1": v["f1"]} for k, v in results.items()}
    metrics["best_model"] = best_name
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved: {out_dir/'model.pkl'}, {out_dir/'vectorizer.pkl'}, {out_dir/'metrics.json'}")

    sample = "Drinking turmeric water cures cancer instantly"
    pred = best_model.predict(vectorizer.transform(preprocess_batch([sample])))[0]
    keywords = top_keywords_for_prediction(sample, vectorizer, best_model)
    print(f"\nSample check -> '{sample}'")
    print(f"Prediction: {'Real' if pred == 1 else 'Fake'}")
    print(f"Top influencing words: {keywords}")


if __name__ == "__main__":
    main()