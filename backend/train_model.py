import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

# Import your custom preprocessing module
from preprocess import preprocess_text 

def top_keywords_for_prediction(claim_text, vectorizer_model, clf_model, top_n=3):
    """
    Extracts the most influential words driving the model's prediction,
    navigating through the CalibratedClassifierCV wrapper.
    """
    cleaned_text = preprocess_text(claim_text)
    
    # Extract the base SVM from the calibration wrapper
    if hasattr(clf_model, 'calibrated_classifiers_'):
        calibrated_clf = clf_model.calibrated_classifiers_[0]
        base_svm = getattr(calibrated_clf, 'estimator', getattr(calibrated_clf, 'base_estimator', None))
    else:
        base_svm = clf_model
        
    if not hasattr(base_svm, 'coef_'):
        return ["(No coefficients available)"]

    feature_names = np.array(vectorizer_model.get_feature_names_out())
    coefs = base_svm.coef_[0]
    
    vec = vectorizer_model.transform([cleaned_text])
    nonzero_indices = vec.nonzero()[1]
    
    if len(nonzero_indices) == 0:
        return ["(No recognized vocabulary)"]
        
    word_weights = {feature_names[i]: coefs[i] for i in nonzero_indices}
    sorted_words = sorted(word_weights.items(), key=lambda x: abs(x[1]), reverse=True)
    
    return [word for word, weight in sorted_words[:top_n]]

def main():
    print("=" * 50)
    print(" INITIATING MARK_I TRAINING PIPELINE")
    print("=" * 50)

    # 1. Load the localized dataset (91 Fake / 30 Real)
    data_path = '../data/claims.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run build_dataset.py first.")
    
    df = pd.read_csv(data_path)
    
    # Ensure all text is preprocessed before training
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # 2. SPLIT FIRST (Zero Data Leakage)
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], 
        df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )

    # 3. VECTORIZE (Fit on Train Only)
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. TRAIN CALIBRATED MODEL (For real confidence scores)
    print("Training Calibrated LinearSVC...")
    base_svm = LinearSVC(class_weight="balanced", random_state=42, max_iter=2000)
    model = CalibratedClassifierCV(base_svm, method='sigmoid', cv=5)
    model.fit(X_train_vec, y_train)

    # 5. EVALUATE
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))

    # 6. SAVE ARTIFACTS
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/model.pkl')
    joblib.dump(vectorizer, '../models/vectorizer.pkl')
    print("\n Artifacts saved to /models (model.pkl, vectorizer.pkl)")

    # =========================================================
    # 7. SANITY CHECK MODULE
    # =========================================================
    print("\n" + "=" * 50)
    print(" SANITY CHECK: TESTING PRODUCTION PIPELINE")
    print("=" * 50)

    sanity_tests = [
        ("Paracetamol is commonly used to reduce fever.", 1),
        ("The CDSCO regulates pharmaceutical imports in India.", 1),
        ("Drinking turmeric water cures cancer instantly in 24 hours.", 0),
        ("Chewing raw garlic cures COVID-19.", 0)
    ]

    all_passed = True
    for text, expected in sanity_tests:
        cleaned = preprocess_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        
        probs = model.predict_proba(vec)[0]
        confidence = probs[pred] * 100
        keywords = top_keywords_for_prediction(text, vectorizer, model)
        
        status = "PASS" if pred == expected else "FAIL"
        if pred != expected:
            all_passed = False
            
        exp_str = "Real" if expected == 1 else "Fake"
        got_str = "Real" if pred == 1 else "Fake"
        
        print(f"{status} | Expected: {exp_str} | Got: {got_str} ({confidence:.1f}% Conf)")
        print(f"      Claim: '{text}'")
        print(f"      Flags: {', '.join(keywords)}\n")

    if all_passed:
        print("SUCCESS: Model passed all sanity checks! Ready for backend API.")
    else:
        print("WARNING: Model failed one or more checks.")
    print("=" * 50)

if __name__ == "__main__":
    main()