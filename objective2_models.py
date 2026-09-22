# ============================================================
# OBJECTIVE 2: MULTI-LEVEL DETECTION MODELS
# VS CODE / TERMINAL READY
# ============================================================

# -----------------------------
# 1. IMPORTS
# -----------------------------
import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sns.set(style="whitegrid")

print("Objective 2: Environment ready")

# -----------------------------
# 2. LOAD DATA
# -----------------------------
df = pd.read_csv("final_combined_health_fake_news_dataset_VALIDATED.csv")

df = df.rename(columns={
    "Text": "text",
    "Date Posted": "timestamp",
    "Region": "region"
})

df["binary_label"] = df["initial_label"].map({
    "True News": 1,
    "False / Fake News": 0
})

# Clean text
def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-z\s]", "", t)
    return t.strip()

df["clean_text"] = df["text"].apply(clean_text)

# -----------------------------
# 3. TRAIN / TEST SPLIT (LEAKAGE SAFE)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["binary_label"],
    test_size=0.2,
    random_state=42,
    stratify=df["binary_label"]
)

print("Train size:", len(X_train), "Test size:", len(X_test))

# -----------------------------
# 4. FEATURE ENGINEERING (TF-IDF)
# -----------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------------
# 5. CLASSICAL ML BASELINES
# -----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "NaiveBayes": MultinomialNB(),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1": f1
    })

    print(f"\n{name} Results")
    print(classification_report(y_test, preds))

# -----------------------------
# 6. CALIBRATION ANALYSIS (LOGISTIC REGRESSION)
# -----------------------------
lr = models["LogisticRegression"]
probs = lr.predict_proba(X_test_tfidf)[:, 1]

brier = brier_score_loss(y_test, probs)
print("\nCalibration (Brier Score - Logistic Regression):", brier)

prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)

plt.figure()
plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("Calibration Curve (Logistic Regression)")
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.show()

# -----------------------------
# 7. TRANSFORMER MODEL (RoBERTa)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

model.eval()
device = torch.device("cpu")
model.to(device)

def roberta_predict(texts):
    preds = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        with torch.no_grad():
            out = model(**encoded)
        preds.append(out.logits.argmax(dim=1).item())
    return preds

roberta_preds = roberta_predict(X_test.tolist())

results.append({
    "Model": "RoBERTa",
    "Accuracy": accuracy_score(y_test, roberta_preds),
    "F1": f1_score(y_test, roberta_preds)
})

print("\nRoBERTa Classification Report")
print(classification_report(y_test, roberta_preds))

# -----------------------------
# 8. HYBRID MODEL (TF-IDF + RoBERTa CONFIDENCE)
# -----------------------------
hybrid_preds = []

for tfidf_row, text in zip(X_test_tfidf, X_test):
    lr_pred = lr.predict(tfidf_row)[0]
    rob_pred = roberta_predict([text])[0]

    # Hybrid rule: agree → confident, else fallback to LR
    hybrid_preds.append(rob_pred if lr_pred == rob_pred else lr_pred)

results.append({
    "Model": "Hybrid (LR + RoBERTa)",
    "Accuracy": accuracy_score(y_test, hybrid_preds),
    "F1": f1_score(y_test, hybrid_preds)
})

# -----------------------------
# 9. ROBUSTNESS TESTS
# -----------------------------
print("\nRobustness Test: Platform Split")

for platform in df["region"].dropna().unique()[:3]:
    subset = df[df["region"] == platform]
    if len(subset) < 200:
        continue

    X_sub = tfidf.transform(subset["clean_text"])
    y_sub = subset["binary_label"]

    preds = lr.predict(X_sub)
    print(f"{platform} Accuracy:", accuracy_score(y_sub, preds))

# -----------------------------
# 10. ERROR ANALYSIS
# -----------------------------
error_df = pd.DataFrame({
    "text": X_test,
    "true": y_test,
    "pred": lr.predict(X_test_tfidf)
})

false_pos = error_df[(error_df.true == 0) & (error_df.pred == 1)]
false_neg = error_df[(error_df.true == 1) & (error_df.pred == 0)]

print("\nFalse Positives:", len(false_pos))
print("False Negatives:", len(false_neg))

# -----------------------------
# 11. ABLATION STUDY
# -----------------------------
print("\nAblation Study: Reduced TF-IDF")

tfidf_small = TfidfVectorizer(max_features=1000)
X_train_small = tfidf_small.fit_transform(X_train)
X_test_small = tfidf_small.transform(X_test)

lr_small = LogisticRegression(max_iter=1000)
lr_small.fit(X_train_small, y_train)

preds_small = lr_small.predict(X_test_small)
print("Reduced-feature Accuracy:", accuracy_score(y_test, preds_small))

# -----------------------------
# 12. FINAL RESULTS SUMMARY
# -----------------------------
results_df = pd.DataFrame(results)
print("\nFINAL MODEL COMPARISON")
print(results_df)

plt.figure()
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.xticks(rotation=30)
plt.title("Model Accuracy Comparison")
plt.show()

# ============================================================
# END OF OBJECTIVE 2
# ============================================================
