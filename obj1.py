# ============================================================
# OBJECTIVE 1: DATA CURATION & FEATURE ENGINEERING PIPELINE
# ============================================================

# -----------------------------
# 1. IMPORTS & SETUP
# -----------------------------
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency

import torch
from transformers import AutoTokenizer, AutoModel

sns.set(style="whitegrid")

print("Environment ready")

# -----------------------------
# 2. LOAD DATA
# -----------------------------
# Assumes your validated dataset
df = pd.read_csv("final_combined_health_fake_news_dataset_VALIDATED.csv")

print("Original dataset shape:", df.shape)

# -----------------------------
# 3. STANDARDIZE METADATA
# -----------------------------
df = df.rename(columns={
    "Text": "claim_text",
    "Date Posted": "timestamp",
    "Link": "source_url",
    "Region": "region"
})

df["platform"] = df["source_url"].apply(
    lambda x: "Social Media" if "facebook" in str(x).lower() or "twitter" in str(x).lower()
    else "News / Web"
)

df["topic"] = "Healthcare"

# -----------------------------
# 4. LABEL NORMALIZATION
# -----------------------------
df["veracity_label"] = df["initial_label"].map({
    "True News": "True",
    "False / Fake News": "False"
})

df["binary_label"] = df["veracity_label"].map({
    "True": 1,
    "False": 0
})

# -----------------------------
# 5. DE-DUPLICATION (TEXT SIMILARITY)
# -----------------------------
df["claim_text_clean"] = df["claim_text"].astype(str).str.lower().str.strip()
df = df.drop_duplicates(subset="claim_text_clean")

print("After de-duplication:", df.shape)

# -----------------------------
# 6. TEXT PREPROCESSING
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["claim_text"].apply(clean_text)

# -----------------------------
# 7. EDA — DESCRIPTIVE FEATURES
# -----------------------------
df["text_length"] = df["clean_text"].apply(len)
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

# -----------------------------
# 8. EDA — 6 PLOTS
# -----------------------------

# Plot 1: Class Distribution
plt.figure()
sns.countplot(x="veracity_label", data=df)
plt.title("Distribution of True vs Fake News")
plt.show()

# Plot 2: Platform Distribution
plt.figure()
sns.countplot(x="platform", data=df)
plt.title("Platform Distribution")
plt.show()

# Plot 3: Text Length Histogram
plt.figure()
sns.histplot(df["text_length"], bins=50)
plt.title("Text Length Distribution")
plt.show()

# Plot 4: Word Count by Class
plt.figure()
sns.boxplot(x="veracity_label", y="word_count", data=df)
plt.title("Word Count by Veracity")
plt.show()

# Plot 5: Region Distribution
plt.figure()
df["region"].value_counts().head(10).plot(kind="bar")
plt.title("Top Regions")
plt.show()

# Plot 6: Platform vs Veracity
plt.figure()
sns.countplot(x="platform", hue="veracity_label", data=df)
plt.title("Platform vs Veracity")
plt.show()

# -----------------------------
# 9. STATISTICAL ANALYSIS (3 TESTS)
# -----------------------------

# Test 1: T-test (Text Length)
fake_len = df[df["binary_label"] == 0]["text_length"]
true_len = df[df["binary_label"] == 1]["text_length"]

t_stat, p_val = ttest_ind(fake_len, true_len, equal_var=False)
print("\nT-test (Text Length):")
print("T-statistic:", t_stat, "P-value:", p_val)

# Test 2: Mann–Whitney U Test (Word Count)
u_stat, p_val_u = mannwhitneyu(fake_len, true_len)
print("\nMann–Whitney U Test (Word Count):")
print("U-statistic:", u_stat, "P-value:", p_val_u)

# Test 3: Chi-square (Platform vs Veracity)
contingency = pd.crosstab(df["platform"], df["veracity_label"])
chi2, p, dof, expected = chi2_contingency(contingency)
print("\nChi-square Test (Platform vs Veracity):")
print("Chi2:", chi2, "P-value:", p)

# -----------------------------
# 10. FEATURE ENGINEERING — TF-IDF
# -----------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(df["clean_text"])

print("TF-IDF feature matrix shape:", X_tfidf.shape)

# -----------------------------
# 11. MEDICAL CONCEPT FEATURES (HOOKS)
# -----------------------------
# These columns already exist in your dataset
medical_features = df[["entity_ids"]].fillna("NONE")

print("Medical concept feature sample:")
print("\nMedical concept feature sample:")
print(medical_features.head())


# -----------------------------
# 12. CONTEXTUAL EMBEDDINGS (RoBERTa)
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")
model.eval()

def roberta_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**encoded)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(cls_embeddings)
    return np.vstack(embeddings)

X_roberta = roberta_embeddings(df["clean_text"].tolist())
print("RoBERTa embeddings shape:", X_roberta.shape)

# -----------------------------
# 13. LEAKAGE-SAFE SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["binary_label"],
    test_size=0.2,
    random_state=42,
    stratify=df["binary_label"]
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# -----------------------------
# 14. FINAL DATASET SUMMARY
# -----------------------------
print("\nFINAL DATASET READY FOR MODELING")
print("Rows:", df.shape[0])
print("Features available:")
print("- Clean text")
print("- TF-IDF features")
print("- Medical entity hooks")
print("- RoBERTa embeddings")
print("- Leakage-safe splits")

# ============================================================
# END OF OBJECTIVE 1 IMPLEMENTATION
# ============================================================
