# ============================================================
# OBJECTIVE 3: KNOWLEDGE-GRAPH–GROUNDED CLAIM VERIFICATION
# ============================================================

import pandas as pd
import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("final_combined_health_fake_news_dataset_VALIDATED.csv")

df = df.rename(columns={"Text": "claim_text"})
df["true_verdict"] = df["initial_label"].map({
    "True News": "Supported",
    "False / Fake News": "Refuted"
})

print("Dataset size:", df.shape)

# ============================================================
# 2. SIMPLE MEDICAL ENTITY EXTRACTION (RULE-BASED)
# ============================================================

MEDICAL_ENTITIES = [
    "covid", "vaccine", "vaccination", "cancer", "diabetes",
    "virus", "infection", "immunity", "who", "cdc", "antibiotic"
]

def extract_entities(text):
    text = text.lower()
    return [e for e in MEDICAL_ENTITIES if e in text]

df["entities"] = df["claim_text"].apply(extract_entities)

# ============================================================
# 3. CONSTRUCT MEDICAL KNOWLEDGE GRAPH
# ============================================================

G = nx.Graph()

# Nodes
for e in MEDICAL_ENTITIES:
    G.add_node(e, type="medical_entity")

# Trusted sources
sources = ["WHO", "CDC", "NIH"]
for s in sources:
    G.add_node(s, type="authority")

# Relations
edges = [
    ("covid", "virus"), ("covid", "infection"),
    ("vaccine", "immunity"), ("vaccine", "WHO"),
    ("vaccine", "CDC"), ("antibiotic", "infection"),
    ("cancer", "NIH"), ("diabetes", "NIH")
]

G.add_edges_from(edges)

print("Knowledge Graph nodes:", G.number_of_nodes())
print("Knowledge Graph edges:", G.number_of_edges())

# ============================================================
# 4. MULTI-HOP EVIDENCE RETRIEVAL
# ============================================================

def retrieve_graph_evidence(entities, hops=2):
    evidence_nodes = set()
    for ent in entities:
        if ent in G:
            neighbors = nx.single_source_shortest_path_length(G, ent, cutoff=hops)
            evidence_nodes.update(neighbors.keys())
    return list(evidence_nodes)

df["kg_evidence_nodes"] = df["entities"].apply(retrieve_graph_evidence)

# ============================================================
# 5. TEXT-RAG BASELINE (TF-IDF RETRIEVAL)
# ============================================================

tfidf = TfidfVectorizer(max_features=3000)
X_tfidf = tfidf.fit_transform(df["claim_text"])

def text_rag_evidence(query):
    q_vec = tfidf.transform([query])
    sims = cosine_similarity(q_vec, X_tfidf)[0]
    idx = sims.argmax()
    return df.iloc[idx]["claim_text"]

# ============================================================
# 6. LOAD LLM (ROBERTA – CPU SAFE)
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)
model.eval()
model.to("cpu")

# ============================================================
# 7. VERIFICATION FUNCTIONS
# ============================================================

def llm_only_decision(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=1).item()
    return "Supported" if pred == 1 else "Refuted"

def graph_rag_decision(claim, evidence_nodes):
    if not evidence_nodes:
        return "Insufficient Evidence", "No linked KG entities"

    evidence_text = " ".join(evidence_nodes)
    prompt = claim + " " + evidence_text

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)

    pred = outputs.logits.argmax(dim=1).item()
    verdict = "Supported" if pred == 1 else "Refuted"
    return verdict, evidence_text

# ============================================================
# 8. RUN BASELINES & GRAPH-RAG
# ============================================================

results = []

for _, row in df.iterrows():
    claim = row["claim_text"]

    llm_pred = llm_only_decision(claim)
    text_rag_pred = llm_only_decision(text_rag_evidence(claim))
    graph_pred, evidence = graph_rag_decision(
        claim,
        row["kg_evidence_nodes"]
    )

    results.append({
        "true": row["true_verdict"],
        "llm_only": llm_pred,
        "text_rag": text_rag_pred,
        "graph_rag": graph_pred,
        "has_evidence": len(row["kg_evidence_nodes"]) > 0
    })

results_df = pd.DataFrame(results)

# ============================================================
# 9. EVALUATION METRICS
# ============================================================

def accuracy(col):
    valid = results_df[col] != "Insufficient Evidence"
    return accuracy_score(
        results_df.loc[valid, "true"],
        results_df.loc[valid, col]
    )

acc_llm = accuracy("llm_only")
acc_text = accuracy("text_rag")
acc_graph = accuracy("graph_rag")

hallucination_rate = (
    (results_df["graph_rag"] != "Insufficient Evidence") &
    (~results_df["has_evidence"])
).mean()

print("\nACCURACY COMPARISON")
print("LLM-only:", acc_llm)
print("Text-RAG:", acc_text)
print("Graph-RAG:", acc_graph)

print("\nHallucination Rate (Graph-RAG):", hallucination_rate)

# ============================================================
# 10. ABLATION STUDIES
# ============================================================

# Without entity linking
results_df["no_entity_graph"] = results_df["llm_only"]
acc_no_entity = accuracy("no_entity_graph")

# Without multi-hop (1-hop only)
df["kg_1hop"] = df["entities"].apply(lambda e: retrieve_graph_evidence(e, hops=1))
one_hop_results = []

for _, row in df.iterrows():
    pred, _ = graph_rag_decision(row["claim_text"], row["kg_1hop"])
    one_hop_results.append(pred)

results_df["graph_1hop"] = one_hop_results
acc_1hop = accuracy("graph_1hop")

print("\nABLATION RESULTS")
print("No entity linking:", acc_no_entity)
print("1-hop Graph-RAG:", acc_1hop)

# ============================================================
# 11. PLOTS
# ============================================================

plt.figure()
plt.bar(
    ["LLM-only", "Text-RAG", "Graph-RAG"],
    [acc_llm, acc_text, acc_graph]
)
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

plt.figure()
plt.bar(
    ["No Entity", "1-hop", "2-hop"],
    [acc_no_entity, acc_1hop, acc_graph]
)
plt.title("Graph-RAG Ablation Study")
plt.ylabel("Accuracy")
plt.show()

# ============================================================
# END OF OBJECTIVE 3
# ============================================================
