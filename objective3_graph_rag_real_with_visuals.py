# ============================================================
# OBJECTIVE 3: REAL GRAPH-RAG WITH ADVANCED VISUALIZATIONS
# ============================================================

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns

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
# 2. ENTITY EXTRACTION
# ============================================================

MEDICAL_ENTITIES = [
    "covid", "vaccine", "virus", "infection", "immunity",
    "antibiotic", "cancer", "diabetes", "who", "cdc", "nih", "bleach"
]

def extract_entities(text):
    text = str(text).lower()
    return [e for e in MEDICAL_ENTITIES if e in text]

df["entities"] = df["claim_text"].apply(extract_entities)

# ============================================================
# 3. SEMANTIC MEDICAL KNOWLEDGE GRAPH
# ============================================================

G = nx.Graph()

for e in MEDICAL_ENTITIES:
    G.add_node(e)

SEMANTIC_EDGES = [
    ("covid", "virus", "supports"),
    ("vaccine", "immunity", "supports"),
    ("vaccine", "who", "supports"),
    ("vaccine", "cdc", "supports"),
    ("vaccine", "nih", "supports"),
    ("antibiotic", "infection", "supports"),
    ("bleach", "covid", "contradicts"),
    ("bleach", "infection", "contradicts"),
]

for u, v, r in SEMANTIC_EDGES:
    G.add_edge(u, v, relation=r)

# ============================================================
# 4. MULTI-HOP EVIDENCE RETRIEVAL
# ============================================================

def retrieve_evidence_nodes(entities, hops=2):
    ev = set()
    for ent in entities:
        if ent in G:
            neighbors = nx.single_source_shortest_path_length(G, ent, cutoff=hops)
            ev.update(neighbors.keys())
    return list(ev)

df["evidence_nodes"] = df["entities"].apply(retrieve_evidence_nodes)

# ============================================================
# 5. GRAPH-RAG DECISION LOGIC
# ============================================================

def evidence_score(nodes):
    support, contradict = 0, 0
    for u, v, d in G.edges(data=True):
        if u in nodes or v in nodes:
            if d["relation"] == "supports":
                support += 1
            elif d["relation"] == "contradicts":
                contradict += 1
    return support, contradict

def graph_rag_decision(nodes):
    if not nodes:
        return "Insufficient Evidence"
    s, c = evidence_score(nodes)
    if s > c:
        return "Supported"
    elif c > s:
        return "Refuted"
    return "Insufficient Evidence"

df["graph_rag"] = df["evidence_nodes"].apply(graph_rag_decision)

# ============================================================
# 6. METRICS
# ============================================================

valid = df["graph_rag"] != "Insufficient Evidence"
accuracy = accuracy_score(df.loc[valid, "true_verdict"],
                          df.loc[valid, "graph_rag"])

coverage = valid.mean()

print("\nGraph-RAG Accuracy:", accuracy)
print("Graph-RAG Coverage:", coverage)

# ============================================================
# ================== VISUALIZATIONS ==========================
# ============================================================

# ---------------- GRAPH 1: Verdict Distribution ----------------
plt.figure()
df["graph_rag"].value_counts().plot(kind="bar")
plt.title("Graph-RAG Verdict Distribution")
plt.ylabel("Count")
plt.show()

# ---------------- GRAPH 2: Evidence Size vs Correctness ---------
df["evidence_size"] = df["evidence_nodes"].apply(len)
df["correct"] = (df["graph_rag"] == df["true_verdict"])

plt.figure()
sns.boxplot(x=df["correct"], y=df["evidence_size"])
plt.title("Evidence Size vs Prediction Correctness")
plt.xlabel("Correct Prediction")
plt.ylabel("Evidence Node Count")
plt.show()

# ---------------- GRAPH 3: Coverage vs Accuracy -----------------
thresholds = range(1, 6)
accs, covs = [], []

for t in thresholds:
    subset = df[df["evidence_size"] >= t]
    if len(subset) == 0:
        continue
    valid = subset["graph_rag"] != "Insufficient Evidence"
    accs.append(accuracy_score(
        subset.loc[valid, "true_verdict"],
        subset.loc[valid, "graph_rag"]
    ))
    covs.append(len(subset) / len(df))

plt.figure()
plt.plot(covs, accs, marker="o")
plt.xlabel("Coverage")
plt.ylabel("Accuracy")
plt.title("Coverage vs Accuracy Trade-off")
plt.show()

# ---------------- GRAPH 4: KNOWLEDGE GRAPH VISUALIZATION --------
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, seed=42)

edge_colors = ["green" if G[u][v]["relation"] == "supports" else "red"
               for u, v in G.edges()]

nx.draw(G, pos, with_labels=True,
        node_color="lightblue",
        edge_color=edge_colors,
        node_size=1500,
        font_size=9)

plt.title("Medical Knowledge Graph (Green=Supports, Red=Contradicts)")
plt.show()

# ---------------- GRAPH 5: ENTITY FREQUENCY HEATMAP -------------
entity_counts = pd.Series(
    sum(df["entities"].tolist(), [])
).value_counts()

plt.figure(figsize=(8, 3))
sns.heatmap(
    entity_counts.values.reshape(1, -1),
    annot=True,
    fmt="d",
    xticklabels=entity_counts.index,
    yticklabels=["Frequency"],
    cmap="Blues"
)
plt.title("Medical Entity Frequency Heatmap")
plt.show()

# ============================================================
# END
# ============================================================
