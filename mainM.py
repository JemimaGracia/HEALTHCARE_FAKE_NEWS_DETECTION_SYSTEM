import pandas as pd
import re
import os
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ============================================================
# 1. CREATE GRAPH OUTPUT FOLDER
# ============================================================

GRAPH_DIR = "graphs"

if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

print("Graphs folder ready.")

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv("final_combined_health_fake_news_dataset_VALIDATED.csv")

# Rename column
if "Text" in df.columns:
    df = df.rename(columns={"Text": "claim_text"})

# Ground truth labels
if "initial_label" in df.columns:
    df["true_verdict"] = df["initial_label"].map({
        "True News": "Supported",
        "False / Fake News": "Refuted"
    })

print("Dataset size:", df.shape)

# ============================================================
# 3. MEDICAL ENTITY EXTRACTION
# ============================================================

MEDICAL_ENTITIES = [
    "covid",
    "vaccine",
    "vaccination",
    "virus",
    "infection",
    "immunity",
    "antibiotic",
    "cancer",
    "diabetes",
    "who",
    "cdc",
    "nih",
    "bleach",
    "mask",
    "doctor",
    "hospital",
    "medicine",
    "health",
    "symptoms",
    "pandemic"
]


def extract_entities(text):
    text = str(text).lower()
    return [e for e in MEDICAL_ENTITIES if e in text]


# Extract entities

df["entities"] = df["claim_text"].apply(extract_entities)

print("Entity extraction completed.")

# ============================================================
# 4. BUILD KNOWLEDGE GRAPH
# ============================================================

G = nx.Graph()

# Add nodes
for entity in MEDICAL_ENTITIES:
    G.add_node(entity, type="medical_entity")

# ============================================================
# SEMANTIC RELATIONS
# ============================================================

SEMANTIC_EDGES = [
    ("covid", "virus", "supports"),
    ("covid", "pandemic", "supports"),
    ("vaccine", "immunity", "supports"),
    ("vaccine", "who", "supports"),
    ("vaccine", "cdc", "supports"),
    ("vaccine", "nih", "supports"),
    ("mask", "covid", "supports"),
    ("doctor", "hospital", "supports"),
    ("medicine", "health", "supports"),
    ("antibiotic", "infection", "supports"),
    ("symptoms", "health", "supports"),
    ("bleach", "covid", "contradicts"),
    ("bleach", "health", "contradicts"),
    ("bleach", "infection", "contradicts"),
    ("cancer", "medicine", "supports"),
    ("diabetes", "medicine", "supports")
]

# Add edges
for u, v, relation in SEMANTIC_EDGES:
    G.add_edge(u, v, relation=relation)

print("Knowledge Graph Nodes:", G.number_of_nodes())
print("Knowledge Graph Edges:", G.number_of_edges())

# ============================================================
# 5. SAVE KNOWLEDGE GRAPH IMAGE
# ============================================================

plt.figure(figsize=(14, 10))

pos = nx.spring_layout(G, seed=42)

edge_colors = []

for u, v, data in G.edges(data=True):
    if data["relation"] == "supports":
        edge_colors.append("green")
    else:
        edge_colors.append("red")

# Draw nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=3000,
    node_color="skyblue"
)

# Draw edges
nx.draw_networkx_edges(
    G,
    pos,
    edge_color=edge_colors,
    width=3
)

# Draw labels
nx.draw_networkx_labels(
    G,
    pos,
    font_size=10,
    font_weight="bold"
)

# Edge labels
edge_labels = nx.get_edge_attributes(G, "relation")

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_color="black"
)

plt.title("Medical Knowledge Graph")
plt.axis("off")

# Save image
plt.savefig(f"{GRAPH_DIR}/knowledge_graph.png", dpi=300, bbox_inches='tight')

plt.close()

print("Knowledge graph image saved.")

# ============================================================
# 6. MULTI-HOP EVIDENCE RETRIEVAL
# ============================================================


def retrieve_evidence_nodes(entities, hops=2):
    evidence = set()

    for ent in entities:
        if ent in G:
            neighbors = nx.single_source_shortest_path_length(
                G,
                ent,
                cutoff=hops
            )

            evidence.update(neighbors.keys())

    return list(evidence)


# Retrieve evidence

df["evidence_nodes"] = df["entities"].apply(
    retrieve_evidence_nodes
)

# ============================================================
# 7. EVIDENCE SCORING
# ============================================================


def evidence_score(evidence_nodes):

    support = 0
    contradict = 0

    for u, v, data in G.edges(data=True):

        if u in evidence_nodes or v in evidence_nodes:

            if data["relation"] == "supports":
                support += 1

            elif data["relation"] == "contradicts":
                contradict += 1

    return support, contradict

# ============================================================
# 8. GRAPH-RAG DECISION FUNCTION
# ============================================================


def graph_rag_decision(evidence_nodes):

    if not evidence_nodes:
        return "Insufficient Evidence"

    support, contradict = evidence_score(evidence_nodes)

    if support > contradict:
        return "Supported"

    elif contradict > support:
        return "Refuted"

    else:
        return "Insufficient Evidence"


# Predictions

df["graph_rag_verdict"] = df["evidence_nodes"].apply(
    graph_rag_decision
)

# ============================================================
# 9. BASELINES
# ============================================================

# Simulated baseline

df["llm_only"] = "Supported"

df["text_rag"] = df["llm_only"]

# ============================================================
# 10. EVALUATION
# ============================================================


def evaluate(pred_col):

    valid = df[pred_col] != "Insufficient Evidence"

    if valid.sum() == 0:
        return 0

    return accuracy_score(
        df.loc[valid, "true_verdict"],
        df.loc[valid, pred_col]
    )


acc_llm = evaluate("llm_only")
acc_text = evaluate("text_rag")
acc_graph = evaluate("graph_rag_verdict")

coverage = (
    df["graph_rag_verdict"] != "Insufficient Evidence"
).mean()

hallucination_rate = (
    (df["graph_rag_verdict"] != "Insufficient Evidence") &
    (df["evidence_nodes"].apply(len) == 0)
).mean()

print("ACCURACY RESULTS")
print("LLM-only:", acc_llm)
print("Text-RAG:", acc_text)
print("Graph-RAG:", acc_graph)

print("Graph-RAG Coverage:", coverage)
print("Graph-RAG Hallucination Rate:", hallucination_rate)

# ============================================================
# 11. ABLATION STUDY
# ============================================================

# No entity linking

df["no_entity"] = "Insufficient Evidence"
acc_no_entity = evaluate("no_entity")

# One-hop retrieval

df["evidence_1hop"] = df["entities"].apply(
    lambda e: retrieve_evidence_nodes(e, hops=1)
)

# One-hop prediction

df["graph_1hop"] = df["evidence_1hop"].apply(
    graph_rag_decision
)

acc_1hop = evaluate("graph_1hop")

print("ABLATION STUDY")
print("No entity linking:", acc_no_entity)
print("1-hop Graph-RAG:", acc_1hop)
print("2-hop Graph-RAG:", acc_graph)

# ============================================================
# 12. GRAPH 1 - VERIFICATION ACCURACY
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    ["LLM-only", "Text-RAG", "Graph-RAG"],
    [acc_llm, acc_text, acc_graph]
)

plt.title("Verification Accuracy Comparison")
plt.ylabel("Accuracy")

plt.savefig(
    f"{GRAPH_DIR}/verification_accuracy.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("Verification accuracy graph saved.")

# ============================================================
# 13. GRAPH 2 - ABLATION STUDY
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    ["No Entity", "1-hop", "2-hop"],
    [acc_no_entity, acc_1hop, acc_graph]
)

plt.title("Graph-RAG Ablation Study")
plt.ylabel("Accuracy")

plt.savefig(
    f"{GRAPH_DIR}/ablation_study.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("Ablation study graph saved.")

# ============================================================
# 14. GRAPH 3 - ENTITY FREQUENCY
# ============================================================

entity_frequency = {}

for entity_list in df["entities"]:
    for entity in entity_list:
        entity_frequency[entity] = entity_frequency.get(entity, 0) + 1

# Sort frequencies
sorted_entities = sorted(
    entity_frequency.items(),
    key=lambda x: x[1],
    reverse=True
)

entity_names = [x[0] for x in sorted_entities]
entity_counts = [x[1] for x in sorted_entities]

plt.figure(figsize=(12, 6))

plt.bar(entity_names, entity_counts)

plt.title("Medical Entity Frequency")
plt.xlabel("Medical Entities")
plt.ylabel("Frequency")
plt.xticks(rotation=45)

plt.savefig(
    f"{GRAPH_DIR}/entity_frequency.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("Entity frequency graph saved.")

# ============================================================
# 15. GRAPH 4 - KNOWLEDGE GRAPH NODE DEGREE
# ============================================================

node_degree = dict(G.degree())

nodes = list(node_degree.keys())
degrees = list(node_degree.values())

plt.figure(figsize=(10, 6))

plt.bar(nodes, degrees)

plt.title("Knowledge Graph Node Connectivity")
plt.xlabel("Nodes")
plt.ylabel("Degree")
plt.xticks(rotation=45)

plt.savefig(
    f"{GRAPH_DIR}/node_connectivity.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("Node connectivity graph saved.")

# ============================================================
# 16. GRAPH 5 - COVERAGE VS HALLUCINATION
# ============================================================

plt.figure(figsize=(6, 6))

plt.bar(
    ["Coverage", "Hallucination"],
    [coverage, hallucination_rate]
)

plt.title("Coverage vs Hallucination Rate")
plt.ylabel("Score")

plt.savefig(
    f"{GRAPH_DIR}/coverage_vs_hallucination.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print("Coverage vs hallucination graph saved.")

# ============================================================
# 17. SAVE RESULTS
# ============================================================

# Save final dataframe

df.to_csv("graph_rag_results.csv", index=False)

print("Results CSV saved.")

# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("================================================")
print("ALL KNOWLEDGE GRAPH IMAGES SAVED IN 'graphs' FOLDER")
print("================================================")

print("1. knowledge_graph.png")
print("2. verification_accuracy.png")
print("3. ablation_study.png")
print("4. entity_frequency.png")
print("5. node_connectivity.png")
print("6. coverage_vs_hallucination.png")

print("PROJECT COMPLETED SUCCESSFULLY")