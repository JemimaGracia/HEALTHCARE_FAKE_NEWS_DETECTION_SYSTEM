# ============================================================
# PROFESSIONAL KNOWLEDGE GRAPH VISUALIZATION
# HIGH-QUALITY GRAPH FOR RESEARCH / THESIS
# ============================================================

import networkx as nx
import matplotlib.pyplot as plt
import os

# ============================================================
# CREATE GRAPH FOLDER
# ============================================================

GRAPH_DIR = "graphs"

if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

# ============================================================
# CREATE KNOWLEDGE GRAPH
# ============================================================

G = nx.Graph()

# ============================================================
# NODES
# ============================================================

medical_nodes = [
    "COVID-19",
    "Virus",
    "Pandemic",
    "Vaccine",
    "Immunity",
    "WHO",
    "CDC",
    "NIH",
    "Mask",
    "Infection",
    "Antibiotic",
    "Health",
    "Medicine",
    "Doctor",
    "Hospital",
    "Symptoms",
    "Cancer",
    "Diabetes",
    "Bleach"
]

# Add nodes
for node in medical_nodes:
    G.add_node(node)

# ============================================================
# EDGES
# ============================================================

support_edges = [
    ("COVID-19", "Virus"),
    ("COVID-19", "Pandemic"),
    ("Mask", "COVID-19"),
    ("Vaccine", "Immunity"),
    ("Vaccine", "WHO"),
    ("Vaccine", "CDC"),
    ("Vaccine", "NIH"),
    ("Antibiotic", "Infection"),
    ("Medicine", "Health"),
    ("Doctor", "Hospital"),
    ("Symptoms", "Health"),
    ("Cancer", "Medicine"),
    ("Diabetes", "Medicine")
]

contradict_edges = [
    ("Bleach", "COVID-19"),
    ("Bleach", "Health"),
    ("Bleach", "Infection")
]

# Add support edges
for u, v in support_edges:
    G.add_edge(u, v, relation="Supports")

# Add contradiction edges
for u, v in contradict_edges:
    G.add_edge(u, v, relation="Contradicts")

# ============================================================
# PROFESSIONAL LAYOUT
# ============================================================

plt.figure(figsize=(18, 14))

# Better node spacing
pos = nx.spring_layout(
    G,
    k=1.8,
    iterations=300,
    seed=42
)

# ============================================================
# NODE GROUPS
# ============================================================

main_nodes = [
    "COVID-19",
    "Vaccine",
    "Health"
]

danger_nodes = [
    "Bleach"
]

organization_nodes = [
    "WHO",
    "CDC",
    "NIH"
]

# Other nodes
other_nodes = list(
    set(G.nodes()) -
    set(main_nodes) -
    set(danger_nodes) -
    set(organization_nodes)
)

# ============================================================
# DRAW NODES
# ============================================================

# Main nodes
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=main_nodes,
    node_size=5000,
    node_color="#1f77b4",
    edgecolors="black",
    linewidths=2,
    alpha=0.95
)

# Organization nodes
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=organization_nodes,
    node_size=4200,
    node_color="#2ca02c",
    edgecolors="black",
    linewidths=2,
    alpha=0.95
)

# Danger nodes
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=danger_nodes,
    node_size=4500,
    node_color="#d62728",
    edgecolors="black",
    linewidths=2,
    alpha=0.95
)

# Other nodes
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=other_nodes,
    node_size=3500,
    node_color="#87ceeb",
    edgecolors="black",
    linewidths=1.8,
    alpha=0.92
)

# ============================================================
# DRAW EDGES
# ============================================================

support_only = [
    (u, v)
    for u, v, d in G.edges(data=True)
    if d["relation"] == "Supports"
]

contradict_only = [
    (u, v)
    for u, v, d in G.edges(data=True)
    if d["relation"] == "Contradicts"
]

# Support edges
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=support_only,
    width=3,
    edge_color="#2ca02c",
    style="solid",
    alpha=0.85
)

# Contradiction edges
nx.draw_networkx_edges(
    G,
    pos,
    edgelist=contradict_only,
    width=3,
    edge_color="#d62728",
    style="dashed",
    alpha=0.9
)

# ============================================================
# LABELS
# ============================================================

nx.draw_networkx_labels(
    G,
    pos,
    font_size=11,
    font_weight="bold",
    font_family="sans-serif"
)

# ============================================================
# EDGE LABELS
# ============================================================

edge_labels = nx.get_edge_attributes(G, "relation")

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=9,
    font_color="black",
    rotate=False,
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.7
    )
)

# ============================================================
# TITLE
# ============================================================

plt.title(
    "Medical Knowledge Graph for Health Fake News Verification",
    fontsize=22,
    fontweight="bold",
    pad=25
)

# ============================================================
# REMOVE AXIS
# ============================================================

plt.axis("off")

# ============================================================
# SAVE HIGH-QUALITY IMAGE
# ============================================================

plt.savefig(
    f"{GRAPH_DIR}/professional_knowledge_graph.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("Professional Knowledge Graph Saved Successfully!")
print("Saved as: graphs/professional_knowledge_graph.png")