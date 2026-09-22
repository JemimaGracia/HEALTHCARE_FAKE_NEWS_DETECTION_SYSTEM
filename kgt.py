# ============================================================
# TRUE NEWS KNOWLEDGE GRAPH
# PROFESSIONAL VISUALIZATION
# ============================================================

import networkx as nx
import matplotlib.pyplot as plt
import os

# ============================================================
# CREATE GRAPH DIRECTORY
# ============================================================

GRAPH_DIR = "graphs"

if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)

# ============================================================
# CREATE TRUE NEWS KNOWLEDGE GRAPH
# ============================================================

G_true = nx.Graph()

# ============================================================
# TRUE NEWS NODES
# ============================================================

true_news_nodes = [
    "COVID-19",
    "Vaccination",
    "WHO",
    "CDC",
    "NIH",
    "Mask Usage",
    "Public Health",
    "Immunity",
    "Hospital",
    "Doctors",
    "Medicine",
    "Symptoms",
    "Testing",
    "Treatment",
    "Research",
    "Clinical Trials",
    "Virus",
    "Pandemic",
    "Patient Care",
    "Healthcare"
]

# Add nodes
for node in true_news_nodes:
    G_true.add_node(node)

# ============================================================
# TRUE NEWS RELATIONSHIPS
# ============================================================

true_edges = [

    # COVID relationships
    ("COVID-19", "Virus"),
    ("COVID-19", "Pandemic"),
    ("COVID-19", "Symptoms"),
    ("COVID-19", "Testing"),

    # Vaccine relationships
    ("Vaccination", "Immunity"),
    ("Vaccination", "WHO"),
    ("Vaccination", "CDC"),
    ("Vaccination", "NIH"),
    ("Vaccination", "Clinical Trials"),

    # Health relationships
    ("Mask Usage", "Public Health"),
    ("Doctors", "Hospital"),
    ("Hospital", "Patient Care"),
    ("Medicine", "Treatment"),
    ("Treatment", "Healthcare"),

    # Research relationships
    ("Research", "Clinical Trials"),
    ("Research", "Medicine"),
    ("Research", "Healthcare"),

    # Symptoms relationships
    ("Symptoms", "Treatment"),
    ("Testing", "Public Health"),

    # Public health relationships
    ("Public Health", "Healthcare"),
    ("Healthcare", "Patient Care")
]

# Add edges
for u, v in true_edges:
    G_true.add_edge(u, v, relation="Verified")

# ============================================================
# CREATE PROFESSIONAL LAYOUT
# ============================================================

plt.figure(figsize=(20, 15))

# Better spacing
pos = nx.spring_layout(
    G_true,
    k=2.2,
    iterations=400,
    seed=42
)

# ============================================================
# NODE GROUPS
# ============================================================

main_nodes = [
    "COVID-19",
    "Vaccination",
    "Public Health"
]

organization_nodes = [
    "WHO",
    "CDC",
    "NIH"
]

research_nodes = [
    "Research",
    "Clinical Trials",
    "Testing"
]

health_nodes = list(
    set(G_true.nodes()) -
    set(main_nodes) -
    set(organization_nodes) -
    set(research_nodes)
)

# ============================================================
# DRAW MAIN NODES
# ============================================================

nx.draw_networkx_nodes(
    G_true,
    pos,
    nodelist=main_nodes,
    node_size=5500,
    node_color="#1f77b4",
    edgecolors="black",
    linewidths=2.5,
    alpha=0.95
)

# ============================================================
# DRAW ORGANIZATION NODES
# ============================================================

nx.draw_networkx_nodes(
    G_true,
    pos,
    nodelist=organization_nodes,
    node_size=4500,
    node_color="#2ca02c",
    edgecolors="black",
    linewidths=2,
    alpha=0.95
)

# ============================================================
# DRAW RESEARCH NODES
# ============================================================

nx.draw_networkx_nodes(
    G_true,
    pos,
    nodelist=research_nodes,
    node_size=4200,
    node_color="#9467bd",
    edgecolors="black",
    linewidths=2,
    alpha=0.95
)

# ============================================================
# DRAW OTHER HEALTH NODES
# ============================================================

nx.draw_networkx_nodes(
    G_true,
    pos,
    nodelist=health_nodes,
    node_size=3500,
    node_color="#87ceeb",
    edgecolors="black",
    linewidths=1.8,
    alpha=0.92
)

# ============================================================
# DRAW EDGES
# ============================================================

nx.draw_networkx_edges(
    G_true,
    pos,
    width=3,
    edge_color="#2ca02c",
    alpha=0.8
)

# ============================================================
# NODE LABELS
# ============================================================

nx.draw_networkx_labels(
    G_true,
    pos,
    font_size=11,
    font_weight="bold",
    font_family="sans-serif"
)

# ============================================================
# EDGE LABELS
# ============================================================

edge_labels = nx.get_edge_attributes(
    G_true,
    "relation"
)

nx.draw_networkx_edge_labels(
    G_true,
    pos,
    edge_labels=edge_labels,
    font_size=9,
    font_color="darkgreen",
    rotate=False,
    bbox=dict(
        facecolor="white",
        edgecolor="none",
        alpha=0.75
    )
)

# ============================================================
# TITLE
# ============================================================

plt.title(
    "True News Medical Knowledge Graph",
    fontsize=24,
    fontweight="bold",
    pad=30
)

# ============================================================
# REMOVE AXIS
# ============================================================

plt.axis("off")

# ============================================================
# SAVE HIGH-QUALITY IMAGE
# ============================================================

plt.savefig(
    f"{GRAPH_DIR}/true_news_knowledge_graph.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("True News Knowledge Graph Saved Successfully!")
print("Saved as:")
print("graphs/true_news_knowledge_graph.png")