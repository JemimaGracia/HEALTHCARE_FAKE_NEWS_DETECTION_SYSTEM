# ============================================================
# OBJECTIVE 3 — OPTIONAL EXTENSIONS (ADVANCED)
# ============================================================

import pandas as pd
import requests
import spacy
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt

from transformers import pipeline

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("final_combined_health_fake_news_dataset_VALIDATED.csv")

df = df.rename(columns={"Text": "claim_text"})
df["true_verdict"] = df["initial_label"].map({
    "True News": "Supported",
    "False / Fake News": "Refuted"
})

print("Dataset loaded:", df.shape)

# ============================================================
# 2. NEURAL ENTITY LINKING (scispaCy)
# ============================================================
# Requires:
# pip install scispacy spacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

try:
    nlp = spacy.load("en_core_sci_sm")
except:
    nlp = None
    print("scispaCy model not installed. Skipping neural entity linking.")

def neural_entity_linking(text):
    if not nlp:
        return []
    doc = nlp(text)
    return list(set(ent.text.lower() for ent in doc.ents))

df["neural_entities"] = df["claim_text"].astype(str).apply(neural_entity_linking)

print("Sample neural entities:")
print(df["neural_entities"].head())

# ============================================================
# 3. REAL EXTERNAL MEDICAL KG (WIKIDATA)
# ============================================================

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

def query_wikidata(entity):
    query = f"""
    SELECT ?itemLabel WHERE {{
      ?item rdfs:label "{entity}"@en .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 5
    """
    headers = {"Accept": "application/sparql+json"}
    r = requests.get(
        WIKIDATA_ENDPOINT,
        params={"query": query, "format": "json"},
        headers=headers,
        timeout=10
    )
    if r.status_code != 200:
        return []

    results = r.json()["results"]["bindings"]
    return [res["itemLabel"]["value"] for res in results]

# Build external KG
external_KG = nx.Graph()

for entities in df["neural_entities"].head(50):  # limit for demo
    for ent in entities:
        nodes = query_wikidata(ent)
        for node in nodes:
            external_KG.add_edge(ent, node, relation="related")

print("External KG nodes:", external_KG.number_of_nodes())

# ============================================================
# 4. GRAPH-RAG WITH EXTERNAL KG
# ============================================================

def retrieve_external_evidence(entities, hops=2):
    evidence = set()
    for ent in entities:
        if ent in external_KG:
            neighbors = nx.single_source_shortest_path_length(
                external_KG, ent, cutoff=hops
            )
            evidence.update(neighbors.keys())
    return list(evidence)

df["external_evidence"] = df["neural_entities"].apply(retrieve_external_evidence)

# ============================================================
# 5. GENERATIVE RATIONALES (LLM-BASED)
# ============================================================
# NOTE: This is for explanation only, NOT decision-making

try:
    explainer = pipeline(
        "text-generation",
        model="distilgpt2",
        max_new_tokens=80
    )
except:
    explainer = None
    print("LLM explainer not available.")

def generate_rationale(claim, evidence):
    if not explainer or not evidence:
        return "No sufficient evidence available for explanation."
    prompt = (
        f"Claim: {claim}\n"
        f"Evidence: {', '.join(evidence)}\n"
        f"Explanation:"
    )
    return explainer(prompt)[0]["generated_text"]

df["llm_rationale"] = df.apply(
    lambda r: generate_rationale(
        r["claim_text"], r["external_evidence"]
    ),
    axis=1
)

print("Sample rationale:")
print(df["llm_rationale"].head(1).values[0])

# ============================================================
# 6. TEMPORAL ROBUSTNESS ANALYSIS
# ============================================================

# Convert date
df["Date Posted"] = pd.to_datetime(
    df["Date Posted"], errors="coerce"
)

df["year"] = df["Date Posted"].dt.year

temporal_results = {}

for year in sorted(df["year"].dropna().unique()):
    subset = df[df["year"] == year]
    coverage = (subset["external_evidence"].apply(len) > 0).mean()
    temporal_results[year] = coverage

print("Temporal KG coverage:")
for y, c in temporal_results.items():
    print(y, "→", round(c, 3))

# Plot temporal robustness
plt.figure()
plt.plot(list(temporal_results.keys()), list(temporal_results.values()), marker="o")
plt.title("Temporal Robustness of Graph-RAG (KG Coverage)")
plt.xlabel("Year")
plt.ylabel("Evidence Coverage")
plt.show()

# ============================================================
# END OF OPTIONAL EXTENSIONS
# ============================================================
