# CineGraph Network Intelligence

**Media Graph Analytics | Semantic Retrieval | Hybrid Recommendations | Review NLP | Streamlit Product Prototype**

CineGraph Network Intelligence turns the CineGraph TMDB dataset into a graph-aware media intelligence product instead of a flat catalog analysis.

It now has two clear layers:

- a reproducible Python pipeline that builds processed node and edge exports plus analytical summary tables;
- a Streamlit app that uses those exports to demonstrate semantic search, graph-aware recommendations, and explainable ranking.

## Live app

**[Open CineGraph AI Explorer](https://cinegraph-network-intelligence-lcsr.streamlit.app/)**

## What this project demonstrates

- graph-first catalog modeling across movies, TV shows, people, and orphan recommendation nodes;
- reproducible processed exports for downstream analytics and app serving;
- hybrid recommendation logic combining semantic, graph, quality, and business signals;
- explainable recommendation output instead of black-box ranking only;
- review-text mining for interpretable audience signals;
- production-minded project structure with package code, CLI entrypoint, tests, and deployable app.

## Product framing

> How do you turn a large entertainment catalog into an intelligent retrieval and recommendation surface for discovery, strategy, and future AI applications?

This project answers that with a lightweight but real architecture:

- `raw` layer: normalized CSV inputs for movies, TV, people, orphan titles, and reviews;
- `processed` layer: `catalog_nodes.csv`, `graph_edges.csv`, and run metadata;
- `serving` layer: executive summary, relationship coverage, genre financials, streaming coverage, review terms, and people network summaries;
- `application` layer: Streamlit interface for search, recommendations, and graph exploration.

## App modules

1. **Executive Overview**
   Real analytical signals from summary CSVs, including graph coverage, genre profit concentration, streaming footprint, and network centrality.

2. **Asset Explorer**
   Interactive title exploration with filters and semantic matching.

3. **Hybrid Recommender**
   Weighted recommendation engine combining semantic, graph, quality, and business features.

4. **Network Graph**
   Visual neighborhood around a selected title using recommendation, genre, and people links.

5. **Semantic Search**
   Natural-language retrieval over titles, genres, people, and synopsis text.

6. **Decision Explainer**
   Score decomposition showing why a recommendation ranked.

## Pipeline outputs

Running the pipeline generates:

- `data/processed/catalog_nodes.csv`
- `data/processed/graph_edges.csv`
- `data/processed/run_metadata.json`
- `data/executive_summary.csv`
- `data/relationship_coverage.csv`
- `data/genre_financial_summary.csv`
- `data/streaming_coverage.csv`
- `data/low_review_terms.csv`
- `data/high_review_terms.csv`
- `data/top_people_network_centrality.csv`

## Current analytical signals

From the current committed serving layer:

- movie recommendation coverage reaches `100%` when orphan movie nodes are included;
- TV recommendation coverage reaches `100%` when orphan TV nodes are included;
- movie cast-to-people resolution is about `70.6%`;
- movie director-to-people resolution is about `67.2%`;
- TV creator-to-people resolution is about `60.4%`.

That makes the project stronger for graph retrieval and recommendation because it explicitly handles dangling reference problems instead of ignoring them.

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- Streamlit
- Plotly
- Sentence Transformers
- KaggleHub
- pytest

## Project structure

```text
cinegraph-network-intelligence/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── executive_summary.csv
│   ├── genre_financial_summary.csv
│   ├── high_review_terms.csv
│   ├── low_review_terms.csv
│   ├── relationship_coverage.csv
│   ├── streaming_coverage.csv
│   └── top_people_network_centrality.csv
├── scripts/
│   ├── _bootstrap.py
│   └── cinegraph_network_analysis.py
├── src/
│   └── cinegraph_network_intelligence/
│       ├── __init__.py
│       ├── cli.py
│       └── pipeline.py
├── tests/
│   ├── fixtures/raw/
│   ├── test_pipeline_outputs.py
│   └── test_streamlit_app_smoke.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Run the pipeline

If you have the raw CineGraph CSV files locally:

```bash
python scripts/cinegraph_network_analysis.py data/raw data
```

Or via the installed CLI:

```bash
cinegraph-pipeline data/raw data
```

## Run tests

```bash
pytest
```

## Why the structure is stronger now

- The analytical logic is no longer trapped in one exploratory script.
- The processed layer promised by the app is now formalized in code.
- The app is less brittle because it can run without `networkx`.
- Small fixture data and smoke tests make the repo easier to trust and iterate on.
- The Streamlit homepage now surfaces real summary signals instead of generic placeholder storytelling.

## Next production upgrades

- persist processed full-dataset exports in a lightweight release artifact;
- precompute title embeddings and recommendation candidates for faster deploy performance;
- add vector indexing for semantic retrieval at scale;
- add graph-native features such as Node2Vec or GraphSAGE embeddings;
- create a richer title detail page with poster, people, and recommendation trace;
- extend the serving layer toward warehouse or lakehouse deployment patterns.

## Portfolio value

This repo is useful in a portfolio because it shows more than dashboarding:

- data modeling for connected entities;
- analytical engineering and serving-layer thinking;
- applied retrieval and recommendation design;
- explainability;
- product-oriented app delivery on top of a reproducible data workflow.
