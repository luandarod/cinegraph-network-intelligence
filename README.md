# CineGraph Network Intelligence

**Graph Analytics | Neural Search | Hybrid Recommender Systems | NLP | Knowledge Graphs | RAG-Ready Media Intelligence**

CineGraph Network Intelligence explores the CineGraph TMDB dataset as a connected media graph, not as a flat movie table.

The project includes **CineGraph AI Explorer**, a Streamlit application designed as an intelligent media discovery prototype. It combines graph analytics, semantic search, hybrid recommendations and explainable recommendation logic for movie, TV and people-network exploration.

## Live preview

**[Open CineGraph AI Explorer →](https://cinegraph-network-intelligence-lcsr.streamlit.app/)**

> **Preview version:** this Streamlit app is an early public version. It already demonstrates the real product concept, the graph-based recommendation logic, semantic search and explainability layer. The next iterations will improve performance, cache handling, title-level embeddings, visual polish and persistent vector/graph indexes.

## Product question

> How can a large entertainment catalog become an intelligent discovery layer for recommendations, semantic search, content strategy and RAG-ready media retrieval?

## App modules

1. **Home with catalog KPIs**  
   Executive overview of movies, TV shows, people, reviews, graph coverage and financial genre signals.

2. **Movie / TV search**  
   Search by title, genre, person or description using semantic matching.

3. **Hybrid recommender**  
   Recommendation logic combining semantic similarity, graph proximity, quality signals and business signals.

4. **People and title network exploration**  
   Interactive network view connecting the selected node with recommended nodes, people and genre signals.

5. **Semantic search by description**  
   Natural-language search for content discovery, using sentence embeddings when available and TF-IDF as an interpretable fallback.

6. **Recommendation explanation**  
   Explains why each recommendation appeared by decomposing semantic, graph, quality and business scores.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The application loads data in this order:

```text
1. data/processed/catalog_nodes.csv, if available
2. raw CineGraph CSV files in data/raw/ or data/
3. KaggleHub dataset download:
   muhammetyorulmaz1/cinegraph-tmdb-movies-tv-and-people-dataset
```

## Current dataset layer

CineGraph contains normalized TMDB-based data for movies, TV shows, people, orphans and reviews.

| Table | Rows | Description |
|---|---:|---|
| Movies | 22,393 | Full movie metadata, financials, cast, recommendations and streaming availability |
| TV Shows | 15,562 | Series metadata, creators, cast, networks, certifications and streaming availability |
| People | 58,393 | Cast, directors, creators and social/profile metadata |
| Orphan Movies | 8,068 | Lightweight movie nodes referenced by recommendation edges |
| Orphan TV | 3,389 | Lightweight TV nodes referenced by recommendation edges |
| Movie Reviews | 22,712 | Text reviews and optional ratings |
| TV Reviews | 2,923 | Text reviews and optional ratings |

## Why this project is different

Previous portfolio projects focused on healthcare, BI, logistics and predictive modeling. This project adds a deeper technical layer:

- graph-first data modeling;
- many-to-many edge extraction from comma-separated IDs;
- entity resolution and relationship coverage analysis;
- co-star collaboration network analysis;
- recommendation graph analysis with orphan node handling;
- semantic search with neural embeddings when available;
- hybrid recommendation logic;
- explainable recommendation scoring;
- RAG-ready thinking for media knowledge retrieval.

## Key graph metrics

| Relationship | Resolved reference coverage |
|---|---:|
| Movie recommendations, main corpus only | 69.7% |
| Movie recommendations, with orphan movies | 100.0% |
| TV recommendations, main corpus only | 99.9% |
| TV recommendations, with orphan TV | 100.0% |
| Movie cast IDs resolved to people | 70.6% |
| Movie director IDs resolved to people | 67.2% |
| TV creator IDs resolved to people | 60.4% |

The orphan tables matter because they prevent graph edges from dangling. This makes the dataset more suitable for recommendation graphs, network analysis and RAG-style entity retrieval.

## Hybrid recommendation logic

The recommendation layer uses a weighted score:

```text
hybrid_score =
  semantic_weight * semantic_similarity
+ graph_weight    * graph_proximity
+ quality_weight  * rating_signal
+ business_weight * popularity_revenue_roi_signal
```

This makes the recommendation interpretable. The app does not only return a list; it explains whether each result appeared because of semantic similarity, graph proximity, quality or business strength.

## Preview limitations

This public app is a preview and still has known improvement areas:

- Streamlit cold start may be slow when the app downloads or indexes the large Kaggle dataset.
- The app currently uses in-memory semantic vectors; a persistent FAISS index will improve response time.
- Neural embeddings are optional and can be replaced by TF-IDF fallback depending on environment constraints.
- A fully processed `catalog_nodes.csv` and `graph_edges.csv` layer will make the app faster and more stable.
- The graph view is intentionally compact; a future Neo4j or PyVis view can support deeper relationship exploration.
- Review text can be expanded into sentiment, topic modeling and aspect-based recommendation.

## Roadmap / Next improvements

| Area | Improvement |
|---|---|
| Data layer | Build `data/processed/catalog_nodes.csv` and `graph_edges.csv` from the full raw dataset |
| Neural search | Persist sentence-transformer embeddings and add FAISS retrieval |
| Graph intelligence | Add Node2Vec or GraphSAGE embeddings for graph-native recommendation |
| RAG layer | Add natural-language Q&A over movies, TV shows, people and reviews |
| UX | Improve card layout, poster previews, filters and title detail pages |
| Performance | Add cache checkpoints and precomputed recommendation candidates |
| Deployment | Stabilize Streamlit cold start and memory usage |

## Portuguese note / Nota em português

**Versão prévia:** o app publicado no Streamlit é uma primeira versão pública do produto analítico. Ele já demonstra busca semântica, recomendação híbrida, exploração de rede e explicação das recomendações. As próximas melhorias previstas são: criação de arquivos processados leves, embeddings persistentes, índice FAISS, Node2Vec/GraphSAGE, camada RAG, melhoria de layout, filtros mais fortes e otimização de performance no deploy.

## SEO / product positioning

This project is positioned around high-signal keywords for data and AI portfolios:

- graph analytics for media catalogs;
- neural search for movies and TV shows;
- hybrid recommender systems;
- explainable recommendation engine;
- RAG-ready knowledge graph;
- semantic search with sentence embeddings;
- entertainment data intelligence;
- content discovery analytics.

## Methods

1. Loaded normalized CSV files.
2. Validated primary table sizes and schema.
3. Parsed comma-separated ID lists into edge lists.
4. Measured entity resolution coverage for cast, director, creator and recommendation relationships.
5. Added orphan movie/TV nodes to eliminate dangling recommendation edges.
6. Built a co-star network sample with NetworkX.
7. Aggregated financial performance by genre.
8. Compared streaming availability between Turkey and the United States.
9. Applied TF-IDF to review text for interpretable NLP.
10. Added a Streamlit intelligence app for search, hybrid recommendation and graph exploration.
11. Added a neural embedding path using Sentence Transformers when available.
12. Added KaggleHub fallback for real dataset loading.

## Files

```text
cinegraph-network-intelligence/
├── README.md
├── app/
│   └── streamlit_app.py
├── .streamlit/
│   └── config.toml
├── assets/
│   ├── chart_graph_integrity.svg
│   ├── chart_costar_network.svg
│   ├── chart_genre_profit.svg
│   ├── chart_streaming_coverage.svg
│   └── chart_review_terms.svg
├── data/
│   ├── executive_summary.csv
│   ├── relationship_coverage.csv
│   ├── genre_financial_summary.csv
│   ├── streaming_coverage.csv
│   ├── low_review_terms.csv
│   ├── high_review_terms.csv
│   └── top_people_network_centrality.csv
├── requirements.txt
└── scripts/
    └── cinegraph_network_analysis.py
```

## Tools and skills demonstrated

| Area | Tools / concepts |
|---|---|
| Graph analytics | NetworkX, edge lists, entity resolution, co-star networks |
| Neural NLP | Sentence Transformers, embeddings, semantic search |
| NLP fallback | TF-IDF, review text mining, interpretable term analysis |
| Recommendation systems | Hybrid scoring, semantic similarity, graph proximity, explainability |
| Data modeling | Normalized tables, many-to-many relationships, orphan node handling |
| Product analytics | Catalog discovery, content strategy, streaming coverage, financial signals |
| Visualization | Streamlit, Plotly, Matplotlib/SVG charts |
| RAG readiness | Entity-centered schema, graph links, metadata-rich retrieval design |

## Limitations

- The graph is built from TMDB metadata and reflects availability and quality of that source.
- Some person relationships do not resolve to `people.csv`, especially for TV creators and minor credits.
- Reviews are multilingual and unevenly distributed, so NLP results should be interpreted as exploratory.
- Financial analysis only uses movies with both budget and revenue available.
- The current app is a preview and will improve as the processed title-level layer becomes more complete.
