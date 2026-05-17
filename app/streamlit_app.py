"""CineGraph AI Explorer

Interactive intelligence app for graph-based media discovery.

Run locally:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

The app is designed to work with the derived CSV files currently in the repository.
If richer processed catalog files are later added under data/processed/, the same app
will automatically use them for title-level search and recommendations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"

st.set_page_config(
    page_title="CineGraph AI Explorer | Graph + Neural Search for Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #080A12;
            --panel: rgba(255,255,255,.065);
            --line: rgba(255,255,255,.12);
            --text: #F7F4EA;
            --muted: #9FA7B8;
            --cyan: #7DE6FF;
            --violet: #9F7AEA;
            --pink: #FF5C8A;
            --lime: #C8FF59;
        }
        .stApp {
            background:
                radial-gradient(circle at 8% 5%, rgba(125,230,255,.18), transparent 24%),
                radial-gradient(circle at 92% 8%, rgba(159,122,234,.22), transparent 24%),
                linear-gradient(180deg, #080A12 0%, #111827 48%, #080A12 100%);
            color: var(--text);
        }
        .block-container { padding-top: 2.2rem; max-width: 1320px; }
        h1, h2, h3 { letter-spacing: -.055em; }
        h1 { font-size: 4.8rem !important; line-height: .86 !important; margin-bottom: .7rem !important; }
        h2 { font-size: 2.4rem !important; }
        h3 { font-size: 1.55rem !important; }
        p, li { color: #D5DAE6; }
        .hero-card {
            border: 1px solid var(--line);
            border-radius: 30px;
            padding: 32px;
            background: linear-gradient(135deg, rgba(255,255,255,.09), rgba(255,255,255,.035));
            box-shadow: 0 26px 80px rgba(0,0,0,.35);
        }
        .eyebrow {
            color: var(--cyan);
            text-transform: uppercase;
            letter-spacing: .17em;
            font-weight: 900;
            font-size: .78rem;
            margin-bottom: 14px;
        }
        .metric-card {
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 20px;
            background: rgba(255,255,255,.06);
            min-height: 130px;
        }
        .metric-label {
            color: var(--muted);
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .12em;
            font-weight: 800;
        }
        .metric-value {
            color: var(--text);
            font-size: 2.05rem;
            font-weight: 900;
            letter-spacing: -.06em;
            margin-top: 22px;
        }
        .section-card {
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 24px;
            background: rgba(255,255,255,.055);
        }
        .pill {
            display: inline-flex;
            border: 1px solid rgba(125,230,255,.35);
            color: #EAFBFF;
            border-radius: 999px;
            padding: 7px 11px;
            font-size: .78rem;
            margin: 4px 4px 4px 0;
            background: rgba(125,230,255,.08);
        }
        .why-box {
            border-left: 5px solid var(--cyan);
            padding-left: 16px;
            margin-top: 12px;
        }
        .stDataFrame { border-radius: 18px; overflow: hidden; }
        [data-testid="stMetricValue"] { font-size: 2rem; }
        .small-muted { color: var(--muted); font-size: .9rem; }
        div[data-testid="stSidebar"] { background: rgba(8,10,18,.92); }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_summary_data() -> Dict[str, pd.DataFrame]:
    return {
        "executive": read_csv(DATA / "executive_summary.csv"),
        "coverage": read_csv(DATA / "relationship_coverage.csv"),
        "genre": read_csv(DATA / "genre_financial_summary.csv"),
        "streaming": read_csv(DATA / "streaming_coverage.csv"),
        "low_terms": read_csv(DATA / "low_review_terms.csv"),
        "high_terms": read_csv(DATA / "high_review_terms.csv"),
        "people": read_csv(DATA / "top_people_network_centrality.csv"),
        "catalog": read_csv(PROCESSED / "catalog_nodes.csv"),
        "edges": read_csv(PROCESSED / "graph_edges.csv"),
    }


def build_demo_catalog(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Builds a compact searchable demo catalog when title-level processed files are absent."""
    genre = data["genre"].copy()
    people = data["people"].copy()

    rows: List[dict] = []
    for _, row in genre.head(18).iterrows():
        genre_name = str(row.get("genre", "Unknown"))
        rows.append(
            {
                "id": f"genre::{genre_name}",
                "title": f"{genre_name} discovery cluster",
                "media_type": "genre_cluster",
                "overview": (
                    f"A high-signal {genre_name} content cluster with revenue, profit, ROI, vote and rating signals. "
                    f"Useful for catalog strategy, recommendation expansion and audience positioning."
                ),
                "genres": genre_name,
                "rating": float(row.get("avg_rating", 0) or 0),
                "popularity": float(row.get("votes", 0) or 0),
                "revenue": float(row.get("total_revenue", 0) or 0),
                "roi": float(row.get("median_roi", 0) or 0),
                "cast_names": ", ".join(people.head(4)["name"].astype(str).tolist()) if not people.empty else "",
                "why": "Generated from genre financial summary. Add data/processed/catalog_nodes.csv for title-level search.",
            }
        )

    for _, row in people.head(18).iterrows():
        person = str(row.get("name", "Unknown person"))
        rows.append(
            {
                "id": f"person::{person}",
                "title": f"{person} collaboration network",
                "media_type": "people_network",
                "overview": (
                    f"A network-centered view around {person}, based on weighted co-star centrality. "
                    f"Useful for cast graph exploration, influence mapping and recommendation explanation."
                ),
                "genres": "Cast network; collaboration graph; star centrality",
                "rating": 0.0,
                "popularity": float(row.get("weighted_degree", 0) or 0),
                "revenue": 0.0,
                "roi": 0.0,
                "cast_names": person,
                "why": "Generated from top_people_network_centrality.csv. Add processed graph edges for full network traversal.",
            }
        )

    catalog = pd.DataFrame(rows)
    if catalog.empty:
        catalog = pd.DataFrame(
            [
                {
                    "id": "demo::cinegraph",
                    "title": "CineGraph Intelligence Layer",
                    "media_type": "demo",
                    "overview": "Graph analytics, semantic search and hybrid recommendations for movie and TV intelligence.",
                    "genres": "Graph; NLP; Recommender Systems",
                    "rating": 0,
                    "popularity": 1,
                    "revenue": 0,
                    "roi": 0,
                    "cast_names": "",
                    "why": "Demo fallback.",
                }
            ]
        )
    return catalog


def normalize_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    catalog = catalog.copy()
    expected = {
        "id": "",
        "title": "Untitled",
        "media_type": "unknown",
        "overview": "",
        "genres": "",
        "rating": 0.0,
        "popularity": 0.0,
        "revenue": 0.0,
        "roi": 0.0,
        "cast_names": "",
        "why": "",
    }
    for col, default in expected.items():
        if col not in catalog.columns:
            catalog[col] = default
    catalog["search_text"] = (
        catalog["title"].fillna("")
        + " "
        + catalog["overview"].fillna("")
        + " "
        + catalog["genres"].fillna("")
        + " "
        + catalog["cast_names"].fillna("")
    )
    for col in ["rating", "popularity", "revenue", "roi"]:
        catalog[col] = pd.to_numeric(catalog[col], errors="coerce").fillna(0)
    return catalog


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def build_text_vectors(texts: Tuple[str, ...], use_neural: bool):
    if use_neural and HAS_SENTENCE_TRANSFORMERS:
        model = load_embedding_model()
        if model is not None:
            embeddings = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
            return "neural_embeddings", np.asarray(embeddings), None

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_features=3000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(list(texts))
    return "tfidf_fallback", matrix, vectorizer


def semantic_scores(query: str, catalog: pd.DataFrame, use_neural: bool) -> np.ndarray:
    texts = tuple(catalog["search_text"].fillna("").astype(str).tolist())
    mode, matrix, vectorizer = build_text_vectors(texts, use_neural)

    if mode == "neural_embeddings":
        model = load_embedding_model()
        if model is None:
            return np.zeros(len(catalog))
        query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        return cosine_similarity(query_vec, matrix)[0]

    query_vec = vectorizer.transform([query])
    return cosine_similarity(query_vec, matrix)[0]


def minmax(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy().reshape(-1, 1)
    if len(values) == 0 or float(values.max()) == float(values.min()):
        return np.zeros(len(values))
    return MinMaxScaler().fit_transform(values).reshape(-1)


def hybrid_recommendations(
    catalog: pd.DataFrame,
    selected_title: str,
    query_boost: str,
    semantic_weight: float,
    graph_weight: float,
    quality_weight: float,
    business_weight: float,
    use_neural: bool,
) -> pd.DataFrame:
    selected = catalog[catalog["title"] == selected_title].iloc[0]
    query = f"{selected['title']} {selected['overview']} {selected['genres']} {selected['cast_names']} {query_boost}"
    sem = semantic_scores(query, catalog, use_neural)

    selected_genres = set(str(selected.get("genres", "")).lower().replace(";", ",").split(","))
    selected_cast = set(str(selected.get("cast_names", "")).lower().replace(";", ",").split(","))

    def overlap_score(row: pd.Series) -> float:
        genres = set(str(row.get("genres", "")).lower().replace(";", ",").split(","))
        cast = set(str(row.get("cast_names", "")).lower().replace(";", ",").split(","))
        genre_overlap = len(selected_genres & genres) / max(len(selected_genres | genres), 1)
        cast_overlap = len(selected_cast & cast) / max(len(selected_cast | cast), 1)
        return 0.65 * genre_overlap + 0.35 * cast_overlap

    graph = catalog.apply(overlap_score, axis=1).to_numpy()
    quality = minmax(catalog["rating"])
    business = 0.55 * minmax(catalog["popularity"]) + 0.25 * minmax(catalog["revenue"]) + 0.20 * minmax(catalog["roi"])

    score = semantic_weight * sem + graph_weight * graph + quality_weight * quality + business_weight * business
    output = catalog.copy()
    output["semantic_score"] = sem
    output["graph_score"] = graph
    output["quality_score"] = quality
    output["business_score"] = business
    output["hybrid_score"] = score
    output = output[output["title"] != selected_title].sort_values("hybrid_score", ascending=False).head(10)
    return output


def build_network(catalog: pd.DataFrame, center_title: str, recommendations: pd.DataFrame) -> go.Figure:
    graph = nx.Graph()
    graph.add_node(center_title, kind="center")
    for _, row in recommendations.head(8).iterrows():
        title = row["title"]
        graph.add_node(title, kind=row.get("media_type", "node"))
        graph.add_edge(center_title, title, weight=float(row.get("hybrid_score", 0)))
        for token in str(row.get("genres", "")).replace(";", ",").split(",")[:2]:
            token = token.strip()
            if token:
                graph.add_node(token, kind="genre")
                graph.add_edge(title, token, weight=0.25)

    pos = nx.spring_layout(graph, seed=7, k=0.72)
    edge_x, edge_y = [], []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        kind = attrs.get("kind")
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        node_size.append(28 if kind == "center" else 18 if kind == "genre" else 22)
        node_color.append("#7DE6FF" if kind == "center" else "#C8FF59" if kind == "genre" else "#9F7AEA")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.2, color="rgba(255,255,255,.28)"),
            hoverinfo="none",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="rgba(255,255,255,.72)")),
            hovertext=node_text,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        height=580,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def explain_row(row: pd.Series) -> str:
    reasons = []
    if row.get("semantic_score", 0) > 0.15:
        reasons.append("semanticamente próximo da busca ou do título selecionado")
    if row.get("graph_score", 0) > 0.05:
        reasons.append("compartilha sinais de rede, gênero ou elenco")
    if row.get("quality_score", 0) > 0.55:
        reasons.append("tem sinal de qualidade acima da média relativa")
    if row.get("business_score", 0) > 0.55:
        reasons.append("tem força de popularidade, receita ou ROI no catálogo")
    if not reasons:
        reasons.append("aparece como candidato exploratório por similaridade fraca, útil para descoberta")
    return "A recomendação apareceu porque " + ", ".join(reasons) + "."


def metric_card(label: str, value: str, helper: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-muted">{helper}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    data = load_summary_data()
    catalog = data["catalog"] if not data["catalog"].empty else build_demo_catalog(data)
    catalog = normalize_catalog(catalog)

    with st.sidebar:
        st.markdown("### 🎬 CineGraph AI Explorer")
        page = st.radio(
            "Navegação",
            [
                "1. Home",
                "2. Busca filme/série",
                "3. Recomendador híbrido",
                "4. Rede de pessoas e títulos",
                "5. Busca semântica",
                "6. Por que recomendou?",
            ],
        )
        use_neural = st.toggle(
            "Usar embeddings neurais quando disponíveis",
            value=HAS_SENTENCE_TRANSFORMERS,
            help="Quando sentence-transformers estiver instalado, usa MiniLM. Caso contrário, aplica TF-IDF como fallback interpretável.",
        )
        st.caption(
            "SEO/Product angle: graph intelligence, neural search, hybrid recommendation, RAG-ready media catalog."
        )

    executive = data["executive"]
    if not executive.empty:
        summary = executive.iloc[0]
    else:
        summary = pd.Series(dtype="object")

    st.markdown(
        """
        <div class="hero-card">
            <div class="eyebrow">Graph Analytics · Neural Search · Hybrid Recommendations · RAG-ready Media Intelligence</div>
            <h1>CineGraph AI Explorer</h1>
            <p style="font-size:1.22rem;max-width:980px;">
            Uma aplicação inteligente para explorar filmes, séries, pessoas, recomendações e reviews como uma rede de conhecimento audiovisual.
            O objetivo não é só visualizar dados: é explicar conexões, recomendar conteúdo e transformar um catálogo em uma camada de inteligência.
            </p>
            <span class="pill">NetworkX</span><span class="pill">Sentence Embeddings</span><span class="pill">Semantic Search</span><span class="pill">Hybrid Recommender</span><span class="pill">RAG-ready Thinking</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "1. Home":
        st.markdown("## 1. Home com KPIs do catálogo")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Filmes", f"{int(summary.get('movies', 0)):,.0f}".replace(",", "."), "movie nodes")
        with c2:
            metric_card("Séries", f"{int(summary.get('tv_shows', 0)):,.0f}".replace(",", "."), "TV nodes")
        with c3:
            metric_card("Pessoas", f"{int(summary.get('people', 0)):,.0f}".replace(",", "."), "cast, creators, directors")
        with c4:
            metric_card("Reviews", f"{int(summary.get('movie_reviews', 0) + summary.get('tv_reviews', 0)):,.0f}".replace(",", "."), "text mining layer")

        col_a, col_b = st.columns([1.05, 0.95])
        with col_a:
            st.markdown("### Integridade do grafo")
            coverage = data["coverage"]
            if not coverage.empty:
                fig = px.bar(
                    coverage,
                    y="relationship",
                    x="coverage_pct",
                    orientation="h",
                    color="coverage_pct",
                    color_continuous_scale=["#FF5C8A", "#7DE6FF", "#C8FF59"],
                    range_x=[0, 100],
                    text=coverage["coverage_pct"].round(1).astype(str) + "%",
                )
                fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        with col_b:
            st.markdown("### Tese do projeto")
            st.markdown(
                """
                <div class="section-card">
                <p><strong>CineGraph</strong> trata o catálogo como uma rede, não como uma tabela plana.</p>
                <p>A aplicação combina três inteligências:</p>
                <ul>
                    <li><strong>grafo</strong>: relações entre títulos, pessoas, recomendações e gêneros;</li>
                    <li><strong>NLP neural</strong>: embeddings para busca por significado;</li>
                    <li><strong>negócio</strong>: popularidade, receita, ROI e cobertura de streaming.</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Gêneros com maior força financeira reportada")
        genre = data["genre"]
        if not genre.empty:
            fig = px.scatter(
                genre.head(14),
                x="movies",
                y="total_profit",
                size="total_revenue",
                color="median_roi",
                hover_name="genre",
                color_continuous_scale=["#7DE6FF", "#9F7AEA", "#FF5C8A"],
            )
            fig.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    elif page == "2. Busca filme/série":
        st.markdown("## 2. Busca de filme/série")
        query = st.text_input("Busque por título, gênero, pessoa ou descrição", placeholder="Ex.: action adventure high ROI, Johnny Depp, streaming, thriller...")
        if query:
            scores = semantic_scores(query, catalog, use_neural)
            result = catalog.copy()
            result["match_score"] = scores
            result = result.sort_values("match_score", ascending=False).head(12)
        else:
            result = catalog.sort_values("popularity", ascending=False).head(12)
        st.dataframe(
            result[["title", "media_type", "genres", "rating", "popularity", "match_score" if "match_score" in result else "revenue"]],
            use_container_width=True,
            hide_index=True,
        )
        st.info("Para busca título-a-título real, adicione data/processed/catalog_nodes.csv. O app já está pronto para usar esse arquivo automaticamente.")

    elif page == "3. Recomendador híbrido":
        st.markdown("## 3. Recomendador híbrido")
        selected_title = st.selectbox("Escolha um título, cluster ou pessoa central", catalog["title"].sort_values().unique())
        query_boost = st.text_input("Refine a intenção", placeholder="Ex.: dark psychological drama with strong reviews")
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            semantic_weight = st.slider("Semântica", 0.0, 1.0, 0.42, 0.05)
        with w2:
            graph_weight = st.slider("Grafo", 0.0, 1.0, 0.28, 0.05)
        with w3:
            quality_weight = st.slider("Qualidade", 0.0, 1.0, 0.15, 0.05)
        with w4:
            business_weight = st.slider("Negócio", 0.0, 1.0, 0.15, 0.05)
        total = max(semantic_weight + graph_weight + quality_weight + business_weight, 0.01)
        recs = hybrid_recommendations(
            catalog,
            selected_title,
            query_boost,
            semantic_weight / total,
            graph_weight / total,
            quality_weight / total,
            business_weight / total,
            use_neural,
        )
        st.dataframe(
            recs[["title", "media_type", "genres", "hybrid_score", "semantic_score", "graph_score", "quality_score", "business_score"]],
            use_container_width=True,
            hide_index=True,
        )

    elif page == "4. Rede de pessoas e títulos":
        st.markdown("## 4. Exploração de rede de pessoas e títulos")
        selected_title = st.selectbox("Nó central", catalog["title"].sort_values().unique(), key="network_title")
        recs = hybrid_recommendations(catalog, selected_title, "", 0.42, 0.28, 0.15, 0.15, use_neural)
        fig = build_network(catalog, selected_title, recs)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### Pessoas mais centrais na amostra de co-star network")
        people = data["people"]
        if not people.empty:
            st.dataframe(people.head(12), use_container_width=True, hide_index=True)

    elif page == "5. Busca semântica":
        st.markdown("## 5. Busca semântica por descrição")
        description = st.text_area(
            "Descreva o conteúdo que você quer encontrar",
            value="Quero um conteúdo com tensão psicológica, boas avaliações, alta conexão de elenco e potencial de catálogo para streaming.",
            height=120,
        )
        scores = semantic_scores(description, catalog, use_neural)
        result = catalog.copy()
        result["semantic_score"] = scores
        result = result.sort_values("semantic_score", ascending=False).head(12)
        st.dataframe(result[["title", "media_type", "genres", "overview", "semantic_score"]], use_container_width=True, hide_index=True)
        st.caption(
            "Se sentence-transformers estiver disponível, esta etapa usa embeddings neurais. Caso contrário, usa TF-IDF como fallback explicável."
        )

    elif page == "6. Por que recomendou?":
        st.markdown("## 6. Explicação da recomendação")
        selected_title = st.selectbox("Título base", catalog["title"].sort_values().unique(), key="explain_title")
        recs = hybrid_recommendations(catalog, selected_title, "", 0.42, 0.28, 0.15, 0.15, use_neural)
        choice = st.selectbox("Recomendação para explicar", recs["title"].tolist())
        row = recs[recs["title"] == choice].iloc[0]
        st.markdown(
            f"""
            <div class="section-card">
            <h3>{row['title']}</h3>
            <p>{row.get('overview', '')}</p>
            <div class="why-box">
            <strong>Por que apareceu?</strong>
            <p>{explain_row(row)}</p>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Decomposição do score")
        score_df = pd.DataFrame(
            {
                "component": ["semantic", "graph", "quality", "business"],
                "score": [row["semantic_score"], row["graph_score"], row["quality_score"], row["business_score"]],
            }
        )
        fig = px.bar(score_df, x="component", y="score", color="component", color_discrete_sequence=["#7DE6FF", "#9F7AEA", "#C8FF59", "#FF5C8A"])
        fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
