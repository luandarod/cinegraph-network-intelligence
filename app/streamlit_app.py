"""CineGraph AI Explorer — real title-level app.

Run locally:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

Data loading priority:
1. data/processed/catalog_nodes.csv, if already built.
2. Raw CineGraph CSVs in data/raw/ or data/.
3. KaggleHub public dataset download: muhammetyorulmaz1/cinegraph-tmdb-movies-tv-and-people-dataset.

The app builds a real catalog-level semantic and graph recommendation layer from
movies.csv, tv_shows.csv, orphan_movies.csv, orphan_tv.csv and people.csv.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import re

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
    import kagglehub
    HAS_KAGGLEHUB = True
except Exception:
    kagglehub = None
    HAS_KAGGLEHUB = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
KAGGLE_DATASET = "muhammetyorulmaz1/cinegraph-tmdb-movies-tv-and-people-dataset"

RAW_FILENAMES = {
    "movies": "movies.csv",
    "tv": "tv_shows.csv",
    "people": "people.csv",
    "orphan_movies": "orphan_movies.csv",
    "orphan_tv": "orphan_tv.csv",
    "movie_reviews": "movie_reviews.csv",
    "tv_reviews": "tv_reviews.csv",
}

st.set_page_config(
    page_title="CineGraph AI Explorer | Real Movie Graph + Neural Search",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root{--bg:#080A12;--panel:rgba(255,255,255,.065);--line:rgba(255,255,255,.12);--text:#F7F4EA;--muted:#9FA7B8;--cyan:#7DE6FF;--violet:#9F7AEA;--pink:#FF5C8A;--lime:#C8FF59;}
        .stApp{background:radial-gradient(circle at 8% 5%,rgba(125,230,255,.18),transparent 24%),radial-gradient(circle at 92% 8%,rgba(159,122,234,.22),transparent 24%),linear-gradient(180deg,#080A12 0%,#111827 48%,#080A12 100%);color:var(--text);} 
        .block-container{padding-top:2.2rem;max-width:1320px;} h1,h2,h3{letter-spacing:-.055em;} h1{font-size:4.55rem!important;line-height:.88!important;margin-bottom:.7rem!important;} h2{font-size:2.35rem!important;} h3{font-size:1.5rem!important;} p,li{color:#D5DAE6;}
        .hero-card{border:1px solid var(--line);border-radius:30px;padding:32px;background:linear-gradient(135deg,rgba(255,255,255,.09),rgba(255,255,255,.035));box-shadow:0 26px 80px rgba(0,0,0,.35);} .eyebrow{color:var(--cyan);text-transform:uppercase;letter-spacing:.17em;font-weight:900;font-size:.78rem;margin-bottom:14px;}
        .metric-card{border:1px solid var(--line);border-radius:22px;padding:20px;background:rgba(255,255,255,.06);min-height:130px;} .metric-label{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.12em;font-weight:800;} .metric-value{color:var(--text);font-size:2.05rem;font-weight:900;letter-spacing:-.06em;margin-top:22px;}
        .section-card{border:1px solid var(--line);border-radius:24px;padding:24px;background:rgba(255,255,255,.055);} .pill{display:inline-flex;border:1px solid rgba(125,230,255,.35);color:#EAFBFF;border-radius:999px;padding:7px 11px;font-size:.78rem;margin:4px 4px 4px 0;background:rgba(125,230,255,.08);} .why-box{border-left:5px solid var(--cyan);padding-left:16px;margin-top:12px;} .small-muted{color:var(--muted);font-size:.9rem;} div[data-testid="stSidebar"]{background:rgba(8,10,18,.92);}
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_id_list(value: object) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    ids: List[int] = []
    for part in re.split(r"[,;|]", text):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(float(part)))
        except ValueError:
            continue
    return ids


def split_tokens(value: object, limit: int | None = None) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    tokens = [x.strip() for x in re.split(r"[,;|]", text) if x.strip()]
    return tokens[:limit] if limit else tokens


@st.cache_data(show_spinner=False)
def find_raw_files() -> Dict[str, Path]:
    search_roots = [DATA / "raw", DATA, ROOT]
    found: Dict[str, Path] = {}
    for key, filename in RAW_FILENAMES.items():
        for base in search_roots:
            candidate = base / filename
            if candidate.exists():
                found[key] = candidate
                break
    if len(found) >= 5:
        return found

    if HAS_KAGGLEHUB:
        try:
            dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
            for key, filename in RAW_FILENAMES.items():
                if key in found:
                    continue
                matches = list(dataset_path.rglob(filename))
                if matches:
                    found[key] = matches[0]
        except Exception:
            pass
    return found


def read_csv_if_exists(path: Path | None, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, usecols=lambda c: c in set(usecols) if usecols else True, low_memory=False)
    except Exception:
        return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=True)
def load_raw_tables() -> Dict[str, pd.DataFrame]:
    files = find_raw_files()
    movie_cols = [
        "tmdb_id", "imdb_id", "title", "overview", "release_year", "runtime_min", "status", "vote_average",
        "vote_count", "popularity", "budget_usd", "revenue_usd", "profit_usd", "roi_pct", "genres", "keywords",
        "cast_names", "cast_ids", "directors", "director_ids", "similar_ids", "recommended_ids", "similar_titles",
        "recommended_titles", "poster_url", "backdrop_url", "watch_tr_flatrate", "watch_us_flatrate", "watch_us_rent", "watch_us_buy",
    ]
    tv_cols = [
        "tmdb_id", "imdb_id", "title", "overview", "release_year", "status", "show_type", "number_of_seasons",
        "number_of_episodes", "vote_average", "vote_count", "popularity", "genres", "keywords", "networks", "creators",
        "creator_ids", "cast_names", "cast_ids", "similar_ids", "recommended_ids", "similar_titles", "recommended_titles",
        "poster_url", "backdrop_url", "watch_tr_flatrate", "watch_us_flatrate", "watch_us_rent", "watch_us_buy",
    ]
    orphan_movie_cols = ["tmdb_id", "imdb_id", "title", "original_language", "release_year", "overview", "genres", "vote_average", "vote_count", "poster_url"]
    orphan_tv_cols = ["tmdb_id", "title", "original_language", "release_year", "overview", "genres", "vote_average", "vote_count", "poster_url"]
    people_cols = ["tmdb_id", "name", "known_for_dept", "biography", "popularity", "known_movies", "known_movie_ids", "known_tv_shows", "known_tv_ids", "directed_movies", "directed_movie_ids", "profile_url"]
    review_cols = ["review_id", "tmdb_id", "media_type", "rating", "content", "created_at"]

    return {
        "movies": read_csv_if_exists(files.get("movies"), movie_cols),
        "tv": read_csv_if_exists(files.get("tv"), tv_cols),
        "orphan_movies": read_csv_if_exists(files.get("orphan_movies"), orphan_movie_cols),
        "orphan_tv": read_csv_if_exists(files.get("orphan_tv"), orphan_tv_cols),
        "people": read_csv_if_exists(files.get("people"), people_cols),
        "movie_reviews": read_csv_if_exists(files.get("movie_reviews"), review_cols),
        "tv_reviews": read_csv_if_exists(files.get("tv_reviews"), review_cols),
    }


def catalog_from_raw(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    movies = tables["movies"].copy()
    if not movies.empty:
        movies["node_id"] = "movie::" + movies["tmdb_id"].astype(str)
        movies["media_type"] = "movie"
        movies["rating"] = pd.to_numeric(movies.get("vote_average"), errors="coerce")
        movies["business_value"] = pd.to_numeric(movies.get("revenue_usd"), errors="coerce").fillna(0)
        movies["roi"] = pd.to_numeric(movies.get("roi_pct"), errors="coerce").fillna(0)
        frames.append(movies)

    tv = tables["tv"].copy()
    if not tv.empty:
        tv["node_id"] = "tv::" + tv["tmdb_id"].astype(str)
        tv["media_type"] = "tv"
        tv["rating"] = pd.to_numeric(tv.get("vote_average"), errors="coerce")
        tv["business_value"] = pd.to_numeric(tv.get("vote_count"), errors="coerce").fillna(0)
        tv["roi"] = 0.0
        tv["runtime_min"] = np.nan
        tv["directors"] = ""
        tv["director_ids"] = ""
        frames.append(tv)

    orphan_movies = tables["orphan_movies"].copy()
    if not orphan_movies.empty:
        orphan_movies["node_id"] = "orphan_movie::" + orphan_movies["tmdb_id"].astype(str)
        orphan_movies["media_type"] = "orphan_movie"
        orphan_movies["rating"] = pd.to_numeric(orphan_movies.get("vote_average"), errors="coerce")
        orphan_movies["popularity"] = pd.to_numeric(orphan_movies.get("vote_count"), errors="coerce").fillna(0)
        orphan_movies["business_value"] = 0.0
        orphan_movies["roi"] = 0.0
        orphan_movies["cast_names"] = ""
        orphan_movies["cast_ids"] = ""
        orphan_movies["directors"] = ""
        orphan_movies["director_ids"] = ""
        orphan_movies["similar_ids"] = ""
        orphan_movies["recommended_ids"] = ""
        frames.append(orphan_movies)

    orphan_tv = tables["orphan_tv"].copy()
    if not orphan_tv.empty:
        orphan_tv["node_id"] = "orphan_tv::" + orphan_tv["tmdb_id"].astype(str)
        orphan_tv["media_type"] = "orphan_tv"
        orphan_tv["rating"] = pd.to_numeric(orphan_tv.get("vote_average"), errors="coerce")
        orphan_tv["popularity"] = pd.to_numeric(orphan_tv.get("vote_count"), errors="coerce").fillna(0)
        orphan_tv["business_value"] = 0.0
        orphan_tv["roi"] = 0.0
        orphan_tv["cast_names"] = ""
        orphan_tv["cast_ids"] = ""
        orphan_tv["directors"] = ""
        orphan_tv["director_ids"] = ""
        orphan_tv["similar_ids"] = ""
        orphan_tv["recommended_ids"] = ""
        frames.append(orphan_tv)

    if not frames:
        return pd.DataFrame()

    catalog = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["tmdb_id", "title", "overview", "genres", "keywords", "cast_names", "cast_ids", "directors", "director_ids", "creators", "creator_ids", "similar_ids", "recommended_ids", "poster_url", "backdrop_url"]:
        if col not in catalog.columns:
            catalog[col] = ""
    for col in ["rating", "vote_count", "popularity", "business_value", "roi", "release_year"]:
        catalog[col] = pd.to_numeric(catalog.get(col), errors="coerce").fillna(0)
    catalog["people_names"] = (catalog["cast_names"].fillna("") + ", " + catalog.get("directors", "").fillna("") + ", " + catalog.get("creators", "").fillna(""))
    catalog["people_ids"] = (catalog["cast_ids"].fillna("") + "," + catalog.get("director_ids", "").fillna("") + "," + catalog.get("creator_ids", "").fillna(""))
    catalog["search_text"] = (
        catalog["title"].fillna("") + " " + catalog["overview"].fillna("") + " " + catalog["genres"].fillna("") + " " + catalog["keywords"].fillna("") + " " + catalog["people_names"].fillna("")
    )
    catalog = catalog.dropna(subset=["title"]).drop_duplicates(subset=["media_type", "tmdb_id"])
    return catalog.reset_index(drop=True)


@st.cache_data(show_spinner=True)
def load_catalog() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    processed = read_csv_if_exists(PROCESSED / "catalog_nodes.csv")
    if not processed.empty:
        tables = load_raw_tables()
        processed["search_text"] = processed[[c for c in ["title", "overview", "genres", "keywords", "people_names"] if c in processed.columns]].fillna("").agg(" ".join, axis=1)
        return processed, tables
    tables = load_raw_tables()
    catalog = catalog_from_raw(tables)
    return catalog, tables


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


@st.cache_data(show_spinner=True)
def build_text_vectors(texts: Tuple[str, ...], use_neural: bool):
    if use_neural and HAS_SENTENCE_TRANSFORMERS:
        model = load_embedding_model()
        if model is not None:
            embeddings = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=128)
            return "neural_embeddings", np.asarray(embeddings), None
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2, max_features=18000, ngram_range=(1, 2))
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


def id_set(value: object) -> set[int]:
    return set(parse_id_list(value))


def token_set(value: object) -> set[str]:
    return {x.lower() for x in split_tokens(value) if x}


def hybrid_recommendations(catalog: pd.DataFrame, selected_idx: int, query_boost: str, weights: Tuple[float, float, float, float], use_neural: bool) -> pd.DataFrame:
    selected = catalog.iloc[selected_idx]
    query = f"{selected.get('title','')} {selected.get('overview','')} {selected.get('genres','')} {selected.get('people_names','')} {query_boost}"
    sem = semantic_scores(query, catalog, use_neural)

    selected_genres = token_set(selected.get("genres", ""))
    selected_people = id_set(selected.get("people_ids", ""))
    selected_similar = id_set(selected.get("similar_ids", "")) | id_set(selected.get("recommended_ids", ""))

    def graph_score(row: pd.Series) -> float:
        candidate_id = int(row.get("tmdb_id", 0) or 0)
        direct = 1.0 if candidate_id in selected_similar else 0.0
        genres = token_set(row.get("genres", ""))
        people = id_set(row.get("people_ids", ""))
        genre_overlap = len(selected_genres & genres) / max(len(selected_genres | genres), 1)
        people_overlap = len(selected_people & people) / max(len(selected_people | people), 1)
        return 0.48 * direct + 0.30 * genre_overlap + 0.22 * people_overlap

    graph = catalog.apply(graph_score, axis=1).to_numpy()
    quality = 0.70 * minmax(catalog["rating"]) + 0.30 * minmax(catalog["vote_count"])
    business = 0.45 * minmax(catalog["popularity"]) + 0.35 * minmax(catalog["business_value"]) + 0.20 * minmax(catalog["roi"].clip(lower=0, upper=catalog["roi"].quantile(0.95) if len(catalog) > 20 else catalog["roi"].max()))
    w_sem, w_graph, w_quality, w_business = weights
    score = w_sem * sem + w_graph * graph + w_quality * quality + w_business * business

    output = catalog.copy()
    output["semantic_score"] = sem
    output["graph_score"] = graph
    output["quality_score"] = quality
    output["business_score"] = business
    output["hybrid_score"] = score
    output = output[output.index != selected_idx].sort_values("hybrid_score", ascending=False).head(20)
    return output


def graph_figure(catalog: pd.DataFrame, center: pd.Series, recommendations: pd.DataFrame, people: pd.DataFrame) -> go.Figure:
    graph = nx.Graph()
    center_title = center["title"]
    graph.add_node(center_title, kind="center")
    people_lookup = dict(zip(pd.to_numeric(people.get("tmdb_id", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int), people.get("name", pd.Series(dtype=str)).astype(str))) if not people.empty else {}
    for _, row in recommendations.head(7).iterrows():
        title = str(row["title"])
        graph.add_node(title, kind=str(row.get("media_type", "title")))
        graph.add_edge(center_title, title, weight=float(row.get("hybrid_score", 0)))
        for genre in split_tokens(row.get("genres", ""), limit=2):
            graph.add_node(genre, kind="genre")
            graph.add_edge(title, genre, weight=0.25)
        for pid in parse_id_list(row.get("people_ids", ""))[:2]:
            pname = people_lookup.get(pid)
            if pname:
                graph.add_node(pname, kind="person")
                graph.add_edge(title, pname, weight=0.35)

    pos = nx.spring_layout(graph, seed=11, k=0.76)
    edge_x, edge_y = [], []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    color_map = {"center": "#7DE6FF", "genre": "#C8FF59", "person": "#FF5C8A", "movie": "#9F7AEA", "tv": "#9F7AEA", "orphan_movie": "#A3A3A3", "orphan_tv": "#A3A3A3"}
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node, attrs in graph.nodes(data=True):
        x, y = pos[node]
        kind = attrs.get("kind", "movie")
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        node_color.append(color_map.get(kind, "#9F7AEA"))
        node_size.append(30 if kind == "center" else 19 if kind == "genre" else 23)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1.15, color="rgba(255,255,255,.28)"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=node_text, textposition="top center", marker=dict(size=node_size, color=node_color, line=dict(width=1, color="rgba(255,255,255,.75)")), hoverinfo="text"))
    fig.update_layout(height=610, margin=dict(l=0, r=0, t=20, b=0), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def explain(row: pd.Series) -> str:
    reasons = []
    if row.get("semantic_score", 0) > 0.18:
        reasons.append("tem alta similaridade semântica com a intenção de busca ou com a sinopse do título-base")
    if row.get("graph_score", 0) > 0.12:
        reasons.append("está próximo no grafo por recomendações, gêneros ou pessoas em comum")
    if row.get("quality_score", 0) > 0.55:
        reasons.append("tem sinal relativo de qualidade por rating e volume de votos")
    if row.get("business_score", 0) > 0.55:
        reasons.append("tem força de negócio por popularidade, receita, ROI ou volume")
    if not reasons:
        reasons.append("aparece como descoberta exploratória com sinal fraco, mas ainda relacionado ao espaço semântico do catálogo")
    return "A recomendação apareceu porque " + "; ".join(reasons) + "."


def metric_card(label: str, value: str, helper: str = "") -> None:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='small-muted'>{helper}</div></div>", unsafe_allow_html=True)


def main() -> None:
    catalog, tables = load_catalog()
    if catalog.empty:
        st.error("Não encontrei os arquivos reais. Coloque movies.csv, tv_shows.csv, people.csv, orphan_movies.csv e orphan_tv.csv em data/raw/ ou permita o download via KaggleHub.")
        st.stop()

    st.sidebar.markdown("### 🎬 CineGraph AI Explorer")
    page = st.sidebar.radio("Navegação", ["1. Home", "2. Busca filme/série", "3. Recomendador híbrido", "4. Rede de pessoas e títulos", "5. Busca semântica", "6. Por que recomendou?"])
    media_filter = st.sidebar.multiselect("Tipos de mídia", sorted(catalog["media_type"].dropna().unique()), default=[x for x in ["movie", "tv"] if x in set(catalog["media_type"])])
    max_records = st.sidebar.slider("Registros no índice semântico", 2000, min(50000, len(catalog)), min(18000, len(catalog)), 1000)
    use_neural = st.sidebar.toggle("Embeddings neurais", value=False, help="Ative para usar MiniLM quando sentence-transformers estiver disponível. TF-IDF continua como fallback mais rápido.")
    filtered = catalog[catalog["media_type"].isin(media_filter)] if media_filter else catalog
    filtered = filtered.sort_values(["popularity", "vote_count"], ascending=False).head(max_records).reset_index(drop=True)

    people = tables.get("people", pd.DataFrame())

    st.markdown("""
    <div class='hero-card'>
      <div class='eyebrow'>Real TMDB catalog · graph intelligence · semantic search · hybrid recommender · explainable AI</div>
      <h1>CineGraph AI Explorer</h1>
      <p style='font-size:1.18rem;max-width:1010px;'>Aplicação real de inteligência audiovisual construída sobre filmes, séries, pessoas, órfãos de recomendação e reviews. O app transforma o dataset CineGraph em uma camada interativa de busca neural, recomendação híbrida e exploração de rede.</p>
      <span class='pill'>movies.csv</span><span class='pill'>tv_shows.csv</span><span class='pill'>people.csv</span><span class='pill'>orphan nodes</span><span class='pill'>semantic search</span><span class='pill'>graph recommender</span>
    </div>
    """, unsafe_allow_html=True)

    if page == "1. Home":
        st.markdown("## 1. Home com KPIs do catálogo")
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Filmes", f"{(catalog['media_type']=='movie').sum():,}".replace(",", "."), "movie nodes")
        with c2: metric_card("Séries", f"{(catalog['media_type']=='tv').sum():,}".replace(",", "."), "TV nodes")
        with c3: metric_card("Pessoas", f"{len(people):,}".replace(",", "."), "people.csv")
        with c4: metric_card("Índice ativo", f"{len(filtered):,}".replace(",", "."), "semantic + graph candidates")
        col1, col2 = st.columns([1.1, .9])
        with col1:
            genre_rows = []
            for genres in catalog["genres"].dropna().head(80000):
                for g in split_tokens(genres):
                    genre_rows.append(g)
            genre_df = pd.Series(genre_rows).value_counts().head(18).reset_index()
            genre_df.columns = ["genre", "titles"]
            fig = px.bar(genre_df, x="titles", y="genre", orientation="h", color="titles", color_continuous_scale=["#7DE6FF", "#9F7AEA", "#FF5C8A"])
            fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### Como a inteligência funciona")
            st.markdown("""
            <div class='section-card'>
            <p><strong>1. Busca semântica:</strong> compara a intenção do usuário com sinopse, gênero, keywords e pessoas.</p>
            <p><strong>2. Grafo:</strong> usa recommended_ids, similar_ids, elenco, diretores, criadores e gêneros.</p>
            <p><strong>3. Negócio:</strong> considera popularidade, votos, receita, ROI e qualidade relativa.</p>
            <p><strong>4. Explicação:</strong> decompõe cada recomendação em score semântico, grafo, qualidade e negócio.</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("### Amostra real do catálogo")
        st.dataframe(filtered[["title", "media_type", "release_year", "genres", "rating", "vote_count", "popularity"]].head(20), use_container_width=True, hide_index=True)

    elif page == "2. Busca filme/série":
        st.markdown("## 2. Busca de filme/série")
        query = st.text_input("Busque por título, gênero, pessoa ou descrição", placeholder="Ex.: Matrix, cyberpunk, Christopher Nolan, dark psychological thriller")
        if query:
            scores = semantic_scores(query, filtered, use_neural)
            results = filtered.copy(); results["match_score"] = scores
            results = results.sort_values("match_score", ascending=False).head(30)
        else:
            results = filtered.sort_values("popularity", ascending=False).head(30)
        st.dataframe(results[["title", "media_type", "release_year", "genres", "people_names", "rating", "popularity"] + (["match_score"] if "match_score" in results.columns else [])], use_container_width=True, hide_index=True)

    elif page == "3. Recomendador híbrido":
        st.markdown("## 3. Recomendador híbrido")
        names = filtered["title"].fillna("Untitled") + " — " + filtered["media_type"].astype(str) + " #" + filtered["tmdb_id"].astype(str)
        selected_label = st.selectbox("Escolha um título real", names.tolist())
        selected_idx = names[names == selected_label].index[0]
        query_boost = st.text_input("Refine a intenção", placeholder="Ex.: visually ambitious, dystopian, strong cast, high quality")
        cols = st.columns(4)
        vals = [cols[0].slider("Semântica", 0.0, 1.0, .40, .05), cols[1].slider("Grafo", 0.0, 1.0, .30, .05), cols[2].slider("Qualidade", 0.0, 1.0, .15, .05), cols[3].slider("Negócio", 0.0, 1.0, .15, .05)]
        total = max(sum(vals), .01)
        recs = hybrid_recommendations(filtered, selected_idx, query_boost, tuple(v/total for v in vals), use_neural)
        st.dataframe(recs[["title", "media_type", "release_year", "genres", "rating", "hybrid_score", "semantic_score", "graph_score", "quality_score", "business_score"]], use_container_width=True, hide_index=True)

    elif page == "4. Rede de pessoas e títulos":
        st.markdown("## 4. Exploração de rede de pessoas e títulos")
        names = filtered["title"].fillna("Untitled") + " — " + filtered["media_type"].astype(str) + " #" + filtered["tmdb_id"].astype(str)
        selected_label = st.selectbox("Nó central", names.tolist(), key="network")
        selected_idx = names[names == selected_label].index[0]
        recs = hybrid_recommendations(filtered, selected_idx, "", (.40,.30,.15,.15), use_neural)
        fig = graph_figure(filtered, filtered.iloc[selected_idx], recs, people)
        st.plotly_chart(fig, use_container_width=True)
        if not people.empty:
            st.markdown("### Pessoas com maior popularidade no dataset")
            st.dataframe(people.sort_values("popularity", ascending=False).head(15)[["name", "known_for_dept", "popularity", "known_movies", "known_tv_shows"]], use_container_width=True, hide_index=True)

    elif page == "5. Busca semântica":
        st.markdown("## 5. Busca semântica por descrição")
        description = st.text_area("Descreva o conteúdo que você quer encontrar", value="Quero um filme sombrio, psicológico, com tecnologia, tensão e boa avaliação.", height=120)
        scores = semantic_scores(description, filtered, use_neural)
        result = filtered.copy(); result["semantic_score"] = scores
        result = result.sort_values("semantic_score", ascending=False).head(30)
        st.dataframe(result[["title", "media_type", "release_year", "genres", "overview", "semantic_score"]], use_container_width=True, hide_index=True)

    elif page == "6. Por que recomendou?":
        st.markdown("## 6. Explicação da recomendação")
        names = filtered["title"].fillna("Untitled") + " — " + filtered["media_type"].astype(str) + " #" + filtered["tmdb_id"].astype(str)
        selected_label = st.selectbox("Título base", names.tolist(), key="explain")
        selected_idx = names[names == selected_label].index[0]
        recs = hybrid_recommendations(filtered, selected_idx, "", (.40,.30,.15,.15), use_neural)
        choice = st.selectbox("Recomendação para explicar", recs["title"].tolist())
        row = recs[recs["title"] == choice].iloc[0]
        st.markdown(f"""
        <div class='section-card'>
          <h3>{row['title']}</h3>
          <p>{row.get('overview','')}</p>
          <div class='why-box'><strong>Por que apareceu?</strong><p>{explain(row)}</p></div>
        </div>
        """, unsafe_allow_html=True)
        score_df = pd.DataFrame({"component":["semantic","graph","quality","business"],"score":[row["semantic_score"],row["graph_score"],row["quality_score"],row["business_score"]]})
        fig = px.bar(score_df, x="component", y="score", color="component", color_discrete_sequence=["#7DE6FF", "#9F7AEA", "#C8FF59", "#FF5C8A"])
        fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
