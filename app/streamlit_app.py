"""CineGraph AI Explorer.

Run locally:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

Data loading priority:
1. data/processed/catalog_nodes.csv
2. Raw CineGraph CSVs in data/raw/ or data/
3. KaggleHub public dataset download:
   muhammetyorulmaz1/cinegraph-tmdb-movies-tv-and-people-dataset
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

SUMMARY_FILENAMES = {
    "executive_summary": "executive_summary.csv",
    "relationship_coverage": "relationship_coverage.csv",
    "genre_financial_summary": "genre_financial_summary.csv",
    "streaming_coverage": "streaming_coverage.csv",
    "low_review_terms": "low_review_terms.csv",
    "high_review_terms": "high_review_terms.csv",
    "top_people_network_centrality": "top_people_network_centrality.csv",
}

st.set_page_config(
    page_title="CineGraph AI Explorer | Media Graph Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root{
            --bg:#051217;
            --bg-soft:#0A1C23;
            --panel:rgba(255,255,255,.055);
            --line:rgba(125,230,255,.14);
            --text:#F4F7F6;
            --muted:#99AFB6;
            --teal:#16C6C9;
            --teal-strong:#0E8F98;
            --sea:#7DE6FF;
            --gold:#D9C48B;
        }
        .stApp{
            background:
                radial-gradient(circle at 10% 0%, rgba(22,198,201,.18), transparent 28%),
                radial-gradient(circle at 90% 10%, rgba(125,230,255,.14), transparent 24%),
                linear-gradient(180deg, #051217 0%, #071920 48%, #051217 100%);
            color:var(--text);
        }
        .block-container{padding-top:2.2rem;max-width:1340px;}
        h1,h2,h3{letter-spacing:-.04em;}
        h1{font-size:4.4rem!important;line-height:.9!important;margin-bottom:.7rem!important;}
        h2{font-size:2.15rem!important;}
        h3{font-size:1.35rem!important;}
        p,li,span{color:#D6E1E3;}
        .hero-card{
            border:1px solid var(--line);
            border-radius:30px;
            padding:32px;
            background:linear-gradient(145deg, rgba(10,28,35,.96), rgba(8,19,24,.72));
            box-shadow:0 24px 70px rgba(0,0,0,.32);
        }
        .eyebrow{
            color:var(--sea);
            text-transform:uppercase;
            letter-spacing:.16em;
            font-weight:800;
            font-size:.78rem;
            margin-bottom:14px;
        }
        .metric-card{
            border:1px solid var(--line);
            border-radius:22px;
            padding:20px;
            background:rgba(255,255,255,.045);
            min-height:130px;
        }
        .metric-label{
            color:var(--muted);
            font-size:.78rem;
            text-transform:uppercase;
            letter-spacing:.12em;
            font-weight:800;
        }
        .metric-value{
            color:var(--text);
            font-size:2rem;
            font-weight:900;
            letter-spacing:-.06em;
            margin-top:20px;
        }
        .section-card{
            border:1px solid var(--line);
            border-radius:24px;
            padding:24px;
            background:rgba(255,255,255,.045);
        }
        .pill{
            display:inline-flex;
            border:1px solid rgba(125,230,255,.24);
            color:#E7FBFC;
            border-radius:999px;
            padding:7px 11px;
            font-size:.78rem;
            margin:4px 4px 4px 0;
            background:rgba(125,230,255,.08);
        }
        .why-box{
            border-left:5px solid var(--teal);
            padding-left:16px;
            margin-top:12px;
        }
        .small-muted{color:var(--muted);font-size:.9rem;}
        div[data-testid="stSidebar"]{background:rgba(4,12,16,.94);}
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
        token = part.strip()
        if not token:
            continue
        try:
            ids.append(int(float(token)))
        except ValueError:
            continue
    return ids


def split_tokens(value: object, limit: int | None = None) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    tokens = [item.strip() for item in re.split(r"[,;|]", text) if item.strip()]
    return tokens[:limit] if limit else tokens


def normalize_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df


def normalize_text(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = ""
        df[column] = (
            df[column]
            .where(df[column].notna(), "")
            .astype(str)
            .replace({"nan": "", "None": "", "null": ""})
            .str.strip()
        )
    return df


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
        return pd.read_csv(path, usecols=lambda column: column in set(usecols) if usecols else True, low_memory=False)
    except Exception:
        return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner=True)
def load_raw_tables() -> Dict[str, pd.DataFrame]:
    files = find_raw_files()
    movie_cols = [
        "tmdb_id", "imdb_id", "title", "overview", "release_year", "runtime_min", "status", "vote_average",
        "vote_count", "popularity", "budget_usd", "revenue_usd", "profit_usd", "roi_pct", "genres", "keywords",
        "cast_names", "cast_ids", "directors", "director_ids", "similar_ids", "recommended_ids", "similar_titles",
        "recommended_titles", "poster_url", "backdrop_url", "watch_tr_flatrate", "watch_tr_rent", "watch_tr_buy",
        "watch_tr_free", "watch_us_flatrate", "watch_us_rent", "watch_us_buy", "watch_us_free",
    ]
    tv_cols = [
        "tmdb_id", "imdb_id", "title", "overview", "release_year", "status", "show_type", "number_of_seasons",
        "number_of_episodes", "vote_average", "vote_count", "popularity", "genres", "keywords", "networks",
        "creators", "creator_ids", "cast_names", "cast_ids", "similar_ids", "recommended_ids", "similar_titles",
        "recommended_titles", "poster_url", "backdrop_url", "watch_tr_flatrate", "watch_tr_rent", "watch_tr_buy",
        "watch_tr_free", "watch_us_flatrate", "watch_us_rent", "watch_us_buy", "watch_us_free",
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
        orphan_movies["creators"] = ""
        orphan_movies["creator_ids"] = ""
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
        orphan_tv["creators"] = ""
        orphan_tv["creator_ids"] = ""
        orphan_tv["similar_ids"] = ""
        orphan_tv["recommended_ids"] = ""
        frames.append(orphan_tv)

    if not frames:
        return pd.DataFrame()

    catalog = pd.concat(frames, ignore_index=True, sort=False)
    catalog = normalize_text(
        catalog,
        [
            "title",
            "overview",
            "genres",
            "keywords",
            "cast_names",
            "cast_ids",
            "directors",
            "director_ids",
            "creators",
            "creator_ids",
            "similar_ids",
            "recommended_ids",
            "poster_url",
            "backdrop_url",
        ],
    )
    catalog = normalize_numeric(catalog, ["rating", "vote_count", "popularity", "business_value", "roi", "release_year"])
    catalog["people_names"] = (
        catalog["cast_names"] + ", " + catalog["directors"] + ", " + catalog["creators"]
    ).str.strip(", ")
    catalog["people_ids"] = (
        catalog["cast_ids"] + "," + catalog["director_ids"] + "," + catalog["creator_ids"]
    ).str.strip(",")
    catalog["search_text"] = (
        catalog["title"] + " "
        + catalog["overview"] + " "
        + catalog["genres"] + " "
        + catalog["keywords"] + " "
        + catalog["people_names"]
    ).str.strip()
    catalog = catalog.dropna(subset=["title"]).drop_duplicates(subset=["media_type", "tmdb_id"])
    return catalog.reset_index(drop=True)


@st.cache_data(show_spinner=True)
def load_catalog() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    processed = read_csv_if_exists(PROCESSED / "catalog_nodes.csv")
    if not processed.empty:
        tables = load_raw_tables()
        processed = normalize_text(processed, [column for column in ["title", "overview", "genres", "keywords", "people_names"] if column in processed.columns])
        processed["search_text"] = processed[[column for column in ["title", "overview", "genres", "keywords", "people_names"] if column in processed.columns]].agg(" ".join, axis=1)
        return processed, tables
    tables = load_raw_tables()
    catalog = catalog_from_raw(tables)
    return catalog, tables


@st.cache_data(show_spinner=False)
def load_summary_exports() -> Dict[str, pd.DataFrame]:
    return {key: read_csv_if_exists(DATA / filename) for key, filename in SUMMARY_FILENAMES.items()}


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

    min_df = 2 if len(texts) >= 20 else 1
    vectorizer = TfidfVectorizer(stop_words="english", min_df=min_df, max_features=18000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(list(texts))
    return "tfidf_fallback", matrix, vectorizer


def semantic_scores(query: str, catalog: pd.DataFrame, use_neural: bool) -> np.ndarray:
    if catalog.empty:
        return np.array([])
    texts = tuple(catalog["search_text"].fillna("").astype(str).tolist())
    if len(texts) == 0:
        return np.zeros(len(catalog))

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
    return {token.lower() for token in split_tokens(value) if token}


def hybrid_recommendations(
    catalog: pd.DataFrame,
    selected_idx: int,
    query_boost: str,
    weights: Tuple[float, float, float, float],
    use_neural: bool,
) -> pd.DataFrame:
    if catalog.empty:
        return pd.DataFrame()

    selected = catalog.iloc[selected_idx]
    query = f"{selected.get('title', '')} {selected.get('overview', '')} {selected.get('genres', '')} {selected.get('people_names', '')} {query_boost}"
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
    roi_ceiling = catalog["roi"].max() if len(catalog) <= 20 else catalog["roi"].quantile(0.95)
    business = (
        0.45 * minmax(catalog["popularity"])
        + 0.35 * minmax(catalog["business_value"])
        + 0.20 * minmax(catalog["roi"].clip(lower=0, upper=roi_ceiling))
    )

    w_semantic, w_graph, w_quality, w_business = weights
    score = w_semantic * sem + w_graph * graph + w_quality * quality + w_business * business

    output = catalog.copy()
    output["semantic_score"] = sem
    output["graph_score"] = graph
    output["quality_score"] = quality
    output["business_score"] = business
    output["hybrid_score"] = score
    output = output[output.index != selected_idx].sort_values("hybrid_score", ascending=False).head(20)
    return output


def graph_figure(catalog: pd.DataFrame, center: pd.Series, recommendations: pd.DataFrame, people: pd.DataFrame) -> go.Figure:
    center_title = str(center["title"])
    people_lookup = (
        dict(
            zip(
                pd.to_numeric(people.get("tmdb_id", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int),
                people.get("name", pd.Series(dtype=str)).astype(str),
            )
        )
        if not people.empty
        else {}
    )

    nodes: dict[str, dict[str, object]] = {
        center_title: {"kind": "center", "x": 0.0, "y": 0.0}
    }
    edges: list[tuple[str, str]] = []
    recommendation_rows = list(recommendations.head(7).iterrows())
    if recommendation_rows:
        angle_step = (2 * math.pi) / len(recommendation_rows)
    else:
        angle_step = 2 * math.pi

    for index, (_, row) in enumerate(recommendation_rows):
        title = str(row["title"])
        angle = index * angle_step - math.pi / 2
        title_x = round(math.cos(angle) * 1.0, 4)
        title_y = round(math.sin(angle) * 1.0, 4)
        nodes[title] = {"kind": str(row.get("media_type", "title")), "x": title_x, "y": title_y}
        edges.append((center_title, title))

        for genre_index, genre in enumerate(split_tokens(row.get("genres", ""), limit=2)):
            offset = -0.18 if genre_index == 0 else 0.18
            genre_angle = angle + offset
            genre_key = f"{title}::genre::{genre}"
            nodes[genre_key] = {
                "kind": "genre",
                "label": genre,
                "x": round(math.cos(genre_angle) * 1.55, 4),
                "y": round(math.sin(genre_angle) * 1.55, 4),
            }
            edges.append((title, genre_key))

        for person_index, person_id in enumerate(parse_id_list(row.get("people_ids", ""))[:2]):
            person_name = people_lookup.get(person_id)
            if not person_name:
                continue
            offset = -0.1 if person_index == 0 else 0.1
            person_angle = angle + offset
            person_key = f"{title}::person::{person_name}"
            nodes[person_key] = {
                "kind": "person",
                "label": person_name,
                "x": round(math.cos(person_angle) * 1.35, 4),
                "y": round(math.sin(person_angle) * 1.35, 4),
            }
            edges.append((title, person_key))

    edge_x, edge_y = [], []
    for source, target in edges:
        x0, y0 = float(nodes[source]["x"]), float(nodes[source]["y"])
        x1, y1 = float(nodes[target]["x"]), float(nodes[target]["y"])
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    color_map = {
        "center": "#7DE6FF",
        "genre": "#D9C48B",
        "person": "#16C6C9",
        "movie": "#0E8F98",
        "tv": "#1299A8",
        "orphan_movie": "#92A4A9",
        "orphan_tv": "#92A4A9",
    }

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node_id, attrs in nodes.items():
        x_pos, y_pos = float(attrs["x"]), float(attrs["y"])
        kind = str(attrs.get("kind", "movie"))
        node_x.append(x_pos)
        node_y.append(y_pos)
        node_text.append(str(attrs.get("label", node_id)))
        node_color.append(color_map.get(kind, "#1299A8"))
        node_size.append(30 if kind == "center" else 19 if kind == "genre" else 23)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.15, color="rgba(255,255,255,.24)"),
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
            hoverinfo="text",
        )
    )
    fig.update_layout(
        height=610,
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def explain(row: pd.Series) -> str:
    reasons = []
    if row.get("semantic_score", 0) > 0.18:
        reasons.append("it is semantically close to the base title or the search intent")
    if row.get("graph_score", 0) > 0.12:
        reasons.append("it sits near the source node through recommendation links, genres, or shared people")
    if row.get("quality_score", 0) > 0.55:
        reasons.append("it carries a strong quality signal from ratings and vote depth")
    if row.get("business_score", 0) > 0.55:
        reasons.append("it shows commercial strength through popularity, revenue, ROI, or audience scale")
    if not reasons:
        reasons.append("it remains a weaker exploratory discovery that still belongs to the same catalog neighborhood")
    return "This recommendation surfaced because " + "; ".join(reasons) + "."


def format_int(value: object) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return "0"


def format_pct(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "0.0%"


def format_billions(value: object) -> str:
    try:
        return f"${float(value) / 1_000_000_000:.1f}B"
    except (TypeError, ValueError):
        return "$0.0B"


def build_title_labels(df: pd.DataFrame) -> pd.Series:
    return (
        df["title"].fillna("Untitled")
        + " | "
        + df["media_type"].astype(str)
        + " | #"
        + df["tmdb_id"].astype(str)
    )


def metric_card(label: str, value: str, helper: str = "") -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='small-muted'>{helper}</div></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    catalog, tables = load_catalog()
    if catalog.empty:
        st.error("Real CineGraph files were not found. Add the CSVs to data/raw/ or allow the KaggleHub fallback download.")
        st.stop()

    summaries = load_summary_exports()
    executive_summary = summaries["executive_summary"]
    relationship_coverage = summaries["relationship_coverage"]
    genre_financial_summary = summaries["genre_financial_summary"]
    streaming_coverage = summaries["streaming_coverage"]
    top_people_network = summaries["top_people_network_centrality"]

    processed_ready = (PROCESSED / "catalog_nodes.csv").exists()
    raw_files = find_raw_files()
    runtime_mode = "Processed title layer" if processed_ready else "Raw/Kaggle runtime layer"
    available_media_types = sorted(catalog["media_type"].dropna().astype(str).unique().tolist())
    default_media = [media for media in ["movie", "tv"] if media in available_media_types] or available_media_types

    slider_max = max(1, len(catalog))
    slider_min = 1 if slider_max < 1000 else 1000
    slider_default = min(12000, slider_max)
    slider_step = 1 if slider_max < 1000 else 500

    st.sidebar.markdown("### CineGraph AI Explorer")
    st.sidebar.caption("Graph intelligence, semantic retrieval, and explainable recommendations on a real entertainment catalog.")
    page = st.sidebar.radio(
        "Navigation",
        [
            "1. Executive Overview",
            "2. Asset Explorer",
            "3. Hybrid Recommender",
            "4. Network Graph",
            "5. Semantic Search",
            "6. Decision Explainer",
        ],
    )
    media_filter = st.sidebar.multiselect("Media types", available_media_types, default=default_media)
    max_records = st.sidebar.slider("Titles in active retrieval set", slider_min, slider_max, slider_default, slider_step)
    use_neural = st.sidebar.toggle(
        "Use neural embeddings",
        value=False,
        help="Enables MiniLM if sentence-transformers is available. TF-IDF remains the fast explainable fallback.",
    )

    filtered = catalog[catalog["media_type"].isin(media_filter)] if media_filter else catalog
    filtered = filtered.sort_values(["popularity", "vote_count"], ascending=False).head(max_records).reset_index(drop=True)
    if filtered.empty:
        st.warning("No titles remain after the current media-type filters. Expand the filters to continue.")
        st.stop()

    people = tables.get("people", pd.DataFrame())
    executive_row = executive_summary.iloc[0] if not executive_summary.empty else None
    vector_mode = "MiniLM embeddings" if use_neural and HAS_SENTENCE_TRANSFORMERS and load_embedding_model() is not None else "TF-IDF fallback"

    st.markdown(
        """
        <div class='hero-card'>
          <div class='eyebrow'>Real TMDB catalog · graph intelligence · semantic retrieval · hybrid recommendations · explainable AI</div>
          <h1>CineGraph AI Explorer</h1>
          <p style='font-size:1.18rem;max-width:1010px;'>A production-style media intelligence prototype that turns movies, TV, people, orphan recommendation nodes, and review text into an interactive discovery layer for retrieval, recommendation, graph exploration, and explainable ranking.</p>
          <span class='pill'>graph-aware catalog</span><span class='pill'>semantic search</span><span class='pill'>hybrid scoring</span><span class='pill'>orphan node recovery</span><span class='pill'>review NLP</span><span class='pill'>streaming footprint</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    runtime_col1, runtime_col2, runtime_col3, runtime_col4 = st.columns(4)
    with runtime_col1:
        metric_card("Runtime mode", "Processed" if processed_ready else "Runtime", runtime_mode)
    with runtime_col2:
        metric_card("Retrieval set", format_int(len(filtered)), "Titles currently indexed in-app")
    with runtime_col3:
        metric_card("Raw sources", format_int(len(raw_files)), "CSV files discovered locally or via Kaggle")
    with runtime_col4:
        metric_card("Search mode", "Neural" if vector_mode.startswith("MiniLM") else "TF-IDF", vector_mode)

    if page == "1. Executive Overview":
        st.markdown("## Executive Overview")
        top_profit = genre_financial_summary.iloc[0]["total_profit"] if not genre_financial_summary.empty else 0
        orphan_nodes = 0
        if executive_row is not None and "orphan_movies" in executive_row and "orphan_tv" in executive_row:
            orphan_nodes = float(executive_row["orphan_movies"]) + float(executive_row["orphan_tv"])

        metric_row = st.columns(4)
        with metric_row[0]:
            metric_card("Movies", format_int(executive_row["movies"] if executive_row is not None else (catalog["media_type"] == "movie").sum()), "Primary movie nodes")
        with metric_row[1]:
            metric_card("TV shows", format_int(executive_row["tv_shows"] if executive_row is not None else (catalog["media_type"] == "tv").sum()), "Primary TV nodes")
        with metric_row[2]:
            metric_card("Orphan nodes", format_int(orphan_nodes), "Recommendation-only recovery layer")
        with metric_row[3]:
            metric_card("Top genre profit", format_billions(top_profit), "Highest total profit bucket")

        top_row_left, top_row_right = st.columns([1.1, 0.9])
        with top_row_left:
            if not genre_financial_summary.empty:
                genre_df = genre_financial_summary.head(12).sort_values("total_profit", ascending=True).copy()
                genre_df["label"] = genre_df["genre"] + " · " + genre_df["movies"].astype(int).astype(str) + " films"
                fig = px.bar(
                    genre_df,
                    x="total_profit",
                    y="label",
                    orientation="h",
                    color="median_roi",
                    color_continuous_scale=["#7DE6FF", "#1299A8", "#0A5B63"],
                )
                fig.update_xaxes(title="Total profit (USD)")
                fig.update_yaxes(title="")
                fig.update_layout(height=520, coloraxis_colorbar_title="Median ROI", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.markdown("### Genre profit concentration")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Genre financial export not found.")

        with top_row_right:
            if not relationship_coverage.empty:
                coverage_df = relationship_coverage.copy().sort_values("coverage_pct", ascending=True)
                coverage_df["relationship"] = coverage_df["relationship"].str.replace("_", " ", regex=False).str.title()
                fig = px.bar(
                    coverage_df,
                    x="coverage_pct",
                    y="relationship",
                    orientation="h",
                    color="coverage_pct",
                    color_continuous_scale=["#174A54", "#0E7C86", "#7DE6FF"],
                )
                fig.update_xaxes(title="Resolved coverage %", range=[0, 100])
                fig.update_yaxes(title="")
                fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
                st.markdown("### Relationship integrity")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Coverage exports are not available yet.")

        lower_left, lower_right = st.columns([1.05, 0.95])
        with lower_left:
            st.markdown("### Streaming availability footprint")
            if not streaming_coverage.empty:
                streaming_matrix = streaming_coverage.copy()
                streaming_matrix["bucket"] = streaming_matrix["media"].str.upper() + " · " + streaming_matrix["region"] + " · " + streaming_matrix["mode"]
                fig = px.bar(
                    streaming_matrix.sort_values("coverage_pct", ascending=False),
                    x="coverage_pct",
                    y="bucket",
                    orientation="h",
                    color="media",
                    color_discrete_sequence=["#7DE6FF", "#1299A8"],
                )
                fig.update_xaxes(title="Coverage %")
                fig.update_yaxes(title="")
                fig.update_layout(height=430, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Streaming coverage export not found.")
        with lower_right:
            st.markdown("### Build notes")
            st.markdown(
                """
                <div class='section-card'>
                <p><strong>Lakehouse-style thinking:</strong> raw CSV ingestion, processed node/edge exports, and summary-serving tables already map well to Bronze, Silver, and Gold-style layers.</p>
                <p><strong>Graph-aware modeling:</strong> orphan nodes are preserved to avoid dangling recommendation edges and keep retrieval paths intact.</p>
                <p><strong>Search architecture:</strong> the app supports a neural route with MiniLM and an interpretable TF-IDF fallback for constrained environments.</p>
                <p><strong>Next production steps:</strong> persist embeddings, precompute graph features, materialize recommendation candidates, and add a proper vector-serving layer.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Network centrality snapshot")
        if not top_people_network.empty:
            st.dataframe(top_people_network.head(12)[["name", "weighted_degree", "tmdb_id"]], use_container_width=True, hide_index=True)
        else:
            st.info("Top people network export not found.")

    elif page == "2. Asset Explorer":
        st.markdown("## Asset Explorer")
        query = st.text_input(
            "Search by title, genre, person, keyword, or synopsis",
            placeholder="Examples: cyberpunk thriller, Christopher Nolan, newsroom conspiracy, psychological drama",
        )
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 0.0, 0.1)
        explorer = filtered[pd.to_numeric(filtered["rating"], errors="coerce").fillna(0) >= min_rating].copy()
        if explorer.empty:
            st.info("No titles match the current explorer filters.")
            return

        if query:
            scores = semantic_scores(query, explorer, use_neural)
            results = explorer.copy()
            results["match_score"] = scores
            results = results.sort_values("match_score", ascending=False).head(30)
        else:
            results = explorer.sort_values(["popularity", "vote_count"], ascending=False).head(30)

        st.dataframe(
            results[
                ["title", "media_type", "release_year", "genres", "people_names", "rating", "vote_count", "popularity"]
                + (["match_score"] if "match_score" in results.columns else [])
            ],
            use_container_width=True,
            hide_index=True,
        )

    elif page == "3. Hybrid Recommender":
        st.markdown("## Hybrid Recommender")
        names = build_title_labels(filtered)
        selected_label = st.selectbox("Choose a base title", names.tolist())
        selected_idx = names[names == selected_label].index[0]
        query_boost = st.text_input("Intent boost", placeholder="Examples: visually ambitious, dystopian, prestige cast, strong reviews")
        weight_cols = st.columns(4)
        weights = [
            weight_cols[0].slider("Semantic", 0.0, 1.0, 0.40, 0.05),
            weight_cols[1].slider("Graph", 0.0, 1.0, 0.30, 0.05),
            weight_cols[2].slider("Quality", 0.0, 1.0, 0.15, 0.05),
            weight_cols[3].slider("Business", 0.0, 1.0, 0.15, 0.05),
        ]
        total = max(sum(weights), 0.01)
        recommendations = hybrid_recommendations(filtered, selected_idx, query_boost, tuple(weight / total for weight in weights), use_neural)
        recommendations["why_it_ranked"] = recommendations.apply(explain, axis=1)
        st.dataframe(
            recommendations[
                ["title", "media_type", "release_year", "genres", "rating", "hybrid_score", "semantic_score", "graph_score", "quality_score", "business_score", "why_it_ranked"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    elif page == "4. Network Graph":
        st.markdown("## Network Graph")
        names = build_title_labels(filtered)
        selected_label = st.selectbox("Central node", names.tolist(), key="network")
        selected_idx = names[names == selected_label].index[0]
        recommendations = hybrid_recommendations(filtered, selected_idx, "", (0.40, 0.30, 0.15, 0.15), use_neural)
        figure = graph_figure(filtered, filtered.iloc[selected_idx], recommendations, people)
        st.plotly_chart(figure, use_container_width=True)
        if not people.empty:
            st.markdown("### Highest-popularity people in the dataset")
            st.dataframe(
                people.sort_values("popularity", ascending=False).head(15)[["name", "known_for_dept", "popularity", "known_movies", "known_tv_shows"]],
                use_container_width=True,
                hide_index=True,
            )

    elif page == "5. Semantic Search":
        st.markdown("## Semantic Search")
        description = st.text_area(
            "Describe the content you want to find",
            value="I want a dark, high-quality, psychologically tense story with technology and a strong cast.",
            height=120,
        )
        scores = semantic_scores(description, filtered, use_neural)
        results = filtered.copy()
        results["semantic_score"] = scores
        results = results.sort_values("semantic_score", ascending=False).head(30)
        st.dataframe(results[["title", "media_type", "release_year", "genres", "overview", "semantic_score"]], use_container_width=True, hide_index=True)

    elif page == "6. Decision Explainer":
        st.markdown("## Decision Explainer")
        names = build_title_labels(filtered)
        selected_label = st.selectbox("Base title", names.tolist(), key="explain")
        selected_idx = names[names == selected_label].index[0]
        recommendations = hybrid_recommendations(filtered, selected_idx, "", (0.40, 0.30, 0.15, 0.15), use_neural)
        choice = st.selectbox("Recommendation to unpack", recommendations["title"].tolist())
        row = recommendations[recommendations["title"] == choice].iloc[0]

        st.markdown(
            f"""
            <div class='section-card'>
              <h3>{row['title']}</h3>
              <p>{row.get('overview', '')}</p>
              <div class='why-box'><strong>Why it appeared</strong><p>{explain(row)}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        score_cards = st.columns(4)
        with score_cards[0]:
            metric_card("Semantic", format_pct(row["semantic_score"] * 100, 0), "Intent and synopsis proximity")
        with score_cards[1]:
            metric_card("Graph", format_pct(row["graph_score"] * 100, 0), "Recommendation and entity overlap")
        with score_cards[2]:
            metric_card("Quality", format_pct(row["quality_score"] * 100, 0), "Rating and vote confidence")
        with score_cards[3]:
            metric_card("Business", format_pct(row["business_score"] * 100, 0), "Popularity and revenue strength")

        score_df = pd.DataFrame(
            {
                "component": ["semantic", "graph", "quality", "business"],
                "score": [row["semantic_score"], row["graph_score"], row["quality_score"], row["business_score"]],
            }
        )
        figure = px.bar(
            score_df,
            x="component",
            y="score",
            color="component",
            color_discrete_sequence=["#7DE6FF", "#1299A8", "#D9C48B", "#16C6C9"],
        )
        figure.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, use_container_width=True)


if __name__ == "__main__":
    main()
