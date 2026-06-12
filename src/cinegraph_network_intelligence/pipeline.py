from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


RAW_FILES = {
    "movies": "movies.csv",
    "tv": "tv_shows.csv",
    "people": "people.csv",
    "orphan_movies": "orphan_movies.csv",
    "orphan_tv": "orphan_tv.csv",
    "movie_reviews": "movie_reviews.csv",
    "tv_reviews": "tv_reviews.csv",
}


def parse_id_list(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    ids: list[int] = []
    for part in re.split(r"[,;|]", text):
        token = part.strip()
        if not token:
            continue
        try:
            ids.append(int(float(token)))
        except ValueError:
            continue
    return ids


def split_tokens(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return [token.strip() for token in re.split(r"[,;|]", text) if token.strip()]


def read_raw_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    tables: dict[str, pd.DataFrame] = {}
    for key, filename in RAW_FILES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        tables[key] = pd.read_csv(path, low_memory=False)
    return tables


def normalize_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return df


def normalize_text(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
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


def build_catalog_nodes(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

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
    for column in [
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
    ]:
        if column not in catalog.columns:
            catalog[column] = ""
    catalog = normalize_numeric(catalog, ["rating", "vote_count", "popularity", "business_value", "roi", "release_year"])
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
    catalog["people_names"] = (
        catalog["cast_names"].fillna("") + ", " +
        catalog.get("directors", "").fillna("") + ", " +
        catalog.get("creators", "").fillna("")
    ).str.strip(", ")
    catalog["people_ids"] = (
        catalog["cast_ids"].fillna("") + "," +
        catalog.get("director_ids", "").fillna("") + "," +
        catalog.get("creator_ids", "").fillna("")
    ).str.strip(",")
    catalog["search_text"] = (
        catalog["title"].fillna("") + " " +
        catalog["overview"].fillna("") + " " +
        catalog["genres"].fillna("") + " " +
        catalog["keywords"].fillna("") + " " +
        catalog["people_names"].fillna("")
    ).str.strip()
    catalog = catalog.dropna(subset=["title"]).drop_duplicates(subset=["media_type", "tmdb_id"])
    return catalog.reset_index(drop=True)


def media_node_lookup(catalog: pd.DataFrame) -> dict[int, tuple[str, str]]:
    lookup: dict[int, tuple[str, str]] = {}
    for _, row in catalog.iterrows():
        lookup[int(row["tmdb_id"])] = (str(row["node_id"]), str(row["media_type"]))
    return lookup


def build_graph_edges(catalog: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    media_lookup = media_node_lookup(catalog)
    people_lookup = {
        int(row["tmdb_id"]): (f"person::{int(row['tmdb_id'])}", "person")
        for _, row in people.iterrows()
    }

    edge_rows: list[dict[str, object]] = []
    for _, row in catalog.iterrows():
        source_node_id = str(row["node_id"])
        source_media_type = str(row["media_type"])

        for target_id in parse_id_list(row.get("similar_ids", "")):
            target = media_lookup.get(target_id)
            if target:
                edge_rows.append(
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target[0],
                        "edge_type": "similar_title",
                        "source_media_type": source_media_type,
                        "target_media_type": target[1],
                    }
                )

        for target_id in parse_id_list(row.get("recommended_ids", "")):
            target = media_lookup.get(target_id)
            if target:
                edge_rows.append(
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target[0],
                        "edge_type": "recommended_title",
                        "source_media_type": source_media_type,
                        "target_media_type": target[1],
                    }
                )

        for person_id in parse_id_list(row.get("cast_ids", "")):
            target = people_lookup.get(person_id)
            if target:
                edge_rows.append(
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target[0],
                        "edge_type": "cast_person",
                        "source_media_type": source_media_type,
                        "target_media_type": target[1],
                    }
                )

        for person_id in parse_id_list(row.get("director_ids", "")):
            target = people_lookup.get(person_id)
            if target:
                edge_rows.append(
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target[0],
                        "edge_type": "director_person",
                        "source_media_type": source_media_type,
                        "target_media_type": target[1],
                    }
                )

        for person_id in parse_id_list(row.get("creator_ids", "")):
            target = people_lookup.get(person_id)
            if target:
                edge_rows.append(
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target[0],
                        "edge_type": "creator_person",
                        "source_media_type": source_media_type,
                        "target_media_type": target[1],
                    }
                )

    edges = pd.DataFrame(edge_rows).drop_duplicates()
    return edges


def explode_edges(df: pd.DataFrame, ids_col: str, source_col: str = "tmdb_id", target_col: str = "target_id") -> pd.DataFrame:
    edges = df[[source_col, ids_col]].dropna().copy()
    if edges.empty:
        return pd.DataFrame(columns=["source_id", target_col])
    edges[ids_col] = edges[ids_col].apply(parse_id_list)
    edges = edges.explode(ids_col).dropna()
    edges = edges.rename(columns={source_col: "source_id", ids_col: target_col})
    edges[target_col] = edges[target_col].astype(int)
    return edges


def build_costar_strength(movies: pd.DataFrame, people_ids: set[int], top_n_movies: int = 2500, cast_depth: int = 8) -> dict[int, int]:
    pair_weights: dict[tuple[int, int], int] = {}
    top_movies = movies.sort_values(["vote_count", "popularity"], ascending=False).head(top_n_movies)
    for _, row in top_movies.dropna(subset=["cast_ids"]).iterrows():
        cast = parse_id_list(row["cast_ids"])[:cast_depth]
        cast = [person_id for person_id in cast if person_id in people_ids]
        for actor_a, actor_b in itertools.combinations(cast, 2):
            pair = tuple(sorted((actor_a, actor_b)))
            pair_weights[pair] = pair_weights.get(pair, 0) + 1

    weighted_degree: dict[int, int] = {}
    for (actor_a, actor_b), weight in pair_weights.items():
        weighted_degree[actor_a] = weighted_degree.get(actor_a, 0) + weight
        weighted_degree[actor_b] = weighted_degree.get(actor_b, 0) + weight
    return weighted_degree


def top_review_terms(reviews: pd.DataFrame, low_threshold: int = 4, high_threshold: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    reviews = reviews.dropna(subset=["rating"]).copy()
    reviews["content"] = reviews["content"].fillna("").astype(str)
    if reviews.empty:
        empty = pd.DataFrame(columns=["term", "tfidf_gap", "low_score", "high_score"])
        return empty, empty

    min_df = 8 if len(reviews) >= 50 else 1
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
        min_df=min_df,
    )

    matrix = vectorizer.fit_transform(reviews["content"])
    terms = np.array(vectorizer.get_feature_names_out())

    low_mask = (reviews["rating"] <= low_threshold).to_numpy()
    high_mask = (reviews["rating"] >= high_threshold).to_numpy()

    low_mean = np.asarray(matrix[low_mask].mean(axis=0)).ravel() if low_mask.any() else np.zeros(len(terms))
    high_mean = np.asarray(matrix[high_mask].mean(axis=0)).ravel() if high_mask.any() else np.zeros(len(terms))
    gap = low_mean - high_mean

    low_terms = (
        pd.DataFrame({"term": terms, "tfidf_gap": gap, "low_score": low_mean, "high_score": high_mean})
        .sort_values("tfidf_gap", ascending=False)
        .head(30)
    )
    high_terms = (
        pd.DataFrame({"term": terms, "tfidf_gap": -gap, "low_score": low_mean, "high_score": high_mean})
        .sort_values("tfidf_gap", ascending=False)
        .head(30)
    )
    return low_terms, high_terms


def relationship_coverage_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    people_ids = set(pd.to_numeric(tables["people"]["tmdb_id"], errors="coerce").dropna().astype(int))
    movie_ids = set(pd.to_numeric(tables["movies"]["tmdb_id"], errors="coerce").dropna().astype(int))
    tv_ids = set(pd.to_numeric(tables["tv"]["tmdb_id"], errors="coerce").dropna().astype(int))
    movie_graph_ids = movie_ids | set(pd.to_numeric(tables["orphan_movies"]["tmdb_id"], errors="coerce").dropna().astype(int))
    tv_graph_ids = tv_ids | set(pd.to_numeric(tables["orphan_tv"]["tmdb_id"], errors="coerce").dropna().astype(int))

    movie_cast = explode_edges(tables["movies"], "cast_ids", target_col="person_id")
    movie_directors = explode_edges(tables["movies"], "director_ids", target_col="person_id")
    tv_creators = explode_edges(tables["tv"], "creator_ids", target_col="person_id")
    movie_recs = explode_edges(tables["movies"], "recommended_ids", target_col="target_id")
    tv_recs = explode_edges(tables["tv"], "recommended_ids", target_col="target_id")

    return pd.DataFrame(
        [
            {"relationship": "movie_recommendations_main_only", "coverage_pct": movie_recs["target_id"].isin(movie_ids).mean() * 100 if not movie_recs.empty else 0},
            {"relationship": "movie_recommendations_with_orphans", "coverage_pct": movie_recs["target_id"].isin(movie_graph_ids).mean() * 100 if not movie_recs.empty else 0},
            {"relationship": "tv_recommendations_main_only", "coverage_pct": tv_recs["target_id"].isin(tv_ids).mean() * 100 if not tv_recs.empty else 0},
            {"relationship": "tv_recommendations_with_orphans", "coverage_pct": tv_recs["target_id"].isin(tv_graph_ids).mean() * 100 if not tv_recs.empty else 0},
            {"relationship": "movie_cast_to_people", "coverage_pct": movie_cast["person_id"].isin(people_ids).mean() * 100 if not movie_cast.empty else 0},
            {"relationship": "movie_directors_to_people", "coverage_pct": movie_directors["person_id"].isin(people_ids).mean() * 100 if not movie_directors.empty else 0},
            {"relationship": "tv_creators_to_people", "coverage_pct": tv_creators["person_id"].isin(people_ids).mean() * 100 if not tv_creators.empty else 0},
        ]
    ).round(4)


def genre_financial_summary(movies: pd.DataFrame) -> pd.DataFrame:
    financial_movies = movies[(pd.to_numeric(movies.get("budget_usd"), errors="coerce").fillna(0) > 0) & (pd.to_numeric(movies.get("revenue_usd"), errors="coerce").fillna(0) > 0)].copy()
    if financial_movies.empty:
        return pd.DataFrame(columns=["genre", "movies", "total_revenue", "total_profit", "median_roi", "avg_rating", "votes"])
    genre_financial = financial_movies.dropna(subset=["genres"]).copy()
    genre_financial["genre"] = genre_financial["genres"].str.split(",")
    genre_financial = genre_financial.explode("genre")
    genre_financial["genre"] = genre_financial["genre"].str.strip()
    summary = (
        genre_financial.groupby("genre")
        .agg(
            movies=("tmdb_id", "nunique"),
            total_revenue=("revenue_usd", "sum"),
            total_profit=("profit_usd", "sum"),
            median_roi=("roi_pct", "median"),
            avg_rating=("vote_average", "mean"),
            votes=("vote_count", "sum"),
        )
        .reset_index()
        .sort_values("total_profit", ascending=False)
    )
    return summary.round(4)


def streaming_coverage_summary(movies: pd.DataFrame, tv: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for media_name, df in {"movies": movies, "tv": tv}.items():
        for column in [
            "watch_tr_flatrate", "watch_tr_rent", "watch_tr_buy", "watch_tr_free",
            "watch_us_flatrate", "watch_us_rent", "watch_us_buy", "watch_us_free",
        ]:
            if column not in df.columns:
                continue
            region = column.split("_")[1].upper()
            mode = column.split("_")[2]
            rows.append(
                {
                    "media": media_name,
                    "region": region,
                    "mode": mode,
                    "available_count": int(df[column].notna().sum()),
                    "coverage_pct": df[column].notna().mean() * 100 if len(df) else 0,
                }
            )
    return pd.DataFrame(rows).round(4)


def top_people_network_summary(movies: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    people_ids = set(pd.to_numeric(people["tmdb_id"], errors="coerce").dropna().astype(int))
    weighted_degree = sorted(build_costar_strength(movies, people_ids).items(), key=lambda item: item[1], reverse=True)
    people_names = people.set_index("tmdb_id")["name"].to_dict()
    top_people = pd.DataFrame(
        {
            "tmdb_id": [person_id for person_id, _ in weighted_degree[:20]],
            "weighted_degree": [score for _, score in weighted_degree[:20]],
        }
    )
    if not top_people.empty:
        top_people["name"] = top_people["tmdb_id"].map(people_names)
    return top_people


def executive_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    movies = tables["movies"]
    tv = tables["tv"]
    people = tables["people"]
    orphan_movies = tables["orphan_movies"]
    orphan_tv = tables["orphan_tv"]
    movie_reviews = tables["movie_reviews"]
    tv_reviews = tables["tv_reviews"]

    movie_cast = explode_edges(movies, "cast_ids", target_col="person_id")
    tv_cast = explode_edges(tv, "cast_ids", target_col="person_id")
    movie_recs = explode_edges(movies, "recommended_ids", target_col="target_id")
    tv_recs = explode_edges(tv, "recommended_ids", target_col="target_id")
    financial_movies = movies[(pd.to_numeric(movies.get("budget_usd"), errors="coerce").fillna(0) > 0) & (pd.to_numeric(movies.get("revenue_usd"), errors="coerce").fillna(0) > 0)].copy()

    summary = pd.DataFrame(
        [
            {
                "movies": len(movies),
                "tv_shows": len(tv),
                "people": len(people),
                "orphan_movies": len(orphan_movies),
                "orphan_tv": len(orphan_tv),
                "movie_reviews": len(movie_reviews),
                "tv_reviews": len(tv_reviews),
                "movie_cast_edges": len(movie_cast),
                "tv_cast_edges": len(tv_cast),
                "movie_recommendation_edges": len(movie_recs),
                "tv_recommendation_edges": len(tv_recs),
                "movies_with_financials": len(financial_movies),
                "median_movie_roi_pct": pd.to_numeric(financial_movies.get("roi_pct"), errors="coerce").median() if not financial_movies.empty else 0,
                "movie_review_rating_coverage_pct": movie_reviews["rating"].notna().mean() * 100 if not movie_reviews.empty else 0,
                "tv_review_rating_coverage_pct": tv_reviews["rating"].notna().mean() * 100 if not tv_reviews.empty else 0,
            }
        ]
    )
    return summary.round(4)


def write_metadata(output_dir: Path, raw_dir: Path, catalog: pd.DataFrame, edges: pd.DataFrame) -> None:
    metadata = {
        "raw_dir": str(raw_dir),
        "catalog_nodes": int(len(catalog)),
        "graph_edges": int(len(edges)),
        "media_types": sorted(catalog["media_type"].dropna().astype(str).unique().tolist()),
        "exports": [
            "executive_summary.csv",
            "relationship_coverage.csv",
            "genre_financial_summary.csv",
            "streaming_coverage.csv",
            "low_review_terms.csv",
            "high_review_terms.csv",
            "top_people_network_centrality.csv",
            "processed/catalog_nodes.csv",
            "processed/graph_edges.csv",
        ],
    }
    (output_dir / "processed" / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run_pipeline(raw_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    tables = read_raw_tables(raw_dir)
    catalog = build_catalog_nodes(tables)
    edges = build_graph_edges(catalog, tables["people"])
    coverage = relationship_coverage_summary(tables)
    genre_summary = genre_financial_summary(tables["movies"])
    streaming_summary = streaming_coverage_summary(tables["movies"], tables["tv"])
    top_people = top_people_network_summary(tables["movies"], tables["people"])
    reviews = pd.concat([tables["movie_reviews"], tables["tv_reviews"]], ignore_index=True)
    low_terms, high_terms = top_review_terms(reviews)
    summary = executive_summary(tables)

    catalog.to_csv(processed_dir / "catalog_nodes.csv", index=False)
    edges.to_csv(processed_dir / "graph_edges.csv", index=False)
    summary.to_csv(output_dir / "executive_summary.csv", index=False)
    coverage.to_csv(output_dir / "relationship_coverage.csv", index=False)
    genre_summary.to_csv(output_dir / "genre_financial_summary.csv", index=False)
    streaming_summary.to_csv(output_dir / "streaming_coverage.csv", index=False)
    low_terms.to_csv(output_dir / "low_review_terms.csv", index=False)
    high_terms.to_csv(output_dir / "high_review_terms.csv", index=False)
    top_people.to_csv(output_dir / "top_people_network_centrality.csv", index=False)
    write_metadata(output_dir, raw_dir, catalog, edges)
