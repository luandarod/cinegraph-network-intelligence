"""
CineGraph Network Intelligence

Graph analytics, relationship coverage, financial exploration and review text mining
for the CineGraph TMDB dataset.
"""

from pathlib import Path
import itertools
import re

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_id_list(value):
    """Parse comma-separated TMDB IDs stored inside one cell."""
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [
        int(token.strip())
        for token in str(value).split(",")
        if re.match(r"^-?\d+$", token.strip())
    ]


def explode_edges(df, ids_col, source_col="tmdb_id", target_col="target_id"):
    """Create an edge list from a comma-separated ID column."""
    edges = df[[source_col, ids_col]].dropna().copy()
    edges[ids_col] = edges[ids_col].apply(parse_id_list)
    edges = edges.explode(ids_col).dropna()
    edges = edges.rename(columns={source_col: "source_id", ids_col: target_col})
    edges[target_col] = edges[target_col].astype(int)
    return edges


def build_costar_network(movies, people_ids, top_n_movies=2500, cast_depth=8):
    """Build a sampled co-star network from high-vote movies."""
    graph = nx.Graph()
    top_movies = movies.sort_values(["vote_count", "popularity"], ascending=False).head(top_n_movies)

    for _, row in top_movies.dropna(subset=["cast_ids"]).iterrows():
        cast = parse_id_list(row["cast_ids"])[:cast_depth]
        cast = [person_id for person_id in cast if person_id in people_ids]

        for person_id in cast:
            graph.add_node(person_id)

        for actor_a, actor_b in itertools.combinations(cast, 2):
            if graph.has_edge(actor_a, actor_b):
                graph[actor_a][actor_b]["weight"] += 1
            else:
                graph.add_edge(actor_a, actor_b, weight=1)

    return graph


def top_review_terms(reviews, low_threshold=4, high_threshold=8):
    """Compare terms overrepresented in low-rated and high-rated reviews."""
    reviews = reviews.dropna(subset=["rating"]).copy()
    reviews["content"] = reviews["content"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=8000,
        ngram_range=(1, 2),
        min_df=8,
    )

    matrix = vectorizer.fit_transform(reviews["content"])
    terms = np.array(vectorizer.get_feature_names_out())

    low_mask = reviews["rating"] <= low_threshold
    high_mask = reviews["rating"] >= high_threshold

    low_mean = np.asarray(matrix[low_mask].mean(axis=0)).ravel()
    high_mean = np.asarray(matrix[high_mask].mean(axis=0)).ravel()
    gap = low_mean - high_mean

    low_terms = (
        pd.DataFrame({
            "term": terms,
            "tfidf_gap": gap,
            "low_score": low_mean,
            "high_score": high_mean,
        })
        .sort_values("tfidf_gap", ascending=False)
        .head(30)
    )

    high_terms = (
        pd.DataFrame({
            "term": terms,
            "tfidf_gap": -gap,
            "low_score": low_mean,
            "high_score": high_mean,
        })
        .sort_values("tfidf_gap", ascending=False)
        .head(30)
    )

    return low_terms, high_terms


def main(input_dir, output_dir="outputs"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    movies = pd.read_csv(input_path / "movies.csv", encoding="utf-8-sig")
    tv = pd.read_csv(input_path / "tv_shows.csv", encoding="utf-8-sig")
    people = pd.read_csv(input_path / "people.csv", encoding="utf-8-sig")
    orphan_movies = pd.read_csv(input_path / "orphan_movies.csv", encoding="utf-8-sig")
    orphan_tv = pd.read_csv(input_path / "orphan_tv.csv", encoding="utf-8-sig")
    movie_reviews = pd.read_csv(input_path / "movie_reviews.csv", encoding="utf-8-sig")
    tv_reviews = pd.read_csv(input_path / "tv_reviews.csv", encoding="utf-8-sig")

    people_ids = set(people["tmdb_id"])
    movie_ids = set(movies["tmdb_id"])
    tv_ids = set(tv["tmdb_id"])
    movie_graph_ids = movie_ids | set(orphan_movies["tmdb_id"])
    tv_graph_ids = tv_ids | set(orphan_tv["tmdb_id"])

    movie_cast = explode_edges(movies, "cast_ids", target_col="person_id")
    tv_cast = explode_edges(tv, "cast_ids", target_col="person_id")
    movie_directors = explode_edges(movies, "director_ids", target_col="person_id")
    tv_creators = explode_edges(tv, "creator_ids", target_col="person_id")
    movie_recs = explode_edges(movies, "recommended_ids", target_col="target_id")
    tv_recs = explode_edges(tv, "recommended_ids", target_col="target_id")

    relationship_coverage = pd.DataFrame([
        {"relationship": "movie_recommendations_main_only", "coverage_pct": movie_recs["target_id"].isin(movie_ids).mean() * 100},
        {"relationship": "movie_recommendations_with_orphans", "coverage_pct": movie_recs["target_id"].isin(movie_graph_ids).mean() * 100},
        {"relationship": "tv_recommendations_main_only", "coverage_pct": tv_recs["target_id"].isin(tv_ids).mean() * 100},
        {"relationship": "tv_recommendations_with_orphans", "coverage_pct": tv_recs["target_id"].isin(tv_graph_ids).mean() * 100},
        {"relationship": "movie_cast_to_people", "coverage_pct": movie_cast["person_id"].isin(people_ids).mean() * 100},
        {"relationship": "movie_directors_to_people", "coverage_pct": movie_directors["person_id"].isin(people_ids).mean() * 100},
        {"relationship": "tv_creators_to_people", "coverage_pct": tv_creators["person_id"].isin(people_ids).mean() * 100},
    ])

    financial_movies = movies[(movies["budget_usd"].fillna(0) > 0) & (movies["revenue_usd"].fillna(0) > 0)].copy()
    genre_financial = financial_movies.dropna(subset=["genres"]).copy()
    genre_financial["genre"] = genre_financial["genres"].str.split(",")
    genre_financial = genre_financial.explode("genre")
    genre_financial["genre"] = genre_financial["genre"].str.strip()

    genre_summary = (
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
        .query("movies >= 100")
        .sort_values("total_profit", ascending=False)
    )

    streaming_rows = []
    for media_name, df in {"movies": movies, "tv": tv}.items():
        for column in [
            "watch_tr_flatrate", "watch_tr_rent", "watch_tr_buy", "watch_tr_free",
            "watch_us_flatrate", "watch_us_rent", "watch_us_buy", "watch_us_free",
        ]:
            region = column.split("_")[1].upper()
            mode = column.split("_")[2]
            streaming_rows.append({
                "media": media_name,
                "region": region,
                "mode": mode,
                "available_count": int(df[column].notna().sum()),
                "coverage_pct": df[column].notna().mean() * 100,
            })

    graph = build_costar_network(movies, people_ids)
    weighted_degree = sorted(graph.degree(weight="weight"), key=lambda item: item[1], reverse=True)
    people_names = people.set_index("tmdb_id")["name"].to_dict()

    top_people = pd.DataFrame({
        "tmdb_id": [person_id for person_id, _ in weighted_degree[:20]],
        "weighted_degree": [score for _, score in weighted_degree[:20]],
    })
    top_people["name"] = top_people["tmdb_id"].map(people_names)

    reviews = pd.concat([movie_reviews, tv_reviews], ignore_index=True)
    low_terms, high_terms = top_review_terms(reviews)

    executive_summary = pd.DataFrame([{
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
        "median_movie_roi_pct": financial_movies["roi_pct"].median(),
        "movie_review_rating_coverage_pct": movie_reviews["rating"].notna().mean() * 100,
        "tv_review_rating_coverage_pct": tv_reviews["rating"].notna().mean() * 100,
    }])

    executive_summary.to_csv(output_path / "executive_summary.csv", index=False)
    relationship_coverage.to_csv(output_path / "relationship_coverage.csv", index=False)
    genre_summary.to_csv(output_path / "genre_financial_summary.csv", index=False)
    pd.DataFrame(streaming_rows).to_csv(output_path / "streaming_coverage.csv", index=False)
    low_terms.to_csv(output_path / "low_review_terms.csv", index=False)
    high_terms.to_csv(output_path / "high_review_terms.csv", index=False)
    top_people.to_csv(output_path / "top_people_network_centrality.csv", index=False)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "outputs")
