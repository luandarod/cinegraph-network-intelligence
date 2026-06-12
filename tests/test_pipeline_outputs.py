from __future__ import annotations

from pathlib import Path

import pandas as pd

from cinegraph_network_intelligence.pipeline import run_pipeline


REQUIRED_FILES = {
    "executive_summary.csv",
    "relationship_coverage.csv",
    "genre_financial_summary.csv",
    "streaming_coverage.csv",
    "low_review_terms.csv",
    "high_review_terms.csv",
    "top_people_network_centrality.csv",
    "processed/catalog_nodes.csv",
    "processed/graph_edges.csv",
    "processed/run_metadata.json",
}


def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "raw"


def test_pipeline_writes_processed_and_summary_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_pipeline(fixture_dir(), output_dir)

    produced = {
        str(path.relative_to(output_dir)).replace("\\", "/")
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert REQUIRED_FILES.issubset(produced)


def test_processed_catalog_contains_expected_media_types(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_pipeline(fixture_dir(), output_dir)

    catalog = pd.read_csv(output_dir / "processed" / "catalog_nodes.csv")
    assert {"movie", "tv", "orphan_movie", "orphan_tv"}.issubset(set(catalog["media_type"]))


def test_graph_edges_and_summaries_have_expected_columns(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_pipeline(fixture_dir(), output_dir)

    edges = pd.read_csv(output_dir / "processed" / "graph_edges.csv")
    executive = pd.read_csv(output_dir / "executive_summary.csv")
    coverage = pd.read_csv(output_dir / "relationship_coverage.csv")

    assert {"source_node_id", "target_node_id", "edge_type", "source_media_type", "target_media_type"}.issubset(edges.columns)
    assert {"movies", "tv_shows", "people", "movie_recommendation_edges", "tv_recommendation_edges"}.issubset(executive.columns)
    assert {"relationship", "coverage_pct"}.issubset(coverage.columns)
