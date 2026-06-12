from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import app.streamlit_app as streamlit_app
from cinegraph_network_intelligence.pipeline import run_pipeline


def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "raw"


def test_streamlit_main_runs_with_processed_fixture_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    run_pipeline(fixture_dir(), data_dir)

    streamlit_app.DATA = data_dir
    streamlit_app.PROCESSED = data_dir / "processed"
    streamlit_app.ROOT = tmp_path
    streamlit_app.load_catalog.clear()
    streamlit_app.load_raw_tables.clear()
    streamlit_app.load_summary_exports.clear()
    streamlit_app.find_raw_files.clear()
    streamlit_app.build_text_vectors.clear()

    app_test = AppTest.from_function(streamlit_app.main)
    app_test.run(timeout=120)

    assert len(app_test.exception) == 0
