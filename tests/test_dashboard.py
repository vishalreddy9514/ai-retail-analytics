"""Runs the Streamlit app headlessly with small synthetic data and a fake Ollama."""
import io
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from retail_analytics.forecasting import run_forecast, save_outputs  # noqa: E402
from test_assistant import gold  # noqa: E402,F401  (reuse the synthetic Gold fixture)

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


@pytest.fixture
def data_dir(gold, tmp_path, monkeypatch):  # noqa: F811
    (tmp_path / "exports").mkdir()
    gold.to_csv(tmp_path / "exports" / "gold_daily_sales.csv", index=False)
    save_outputs(run_forecast(gold[["sales_date", "revenue"]], test_days=14), tmp_path / "outputs")
    monkeypatch.setenv("RETAIL_DATA_DIR", str(tmp_path))
    return tmp_path


def test_dashboard_renders_metrics_charts_and_forecast(data_dir):
    app = AppTest.from_file(APP, default_timeout=60).run()

    assert not app.exception
    labels = [m.label for m in app.metric]
    assert {"Total revenue", "Orders", "Avg revenue per trading day", "Forecast total"} <= set(labels)
    assert len(app.dataframe) == 2  # next-7-days table and model scores


def test_explain_button_shows_ai_answer(data_dir, monkeypatch):
    def fake_urlopen(request, timeout):
        return io.BytesIO(json.dumps({"message": {"content": "The forecast total is shown above."}}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    app = AppTest.from_file(APP, default_timeout=60).run()
    app.button[0].click().run()

    assert not app.exception
    assert any("The forecast total is shown above." in m.value for m in app.markdown)


def test_missing_gold_file_shows_instructions(tmp_path, monkeypatch):
    monkeypatch.setenv("RETAIL_DATA_DIR", str(tmp_path))
    app = AppTest.from_file(APP, default_timeout=60).run()
    assert "not found" in app.error[0].value
