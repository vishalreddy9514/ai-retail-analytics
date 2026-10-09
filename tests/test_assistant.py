import io
import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

from retail_analytics import assistant
from retail_analytics.assistant import (
    OllamaError, answer_question, build_context, days_mentioned, load_inputs, unverified_numbers,
)
from retail_analytics.forecasting import run_forecast, save_outputs


@pytest.fixture
def gold():
    dates = pd.date_range("2011-09-01", "2011-11-30")
    trading = dates.dayofweek != 5
    revenue = np.where(trading, 40_000 + 100 * np.arange(len(dates)), 0.0)
    return pd.DataFrame({
        "sales_date": dates,
        "day_of_week": dates.strftime("%a"),
        "is_trading_day": trading,
        "revenue": revenue,
        "orders": np.where(trading, 100, 0),
        "units_sold": np.where(trading, 20_000, 0),
        "unique_customers": np.where(trading, 80, 0),
        "line_items": np.where(trading, 2_000, 0),
    })


@pytest.fixture
def inputs(gold, tmp_path):
    gold_csv = tmp_path / "gold_daily_sales.csv"
    gold.to_csv(gold_csv, index=False)
    save_outputs(run_forecast(gold[["sales_date", "revenue"]], test_days=14), tmp_path / "outputs")
    return load_inputs(gold_csv, tmp_path / "outputs")


def test_context_contains_computed_facts(inputs, gold):
    context = build_context(*inputs)
    assert f"Total revenue: £{gold.revenue.sum():,.2f}" in context
    assert "MONTHLY REVENUE" in context and "2011-10: revenue" in context
    assert "- Sat: £0.00" in context
    assert "7-DAY FORECAST (model:" in context
    assert "FORECAST MODEL EVALUATION" in context


def test_context_without_forecast_outputs(gold):
    assert "7-DAY FORECAST: not available" in build_context(gold, None, None)


def test_days_mentioned_finds_dates_and_months(gold):
    assert len(days_mentioned(gold, "What happened on 2011-10-05?")) == 1
    assert len(days_mentioned(gold, "How was October 2011?")) == 31
    assert len(days_mentioned(gold, "Compare oct and nov")) == 61
    assert days_mentioned(gold, "What is the total revenue?").empty
    assert "DAYS MENTIONED IN THE QUESTION" in build_context(gold, None, None, "sales on 2011-10-05")


def test_unverified_numbers_flags_only_unknown_figures():
    context = "- Total revenue: £10,247,905.13\n- 2011-11: revenue £1,161,817.38, orders 2,658"
    answer = "Revenue was £10,247,905.13 (about £10,247,905), November had 2,658 orders and £999,999."
    assert unverified_numbers(answer, context) == ["999999"]


def test_answer_question_sends_facts_and_returns_answer(inputs, monkeypatch):
    sent = {}

    def fake_urlopen(request, timeout):
        sent.update(json.loads(request.data))
        return io.BytesIO(json.dumps({"message": {"content": "Saturday revenue was £0.00."}}).encode())

    monkeypatch.setattr(assistant.urllib.request, "urlopen", fake_urlopen)
    answer = answer_question("Why is Saturday revenue zero?", *inputs, model="test-model")

    assert answer.text == "Saturday revenue was £0.00."
    assert answer.unverified_numbers == []
    assert sent["model"] == "test-model"
    assert sent["messages"][0]["role"] == "system"
    assert "Total revenue" in sent["messages"][1]["content"]


def test_clear_error_when_ollama_is_not_running(inputs, monkeypatch):
    def refuse(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(assistant.urllib.request, "urlopen", refuse)
    with pytest.raises(OllamaError, match="Is it running"):
        answer_question("Total revenue?", *inputs)
