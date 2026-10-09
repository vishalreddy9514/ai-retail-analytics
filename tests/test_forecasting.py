import numpy as np
import pandas as pd
import pytest

from retail_analytics.forecasting import (
    BASELINE, HORIZON, add_features, chronological_split, forecast_next_days,
    load_daily_revenue, run_forecast, save_outputs,
)


@pytest.fixture
def daily():
    """200 days with a weekly pattern, closed on Saturdays, plus noise."""
    dates = pd.date_range("2011-01-03", periods=200)  # starts on a Monday
    rng = np.random.default_rng(0)
    weekday_level = np.array([30_000, 32_000, 35_000, 33_000, 25_000, 0, 15_000])
    revenue = weekday_level[dates.dayofweek] + rng.normal(0, 2_000, len(dates))
    revenue[dates.dayofweek == 5] = 0.0
    return pd.DataFrame({"sales_date": dates, "revenue": revenue.clip(0)})


def test_features_only_look_at_least_a_week_back(daily):
    features = add_features(daily)
    row = features.iloc[40]
    assert row["lag_7"] == daily["revenue"].iloc[33]
    assert row["lag_14"] == daily["revenue"].iloc[26]
    assert row["mean_7_lag7"] == pytest.approx(daily["revenue"].iloc[27:34].mean())
    assert row["mean_28_lag7"] == pytest.approx(daily["revenue"].iloc[6:34].mean())


def test_split_is_chronological(daily):
    train, test = chronological_split(add_features(daily), test_days=28)
    assert len(test) == 28
    assert train["sales_date"].max() < test["sales_date"].min()
    assert not train[["lag_7", "mean_28_lag7"]].isna().any().any()


def test_run_forecast_scores_every_model(daily):
    result = run_forecast(daily, test_days=28)
    models = result["summary"]["models"]

    assert set(models) == {BASELINE, "linear_regression", "random_forest"}
    for m in models.values():
        assert 0 <= m["mae"] <= m["rmse"]
    assert result["summary"]["best_model"] in models
    assert len(result["test_predictions"]) == 28


def test_forecast_covers_the_next_seven_days(daily):
    forecast = forecast_next_days(daily, "random_forest")
    expected = pd.date_range(daily["sales_date"].max() + pd.Timedelta(days=1), periods=HORIZON)

    assert list(forecast["forecast_date"]) == list(expected.date)
    assert (forecast["forecast_revenue"] >= 0).all()
    saturday = forecast.loc[forecast["day_of_week"] == "Sat", "forecast_revenue"].iloc[0]
    assert saturday < 0.2 * forecast["forecast_revenue"].max()  # learnt the closed day


def test_load_drops_partial_last_day_and_outputs_are_saved(daily, tmp_path):
    csv_path = tmp_path / "gold.csv"
    daily.assign(orders=1).to_csv(csv_path, index=False)

    loaded = load_daily_revenue(csv_path)
    assert len(loaded) == len(daily) - 1
    assert list(loaded.columns) == ["sales_date", "revenue"]

    save_outputs(run_forecast(loaded, test_days=28), tmp_path / "out")
    assert {p.name for p in (tmp_path / "out").iterdir()} == {
        "forecast_metrics.json", "forecast_test_predictions.csv", "forecast_next_7_days.csv",
    }
