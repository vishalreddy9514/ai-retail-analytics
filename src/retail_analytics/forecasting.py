"""Seven-day revenue forecast from the Gold daily sales table.

Every feature looks at least 7 days into the past, so one model can predict all
7 future days directly from known history: no predictions are fed back in.

Local: python -m retail_analytics.forecasting
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retail_analytics.config import PROJECT_ROOT

GOLD_CSV = PROJECT_ROOT / "data" / "exports" / "gold_daily_sales.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

TARGET = "revenue"
HORIZON = 7
TEST_DAYS = 56  # last 8 weeks are held out for evaluation
CATEGORICAL = ["day_of_week"]
NUMERIC = ["lag_7", "lag_14", "mean_7_lag7", "mean_28_lag7"]
FEATURES = CATEGORICAL + NUMERIC
BASELINE = "seasonal_naive"


def load_daily_revenue(path: Path, drop_last_day: bool = True) -> pd.DataFrame:
    """Read Gold and keep date + revenue. The UCI data stops at 12:50 on its last
    day, so that partial day is dropped by default to avoid a misleading dip."""
    df = pd.read_csv(path, parse_dates=["sales_date"]).sort_values("sales_date")
    if drop_last_day:
        df = df.iloc[:-1]
    return df[["sales_date", TARGET]].reset_index(drop=True)


def add_features(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()
    out["day_of_week"] = out["sales_date"].dt.dayofweek  # 0 = Monday, 5 = Saturday
    week_ago = out[TARGET].shift(HORIZON)
    out["lag_7"] = week_ago
    out["lag_14"] = out[TARGET].shift(14)
    out["mean_7_lag7"] = week_ago.rolling(7).mean()  # average of days t-13 .. t-7
    out["mean_28_lag7"] = week_ago.rolling(28).mean()  # average of days t-34 .. t-7
    return out


def build_models() -> dict[str, Pipeline]:
    one_hot_day = ("day", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    return {
        "linear_regression": Pipeline([
            ("prep", ColumnTransformer([one_hot_day, ("num", StandardScaler(), NUMERIC)])),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": Pipeline([
            ("prep", ColumnTransformer([one_hot_day], remainder="passthrough")),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42)),
        ]),
    }


def chronological_split(features: pd.DataFrame, test_days: int = TEST_DAYS):
    """Train on the past, test on the most recent days. Never shuffle time series."""
    usable = features.dropna(subset=FEATURES + [TARGET])
    return usable.iloc[:-test_days], usable.iloc[-test_days:]


def score(actual, predicted) -> dict:
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
    }


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit each model on train, score on test. Returns (metrics, test predictions)."""
    predictions = {BASELINE: test["lag_7"].to_numpy()}  # "same as last week"
    for name, model in build_models().items():
        model.fit(train[FEATURES], train[TARGET])
        predictions[name] = np.clip(model.predict(test[FEATURES]), 0, None)

    metrics = {name: score(test[TARGET], pred) for name, pred in predictions.items()}
    table = test[["sales_date", TARGET]].assign(**{n: np.round(p, 2) for n, p in predictions.items()})
    return metrics, table.reset_index(drop=True)


def forecast_next_days(daily: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Refit the chosen model on all history and predict the next HORIZON days."""
    future_dates = pd.date_range(daily["sales_date"].max() + pd.Timedelta(days=1), periods=HORIZON)
    extended = add_features(
        pd.concat([daily, pd.DataFrame({"sales_date": future_dates})], ignore_index=True)
    )
    history = extended.dropna(subset=FEATURES + [TARGET])
    future = extended.tail(HORIZON)

    if model_name == BASELINE:
        predicted = future["lag_7"].to_numpy()
    else:
        model = build_models()[model_name].fit(history[FEATURES], history[TARGET])
        predicted = np.clip(model.predict(future[FEATURES]), 0, None)

    return pd.DataFrame({
        "forecast_date": future["sales_date"].dt.date.to_numpy(),
        "day_of_week": future["sales_date"].dt.day_name().str[:3].to_numpy(),
        "forecast_revenue": np.round(predicted, 2),
        "model": model_name,
    })


def run_forecast(daily: pd.DataFrame, test_days: int = TEST_DAYS) -> dict:
    train, test = chronological_split(add_features(daily), test_days)
    metrics, test_predictions = evaluate(train, test)
    best_model = min(metrics, key=lambda name: metrics[name]["mae"])

    summary = {
        "target": TARGET,
        "horizon_days": HORIZON,
        "train_start": str(train["sales_date"].min().date()),
        "train_end": str(train["sales_date"].max().date()),
        "test_start": str(test["sales_date"].min().date()),
        "test_end": str(test["sales_date"].max().date()),
        "train_rows": len(train),
        "test_rows": len(test),
        "models": metrics,
        "best_model": best_model,
    }
    return {
        "summary": summary,
        "test_predictions": test_predictions,
        "forecast": forecast_next_days(daily, best_model),
    }


def save_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "forecast_metrics.json").write_text(json.dumps(result["summary"], indent=2))
    result["test_predictions"].to_csv(output_dir / "forecast_test_predictions.csv", index=False)
    result["forecast"].to_csv(output_dir / "forecast_next_7_days.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, evaluate and run the 7-day revenue forecast.")
    parser.add_argument("--gold-csv", type=Path, default=GOLD_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if not args.gold_csv.exists():
        raise SystemExit(
            f"{args.gold_csv} not found. Download gold_daily_sales.csv from the Databricks "
            "volume (raw_files/exports) and save it there."
        )

    result = run_forecast(load_daily_revenue(args.gold_csv))
    save_outputs(result, args.output_dir)

    s = result["summary"]
    print(f"Train: {s['train_start']} to {s['train_end']} ({s['train_rows']} days)")
    print(f"Test:  {s['test_start']} to {s['test_end']} ({s['test_rows']} days)\n")
    print(f"{'model':<20}{'MAE':>12}{'RMSE':>12}")
    for name, m in s["models"].items():
        print(f"{name:<20}{m['mae']:>12,.2f}{m['rmse']:>12,.2f}")
    print(f"\nBest model (lowest test MAE): {s['best_model']}\n")
    print("Next 7 days:")
    print(result["forecast"].to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
