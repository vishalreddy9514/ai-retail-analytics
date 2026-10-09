"""Local LLM assistant that answers questions from real sales metrics and forecasts.

How it stays grounded (no vector database needed):
1. pandas computes the facts: totals, monthly and weekday figures, recent days, the
   forecast and the model scores. Days or months named in the question are added too.
2. Those facts are sent to a local Ollama model with instructions to use only them.
3. Every number in the answer is checked against the facts; unmatched ones are flagged.

Local: python -m retail_analytics.assistant "How did November 2011 compare to October?"
"""
import argparse
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from retail_analytics.forecasting import GOLD_CSV, OUTPUT_DIR

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = """You are a retail sales analyst for a UK online gift retailer.
Answer using ONLY the facts provided in the DATA section. Rules:
- Quote figures exactly as they appear in the data, in pounds (£).
- If the data does not contain the answer, say "The data provided does not cover that."
- Do not invent numbers, products, customers or causes. You may point out patterns that
  are visible in the data (for example, Saturdays have no trading).
- Keep answers short: at most 5 sentences or a short bullet list."""

FORECAST_QUESTION = (
    "Explain the 7-day revenue forecast in plain English: the expected total, how it compares "
    "with the last 7 actual days, which days are highest and lowest, and how reliable the "
    "model is based on its test MAE compared with the baseline."
)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


class OllamaError(Exception):
    pass


@dataclass
class Answer:
    question: str
    text: str
    context: str
    unverified_numbers: list[str] = field(default_factory=list)


def money(value: float) -> str:
    return f"£{value:,.2f}"


def load_inputs(gold_csv: Path = GOLD_CSV, output_dir: Path = OUTPUT_DIR):
    """Return (gold, metrics or None, forecast or None)."""
    if not Path(gold_csv).exists():
        raise FileNotFoundError(f"{gold_csv} not found. Download it from Databricks first (see README).")
    gold = pd.read_csv(gold_csv, parse_dates=["sales_date"]).sort_values("sales_date")

    metrics_path = Path(output_dir) / "forecast_metrics.json"
    forecast_path = Path(output_dir) / "forecast_next_7_days.csv"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    forecast = pd.read_csv(forecast_path, parse_dates=["forecast_date"]) if forecast_path.exists() else None
    return gold, metrics, forecast


def _day_rows(days: pd.DataFrame) -> list[str]:
    return [
        f"- {d.sales_date.date()} ({d.day_of_week}): revenue {money(d.revenue)}, "
        f"orders {d.orders}, units {d.units_sold}, customers {d.unique_customers}"
        for d in days.itertuples()
    ]


def days_mentioned(gold: pd.DataFrame, question: str) -> pd.DataFrame:
    """Simple retrieval: rows for ISO dates or month names that appear in the question."""
    q = question.lower()
    mask = gold["sales_date"].dt.strftime("%Y-%m-%d").isin(re.findall(r"\d{4}-\d{2}-\d{2}", q))
    for number, name in enumerate(MONTHS, start=1):
        if re.search(rf"\b({name}|{name[:3]})\b", q):
            in_month = gold["sales_date"].dt.month == number
            years = [int(y) for y in re.findall(r"\b(20\d\d)\b", q)]
            if years:
                in_month &= gold["sales_date"].dt.year.isin(years)
            mask |= in_month
    return gold[mask]


def build_context(gold: pd.DataFrame, metrics: dict | None, forecast: pd.DataFrame | None,
                  question: str = "") -> str:
    trading = gold[gold["is_trading_day"]]
    lines = [
        "DATA COVERAGE",
        f"- Daily sales from {gold.sales_date.min().date()} to {gold.sales_date.max().date()} "
        f"({len(gold)} days, {len(trading)} trading days). The last day is partial (data ends at 12:50).",
        "",
        "TOTALS",
        f"- Total revenue: {money(gold.revenue.sum())}",
        f"- Total orders: {gold.orders.sum():,}; units sold: {gold.units_sold.sum():,}",
        f"- Average revenue per trading day: {money(trading.revenue.mean())}",
        f"- Average order value: {money(gold.revenue.sum() / gold.orders.sum())}",
        "",
        "MONTHLY REVENUE",
    ]
    monthly = gold.groupby(gold.sales_date.dt.to_period("M")).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"), trading_days=("is_trading_day", "sum"))
    lines += [f"- {r.Index}: revenue {money(r.revenue)}, orders {int(r.orders):,}, "
              f"trading days {int(r.trading_days)}" for r in monthly.itertuples()]

    lines += ["", "AVERAGE REVENUE BY DAY OF WEEK (all days)"]
    weekday = gold.groupby("day_of_week").revenue.mean().reindex(WEEKDAYS).dropna()
    lines += [f"- {day}: {money(value)}" for day, value in weekday.items()]

    lines += ["", "TOP 5 REVENUE DAYS"] + _day_rows(gold.nlargest(5, "revenue"))
    lines += ["", "LAST 14 DAYS"] + _day_rows(gold.tail(14))

    if forecast is not None:
        start = forecast.forecast_date.min()
        # Compare with the 7 complete days the forecast was built from (the partial last day is excluded).
        last_7 = gold[gold.sales_date < start].tail(7).revenue.sum()
        total = forecast.forecast_revenue.sum()
        lines += ["", f"7-DAY FORECAST (model: {forecast.model.iloc[0]}, starts {start.date()})"]
        lines += [f"- {f.forecast_date.date()} ({f.day_of_week}): {money(f.forecast_revenue)}"
                  for f in forecast.itertuples()]
        lines += [
            f"- Forecast total for the 7 days: {money(total)}",
            f"- Actual total for the 7 days before the forecast: {money(last_7)}",
            f"- Change versus the last 7 days: {(total - last_7) / last_7:+.1%}",
        ]
    else:
        lines += ["", "7-DAY FORECAST: not available (run the forecasting step)."]

    if metrics is not None:
        lines += ["", "FORECAST MODEL EVALUATION",
                  f"- Trained on {metrics['train_start']} to {metrics['train_end']}, "
                  f"tested on {metrics['test_start']} to {metrics['test_end']} ({metrics['test_rows']} days)."]
        lines += [f"- {name}: MAE {money(m['mae'])}, RMSE {money(m['rmse'])}"
                  for name, m in metrics["models"].items()]
        lines.append(f"- Best model (lowest MAE): {metrics['best_model']}. "
                     "seasonal_naive means 'same as the same weekday last week'.")

    requested = days_mentioned(gold, question)
    if not requested.empty:
        lines += ["", "DAYS MENTIONED IN THE QUESTION"] + _day_rows(requested)
    return "\n".join(lines)


def _numbers(text: str) -> set[str]:
    """Numbers >= 100 in a text, normalised (no commas, no trailing zeros), ignoring dates and years."""
    text = re.sub(r"\d{4}-\d{2}(-\d{2})?", " ", text)
    found = set()
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        value = float(raw.replace(",", ""))
        if value < 100 or (value.is_integer() and 1900 <= value <= 2100):
            continue
        found.add(f"{value:.2f}".rstrip("0").rstrip("."))
    return found


def unverified_numbers(answer: str, context: str) -> list[str]:
    """Numbers in the answer that are not in the context (also accepting whole-pound rounding)."""
    known = _numbers(context)
    known |= {str(round(float(n))) for n in known}
    return sorted(_numbers(answer) - known, key=float)


def ask_ollama(system: str, user: str, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST,
               timeout: int = 300) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "options": {"temperature": 0.1, "num_ctx": 8192},
    }
    request = urllib.request.Request(
        f"{host}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())["message"]["content"].strip()
    except urllib.error.HTTPError as err:
        if err.code == 404:
            raise OllamaError(f"Model '{model}' is not installed. Run: ollama pull {model}") from err
        raise OllamaError(f"Ollama returned HTTP {err.code}: {err.read().decode(errors='ignore')}") from err
    except urllib.error.URLError as err:
        raise OllamaError(f"Cannot reach Ollama at {host}. Is it running? Start the Ollama app.") from err


def answer_question(question: str, gold: pd.DataFrame, metrics: dict | None,
                    forecast: pd.DataFrame | None, model: str = OLLAMA_MODEL) -> Answer:
    context = build_context(gold, metrics, forecast, question)
    text = ask_ollama(SYSTEM_PROMPT, f"DATA\n{context}\n\nQUESTION\n{question}", model=model)
    return Answer(question, text, context, unverified_numbers(text, context))


def print_answer(answer: Answer) -> None:
    print(f"\n{answer.text}\n")
    if answer.unverified_numbers:
        print("Check: these numbers are not in the source data (they may be the model's own "
              f"calculations or mistakes): {', '.join(answer.unverified_numbers)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about the sales data with a local LLM.")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("--explain-forecast", action="store_true", help="Explain the 7-day forecast.")
    parser.add_argument("--show-context", action="store_true", help="Print the facts sent to the model.")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    args = parser.parse_args()

    gold, metrics, forecast = load_inputs()
    questions = [FORECAST_QUESTION] if args.explain_forecast else [args.question] if args.question else None

    try:
        if questions:
            for q in questions:
                answer = answer_question(q, gold, metrics, forecast, args.model)
                if args.show_context:
                    print(answer.context)
                print_answer(answer)
            return
        print(f"Ask about the sales data (model: {args.model}). Press Enter on an empty line to quit.")
        while question := input("\nQuestion> ").strip():
            print_answer(answer_question(question, gold, metrics, forecast, args.model))
    except OllamaError as err:
        raise SystemExit(f"Error: {err}") from None


if __name__ == "__main__":
    main()
