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
- Copy figures exactly as they appear in the DATA, in pounds (£), together with the date or
  month they belong to. Never move a figure to a different date or month.
- Never calculate differences, totals, averages or percentages yourself. Use the precomputed
  ones in the DATA (for example "versus previous month", "COMPARISON", "ranked").
- If a figure the question needs is not in the DATA, say it is not available.
- Forecast values are predictions, not actual sales.
- Do not invent numbers, products, customers or causes. You may point out patterns that
  are visible in the data (for example, Saturdays have no trading).
- Keep answers short: at most 5 sentences or a short bullet list."""

FORECAST_QUESTION = (
    "Explain the 7-day revenue forecast in plain English: the forecast total and its change versus "
    "the previous 7 days, the highest and lowest forecast days (use the ranked list), which model "
    "is used and why, and how large its typical daily error (MAE) is."
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


def signed_money(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}£{abs(value):,.2f}"


def dates_mentioned(gold: pd.DataFrame, question: str) -> pd.DataFrame:
    """Rows for ISO dates (YYYY-MM-DD) that appear in the question."""
    wanted = re.findall(r"\d{4}-\d{2}-\d{2}", question)
    return gold[gold["sales_date"].dt.strftime("%Y-%m-%d").isin(wanted)]


def months_mentioned(gold: pd.DataFrame, question: str) -> list[pd.Period]:
    """Months named in the question ('November 2011', 'oct'), limited to months in the data."""
    q = question.lower()
    years = {int(y) for y in re.findall(r"\b(20\d\d)\b", q)}
    available = gold["sales_date"].dt.to_period("M").unique()
    found = []
    for number, name in enumerate(MONTHS, start=1):
        if re.search(rf"\b({name}|{name[:3]})\b", q):
            found += [m for m in available if m.month == number and (not years or m.year in years)]
    return sorted(set(found))


def monthly_summary(gold: pd.DataFrame) -> pd.DataFrame:
    monthly = gold.groupby(gold.sales_date.dt.to_period("M")).agg(
        revenue=("revenue", "sum"), orders=("orders", "sum"), units_sold=("units_sold", "sum"),
        trading_days=("is_trading_day", "sum"), last_day=("sales_date", "max"))
    monthly["revenue_change"] = monthly.revenue.diff()
    monthly["revenue_change_pct"] = monthly.revenue.pct_change()
    monthly["orders_change"] = monthly.orders.diff()
    monthly["partial"] = monthly.last_day < monthly.index.to_timestamp(how="end").normalize()
    return monthly


def _month_line(month, r) -> str:
    line = (f"- {month}{' (partial month)' if r.partial else ''}: revenue {money(r.revenue)}, "
            f"orders {int(r.orders):,}, units sold {int(r.units_sold):,}, trading days {int(r.trading_days)}")
    if pd.notna(r.revenue_change):
        line += (f"; versus previous month: revenue {signed_money(r.revenue_change)} "
                 f"({r.revenue_change_pct:+.1%}), orders {int(r.orders_change):+,}")
    return line


def _compare_months(monthly: pd.DataFrame, a, b) -> str:
    ra, rb = monthly.loc[a], monthly.loc[b]
    diff = ra.revenue - rb.revenue
    return (f"- {a} versus {b}: revenue {money(ra.revenue)} versus {money(rb.revenue)}, "
            f"difference {signed_money(diff)} ({diff / rb.revenue:+.1%}); orders {int(ra.orders):,} versus "
            f"{int(rb.orders):,}, difference {int(ra.orders - rb.orders):+,}; units sold "
            f"{int(ra.units_sold):,} versus {int(rb.units_sold):,}, difference {int(ra.units_sold - rb.units_sold):+,}")


def build_context(gold: pd.DataFrame, metrics: dict | None, forecast: pd.DataFrame | None,
                  question: str = "") -> str:
    trading = gold[gold["is_trading_day"]]
    lines = [
        "DATA COVERAGE",
        f"- Daily sales from {gold.sales_date.min().date()} to {gold.sales_date.max().date()} "
        f"({len(gold)} days, {len(trading)} trading days). The last day is partial (data ends at 12:50).",
        "- Only daily totals are available: no product-level data, and no unique-customer counts per month.",
        "",
        "TOTALS",
        f"- Total revenue: {money(gold.revenue.sum())}",
        f"- Total orders: {gold.orders.sum():,}; units sold: {gold.units_sold.sum():,}",
        f"- Average revenue per trading day: {money(trading.revenue.mean())}",
        f"- Average order value: {money(gold.revenue.sum() / gold.orders.sum())}",
        "",
        "MONTHLY REVENUE",
    ]
    monthly = monthly_summary(gold)
    lines += [_month_line(m, r) for m, r in zip(monthly.index, monthly.itertuples())]
    best = monthly.revenue.idxmax()
    lines.append(f"- Highest revenue month: {best} ({money(monthly.revenue.max())})")

    months = months_mentioned(gold, question)
    if len(months) >= 2:
        lines += ["", "COMPARISON OF MONTHS IN THE QUESTION (later month first)"]
        lines += [_compare_months(monthly, later, earlier)
                  for earlier, later in zip(months, months[1:])]

    lines += ["", "AVERAGE REVENUE BY DAY OF WEEK (all days)"]
    weekday = gold.groupby("day_of_week").revenue.mean().reindex(WEEKDAYS).dropna()
    lines += [f"- {day}: {money(value)}" for day, value in weekday.items()]

    lines += ["", "TOP 5 REVENUE DAYS"] + _day_rows(gold.nlargest(5, "revenue"))
    lines += ["", "LAST 14 DAYS"] + _day_rows(gold.tail(14))

    if forecast is not None:
        start = forecast.forecast_date.min()
        # Compare with the 7 complete days the forecast was built from (the partial last day is excluded).
        previous = gold[gold.sales_date < start].tail(7)
        last_7 = previous.revenue.sum()
        total = forecast.forecast_revenue.sum()
        ranked = forecast.sort_values("forecast_revenue", ascending=False)
        lines += ["", f"7-DAY FORECAST (predictions, not actual sales; model: {forecast.model.iloc[0]}, "
                      f"starts {start.date()})"]
        lines += [f"- {f.forecast_date.date()} ({f.day_of_week}): {money(f.forecast_revenue)}"
                  for f in forecast.itertuples()]
        lines += [
            "- Forecast days ranked highest to lowest: " + "; ".join(
                f"{f.forecast_date.date()} ({f.day_of_week}) {money(f.forecast_revenue)}" for f in ranked.itertuples()),
            f"- Forecast total for the 7 days: {money(total)}",
            f"- Actual total for the previous 7 days ({previous.sales_date.min().date()} to "
            f"{previous.sales_date.max().date()}, before the forecast starts): {money(last_7)}",
            f"- Change versus the previous 7 days: {signed_money(total - last_7)} ({(total - last_7) / last_7:+.1%})",
        ]
        if ((forecast.day_of_week == "Sat") & (forecast.forecast_revenue == 0)).any():
            lines.append("- Saturday is forecast at £0.00 because the shop does not trade on Saturdays.")
    else:
        lines += ["", "7-DAY FORECAST: not available (run the forecasting step)."]

    if metrics is not None:
        best_model = metrics["best_model"]
        best_mae = metrics["models"][best_model]["mae"]
        lines += ["", "FORECAST MODEL EVALUATION",
                  f"- Trained on {metrics['train_start']} to {metrics['train_end']}, "
                  f"tested on {metrics['test_start']} to {metrics['test_end']} ({metrics['test_rows']} days)."]
        lines += [f"- {name}: MAE {money(m['mae'])}, RMSE {money(m['rmse'])}"
                  for name, m in metrics["models"].items()]
        lines.append(f"- Best model (lowest MAE): {best_model}. "
                     "seasonal_naive means 'same as the same weekday last week'.")
        lines += [f"- {best_model} MAE is {money(m['mae'] - best_mae)} lower than {name}."
                  for name, m in metrics["models"].items() if name != best_model]
        if metrics.get("test_mean_revenue"):
            mean = metrics["test_mean_revenue"]
            lines.append(f"- Average daily revenue in the test period: {money(mean)}; the best model's "
                         f"typical daily error (MAE) is {best_mae / mean:.1%} of that.")

    requested = dates_mentioned(gold, question)
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
        "options": {"temperature": 0, "num_ctx": 8192},
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
