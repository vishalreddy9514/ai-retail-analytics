"""AI Retail Analytics dashboard: historical sales, the 7-day forecast and AI explanations.

Run from the project folder:  streamlit run streamlit_app.py
"""
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retail_analytics.assistant import (  # noqa: E402
    FORECAST_QUESTION, OLLAMA_MODEL, OllamaError, answer_question, load_inputs,
)
from retail_analytics.config import PROJECT_ROOT  # noqa: E402

DATA_DIR = Path(os.environ.get("RETAIL_DATA_DIR", PROJECT_ROOT / "data"))
GOLD_CSV = DATA_DIR / "exports" / "gold_daily_sales.csv"
OUTPUT_DIR = DATA_DIR / "outputs"
TEST_PREDICTIONS_CSV = OUTPUT_DIR / "forecast_test_predictions.csv"

ACTUAL_COLOR = "#2a78d6"    # categorical slot 1 (blue)
FORECAST_COLOR = "#eb6834"  # categorical slot 2 (orange)
SERIES_SCALE = alt.Scale(domain=["Actual", "Forecast"], range=[ACTUAL_COLOR, FORECAST_COLOR])


def money(value: float) -> str:
    return f"£{value:,.0f}"


@st.cache_data
def load_data():
    gold, metrics, forecast = load_inputs(GOLD_CSV, OUTPUT_DIR)
    test = pd.read_csv(TEST_PREDICTIONS_CSV, parse_dates=["sales_date"]) if TEST_PREDICTIONS_CSV.exists() else None
    return gold, metrics, forecast, test


def history_and_forecast_chart(gold: pd.DataFrame, forecast: pd.DataFrame | None, days: int) -> alt.Chart:
    if forecast is not None:
        # Show the history the forecast was built from (the partial last day is excluded).
        gold = gold[gold.sales_date < forecast.forecast_date.min()]
    actual = gold.tail(days).assign(series="Actual")[["sales_date", "revenue", "series"]]
    frames = [actual]
    if forecast is not None:
        # Start the forecast line at the last actual point so the two lines connect.
        bridge = actual.tail(1).assign(series="Forecast")
        future = forecast.rename(columns={"forecast_date": "sales_date", "forecast_revenue": "revenue"})
        frames += [bridge, future.assign(series="Forecast")[["sales_date", "revenue", "series"]]]
    data = pd.concat(frames, ignore_index=True)

    base = alt.Chart(data).encode(
        x=alt.X("sales_date:T", title=None, axis=alt.Axis(format="%d %b")),
        y=alt.Y("revenue:Q", title="Revenue (£)", axis=alt.Axis(format=",.0f")),
        color=alt.Color("series:N", scale=SERIES_SCALE, legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.condition(alt.datum.series == "Forecast", alt.value([6, 4]), alt.value([1, 0])),
        tooltip=[alt.Tooltip("sales_date:T", title="Date", format="%a %d %b %Y"),
                 alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("revenue:Q", title="Revenue (£)", format=",.2f")],
    )
    return (base.mark_line(strokeWidth=2) + base.mark_point(size=40, filled=True, opacity=0.9)).properties(height=320)


def monthly_chart(gold: pd.DataFrame) -> alt.Chart:
    monthly = gold.groupby(gold.sales_date.dt.to_period("M").dt.to_timestamp()).revenue.sum().reset_index()
    return alt.Chart(monthly).mark_bar(color=ACTUAL_COLOR, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("sales_date:T", title=None, timeUnit="yearmonth", axis=alt.Axis(format="%b %Y")),
        y=alt.Y("revenue:Q", title="Revenue (£)", axis=alt.Axis(format=",.0f")),
        tooltip=[alt.Tooltip("sales_date:T", title="Month", timeUnit="yearmonth", format="%B %Y"),
                 alt.Tooltip("revenue:Q", title="Revenue (£)", format=",.2f")],
    ).properties(height=280)


def test_period_chart(test: pd.DataFrame, model: str) -> alt.Chart:
    data = pd.concat([
        test[["sales_date", "revenue"]].assign(series="Actual"),
        test[["sales_date", model]].rename(columns={model: "revenue"}).assign(series="Forecast"),
    ])
    return alt.Chart(data).mark_line(strokeWidth=2, point=True).encode(
        x=alt.X("sales_date:T", title=None, axis=alt.Axis(format="%d %b")),
        y=alt.Y("revenue:Q", title="Revenue (£)", axis=alt.Axis(format=",.0f")),
        color=alt.Color("series:N", scale=SERIES_SCALE, legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("sales_date:T", title="Date", format="%a %d %b %Y"),
                 alt.Tooltip("series:N", title="Series"),
                 alt.Tooltip("revenue:Q", title="Revenue (£)", format=",.2f")],
    ).properties(height=280)


def show_answer(answer) -> None:
    st.markdown(answer.text)
    if answer.unverified_numbers:
        st.warning("These numbers are not in the source data (they may be the model's own calculations "
                   f"or mistakes): {', '.join(answer.unverified_numbers)}")
    with st.expander("Facts sent to the model"):
        st.code(answer.context, language=None)


def ask(question: str, model: str, gold, metrics, forecast, key: str) -> None:
    try:
        with st.spinner(f"Asking {model} (the first answer can take up to a minute)..."):
            st.session_state[key] = answer_question(question, gold, metrics, forecast, model)
    except OllamaError as err:
        st.session_state.pop(key, None)
        st.error(f"{err}\n\nStart the Ollama app, then try again.")


st.set_page_config(page_title="AI Retail Analytics", layout="wide")
st.title("AI Retail Analytics")

if not GOLD_CSV.exists():
    st.error(f"`{GOLD_CSV}` not found. Download `gold_daily_sales.csv` from Databricks "
             "(Catalog > workspace > retail > raw_files > exports) and save it there.")
    st.stop()

gold, metrics, forecast, test = load_data()
st.caption(f"UCI Online Retail, daily sales {gold.sales_date.min():%d %b %Y} to "
           f"{gold.sales_date.max():%d %b %Y} (the last day is partial). Gold table from the Databricks pipeline.")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model", value=OLLAMA_MODEL)
    days = st.slider("Days of history in the chart", min_value=28, max_value=len(gold), value=90, step=7)

trading = gold[gold.is_trading_day]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total revenue", money(gold.revenue.sum()))
c2.metric("Orders", f"{gold.orders.sum():,}")
c3.metric("Avg revenue per trading day", money(trading.revenue.mean()))
c4.metric("Avg order value", f"£{gold.revenue.sum() / gold.orders.sum():,.2f}")

st.subheader("Daily revenue and 7-day forecast")
if forecast is None:
    st.info("No forecast yet. Run `python -m retail_analytics.forecasting` to add it.")
st.altair_chart(history_and_forecast_chart(gold, forecast, days), use_container_width=True)
st.caption("Saturdays show £0: the retailer does not trade on Saturdays.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Monthly revenue")
    st.altair_chart(monthly_chart(gold), use_container_width=True)
    st.caption(f"The last month is partial: data ends on {gold.sales_date.max():%d %b %Y}.")
with right:
    st.subheader("Next 7 days")
    if forecast is not None:
        st.dataframe(
            forecast.assign(forecast_date=forecast.forecast_date.dt.date).rename(columns={
                "forecast_date": "Date", "day_of_week": "Day", "forecast_revenue": "Forecast (£)", "model": "Model"}),
            hide_index=True, use_container_width=True,
            column_config={"Forecast (£)": st.column_config.NumberColumn(format="%.2f")},
        )
        st.metric("Forecast total", f"£{forecast.forecast_revenue.sum():,.2f}")

st.subheader("AI explanation of the forecast")
st.caption(f"Generated locally by Ollama ({model}) from the figures above. Numbers not found in the data are flagged.")
if st.button("Explain the forecast", type="primary", disabled=forecast is None):
    ask(FORECAST_QUESTION, model, gold, metrics, forecast, "explanation")
if "explanation" in st.session_state:
    show_answer(st.session_state["explanation"])

st.subheader("Ask a question")
with st.form("question_form"):
    question = st.text_input("Question", placeholder="How did November 2011 compare to October 2011?")
    submitted = st.form_submit_button("Ask")
if submitted and question.strip():
    ask(question.strip(), model, gold, metrics, forecast, "answer")
if "answer" in st.session_state:
    show_answer(st.session_state["answer"])

if metrics is not None:
    st.subheader("How accurate is the forecast?")
    st.caption(f"Models trained on {metrics['train_start']} to {metrics['train_end']} and tested on "
               f"{metrics['test_start']} to {metrics['test_end']} (chronological hold-out).")
    scores = pd.DataFrame(metrics["models"]).T.rename(columns={"mae": "MAE (£)", "rmse": "RMSE (£)"})
    scores.index.name = "Model"
    best = metrics["best_model"]
    st.dataframe(scores.reset_index(), hide_index=True, use_container_width=True)
    note = f"Best model: **{best}** (lowest MAE)."
    if metrics.get("test_mean_revenue"):
        share = metrics["models"][best]["mae"] / metrics["test_mean_revenue"]
        note += f" Its typical daily error is {share:.1%} of average daily revenue in the test period."
    st.markdown(note)
    if test is not None:
        shown = st.selectbox("Compare actual sales with", list(metrics["models"]), index=list(metrics["models"]).index(best))
        st.altair_chart(test_period_chart(test, shown), use_container_width=True)
