"""Lightweight data quality checks. Each check returns a CheckResult; run_checks fails loudly."""
from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


class DataQualityError(Exception):
    pass


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _count_where(df: DataFrame, condition) -> int:
    return df.filter(condition).count()


def silver_checks(silver_df: DataFrame) -> list[CheckResult]:
    results = []
    total = silver_df.count()
    results.append(CheckResult("silver_not_empty", total > 0, f"{total} rows"))

    for column in ["invoice_no", "stock_code", "invoice_ts", "quantity", "unit_price"]:
        nulls = _count_where(silver_df, F.col(column).isNull())
        results.append(CheckResult(f"{column}_not_null", nulls == 0, f"{nulls} nulls"))

    bad_qty = _count_where(silver_df, F.col("quantity") <= 0)
    results.append(CheckResult("quantity_positive", bad_qty == 0, f"{bad_qty} rows <= 0"))
    bad_price = _count_where(silver_df, F.col("unit_price") <= 0)
    results.append(CheckResult("unit_price_positive", bad_price == 0, f"{bad_price} rows <= 0"))
    cancelled = _count_where(silver_df, F.col("invoice_no").startswith("C"))
    results.append(CheckResult("no_cancelled_invoices", cancelled == 0, f"{cancelled} rows"))
    return results


def gold_checks(gold_df: DataFrame, silver_df: DataFrame) -> list[CheckResult]:
    results = []
    stats = gold_df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("sales_date").alias("distinct_dates"),
        F.min("sales_date").alias("start"),
        F.max("sales_date").alias("end"),
        F.sum("revenue").alias("revenue"),
        F.min("revenue").alias("min_revenue"),
    ).first()

    results.append(
        CheckResult("one_row_per_date", stats.rows == stats.distinct_dates,
                    f"{stats.rows} rows, {stats.distinct_dates} distinct dates")
    )
    expected_days = (stats.end - stats.start).days + 1 if stats.start else 0
    results.append(
        CheckResult("no_missing_dates", stats.rows == expected_days,
                    f"{stats.rows} rows, {expected_days} days from {stats.start} to {stats.end}")
    )
    results.append(
        CheckResult("revenue_non_negative", (stats.min_revenue or 0) >= 0,
                    f"min revenue {stats.min_revenue}")
    )

    # Reconciliation: Gold must add up to Silver (allowing for per-day rounding).
    silver_revenue = silver_df.agg(F.sum("line_total")).first()[0] or 0.0
    gold_revenue = stats.revenue or 0.0
    diff = abs(silver_revenue - gold_revenue)
    results.append(
        CheckResult("revenue_reconciles_with_silver", diff <= 0.01 * max(stats.rows, 1),
                    f"silver {silver_revenue:,.2f} vs gold {gold_revenue:,.2f}")
    )
    return results


def run_checks(results: list[CheckResult]) -> None:
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    failed = [r.name for r in results if not r.passed]
    if failed:
        raise DataQualityError(f"Data quality checks failed: {', '.join(failed)}")
