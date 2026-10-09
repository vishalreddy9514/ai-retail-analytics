"""Run Bronze -> Silver -> Gold end to end.

Local:      python -m retail_analytics.pipeline
Databricks: the notebooks in notebooks/ call the same functions step by step.
"""
import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from retail_analytics.bronze import build_bronze
from retail_analytics.config import (
    BRONZE_TABLE, GOLD_TABLE, QUARANTINE_TABLE, SILVER_TABLE, Settings,
)
from retail_analytics.gold import build_gold_daily_sales
from retail_analytics.quality import gold_checks, run_checks, silver_checks
from retail_analytics.silver import build_silver
from retail_analytics.storage import get_local_spark, read_delta, write_delta


def run_bronze(spark: SparkSession, settings: Settings) -> int:
    write_delta(build_bronze(spark, settings.raw_csv_path), settings, BRONZE_TABLE)
    return read_delta(spark, settings, BRONZE_TABLE).count()


def run_silver(spark: SparkSession, settings: Settings) -> dict:
    silver, quarantine = build_silver(read_delta(spark, settings, BRONZE_TABLE))
    write_delta(silver, settings, SILVER_TABLE)
    write_delta(quarantine, settings, QUARANTINE_TABLE)

    silver = read_delta(spark, settings, SILVER_TABLE)
    print("Silver data quality checks:")
    run_checks(silver_checks(silver))

    reasons = (
        read_delta(spark, settings, QUARANTINE_TABLE)
        .groupBy("rejection_reason").count().orderBy(F.desc("count")).collect()
    )
    return {"silver_rows": silver.count(), "rejected": {r[0]: r[1] for r in reasons}}


def run_gold(spark: SparkSession, settings: Settings) -> int:
    silver = read_delta(spark, settings, SILVER_TABLE)
    write_delta(build_gold_daily_sales(silver), settings, GOLD_TABLE)

    gold = read_delta(spark, settings, GOLD_TABLE)
    print("Gold data quality checks:")
    run_checks(gold_checks(gold, silver))
    return gold.count()


def export_gold_csv(gold_df: DataFrame, export_dir: str) -> str:
    """Write Gold as one small CSV for the local forecasting, LLM and dashboard stages."""
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(export_dir) / f"{GOLD_TABLE}.csv")
    gold_df.orderBy("sales_date").toPandas().to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Bronze/Silver/Gold pipeline locally.")
    parser.add_argument("--csv", help="Path to online_retail.csv (default: data/raw/online_retail.csv)")
    args = parser.parse_args()

    settings = Settings.for_local()
    if args.csv:
        settings = Settings(**{**settings.__dict__, "raw_csv_path": args.csv})
    if not Path(settings.raw_csv_path).exists():
        raise SystemExit(f"{settings.raw_csv_path} not found. Run: python scripts/download_data.py")

    spark = get_local_spark()
    print(f"Bronze rows: {run_bronze(spark, settings):,}")
    silver_stats = run_silver(spark, settings)
    print(f"Silver rows: {silver_stats['silver_rows']:,}")
    print(f"Quarantined rows by reason: {silver_stats['rejected']}")
    print(f"Gold rows (days): {run_gold(spark, settings):,}")
    out = export_gold_csv(read_delta(spark, settings, GOLD_TABLE), settings.export_dir)
    print(f"Exported Gold to {out}")
    spark.stop()


if __name__ == "__main__":
    main()
