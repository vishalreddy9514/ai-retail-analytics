"""End-to-end: CSV -> Bronze -> Silver -> Gold Delta tables -> exported CSV, all in a temp folder."""
import csv

import pandas as pd
import pytest

pytest.importorskip("pyspark")  # Spark tests need requirements-spark.txt

from retail_analytics.bronze import SOURCE_COLUMNS
from retail_analytics.config import Settings
from retail_analytics.pipeline import export_gold_csv, run_bronze, run_gold, run_silver
from retail_analytics.storage import read_delta


def test_pipeline_end_to_end(spark, tmp_path, bronze_rows):
    (tmp_path / "raw").mkdir()
    with open(tmp_path / "raw" / "online_retail.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SOURCE_COLUMNS)
        writer.writerows(bronze_rows)
    settings = Settings.for_local(data_dir=tmp_path)

    assert run_bronze(spark, settings) == len(bronze_rows)
    stats = run_silver(spark, settings)
    assert stats["silver_rows"] == 3
    assert sum(stats["rejected"].values()) == len(bronze_rows) - 3
    assert run_gold(spark, settings) == 3

    out = export_gold_csv(read_delta(spark, settings, "gold_daily_sales"), settings.export_dir)
    exported = pd.read_csv(out)
    assert list(exported["sales_date"]) == ["2010-12-01", "2010-12-02", "2010-12-03"]
    assert exported["revenue"].sum() == pytest.approx(35.64 + 18.5)
