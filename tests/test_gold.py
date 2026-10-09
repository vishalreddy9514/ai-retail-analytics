import pytest

pytest.importorskip("pyspark")  # Spark tests need requirements-spark.txt

from retail_analytics.bronze import RAW_SCHEMA
from retail_analytics.gold import build_gold_daily_sales
from retail_analytics.quality import DataQualityError, gold_checks, run_checks
from retail_analytics.silver import build_silver


@pytest.fixture
def silver(spark, bronze_rows):
    return build_silver(spark.createDataFrame(bronze_rows, RAW_SCHEMA))[0]


def test_gold_daily_metrics(silver):
    rows = {str(r["sales_date"]): r for r in build_gold_daily_sales(silver).collect()}

    assert list(rows) == ["2010-12-01", "2010-12-02", "2010-12-03"]
    day1 = rows["2010-12-01"]
    assert day1["revenue"] == pytest.approx(6 * 2.55 + 6 * 3.39)
    assert day1["orders"] == 1
    assert day1["units_sold"] == 12
    assert day1["unique_customers"] == 1
    assert day1["line_items"] == 2
    assert day1["avg_order_value"] == pytest.approx(35.64)
    assert day1["day_of_week"] == "Wed"


def test_gold_fills_non_trading_days_with_zero(silver):
    day2 = build_gold_daily_sales(silver).filter("sales_date = '2010-12-02'").first()

    assert day2["is_trading_day"] is False
    assert day2["revenue"] == 0.0
    assert day2["orders"] == 0
    assert day2["avg_order_value"] is None


def test_gold_quality_checks_pass(silver):
    run_checks(gold_checks(build_gold_daily_sales(silver), silver))


def test_gold_quality_checks_catch_duplicate_dates(silver):
    gold = build_gold_daily_sales(silver)
    with pytest.raises(DataQualityError, match="one_row_per_date"):
        run_checks(gold_checks(gold.unionByName(gold), silver))
