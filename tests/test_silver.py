import pytest

pytest.importorskip("pyspark")  # Spark tests need requirements-spark.txt

from retail_analytics.bronze import RAW_SCHEMA
from retail_analytics.quality import run_checks, silver_checks
from retail_analytics.silver import build_silver


def test_silver_rejects_bad_rows_with_reasons(spark, bronze_rows):
    silver, quarantine = build_silver(spark.createDataFrame(bronze_rows, RAW_SCHEMA))

    reasons = {r["invoice_no"]: r["rejection_reason"] for r in quarantine.collect()}
    assert reasons == {
        "536365": "duplicate",
        "C536379": "cancelled_invoice",
        "536380": "non_positive_quantity",
        "536381": "non_positive_unit_price",
        "536382": "non_product_code",
        "536383": "invalid_quantity",
        "536384": "invalid_invoice_date",
    }
    assert silver.count() == 3


def test_silver_types_and_derived_columns(spark, bronze_rows):
    silver, _ = build_silver(spark.createDataFrame(bronze_rows, RAW_SCHEMA))
    row = silver.filter("stock_code = '71053'").first()

    assert row["quantity"] == 6
    assert row["unit_price"] == 3.39
    assert row["line_total"] == 20.34
    assert row["customer_id"] == "17850"  # "17850.0" normalised
    assert str(row["invoice_date"]) == "2010-12-01"
    assert silver.filter("country = 'France'").first()["customer_id"] is None


def test_silver_passes_quality_checks(spark, bronze_rows):
    silver, _ = build_silver(spark.createDataFrame(bronze_rows, RAW_SCHEMA))
    run_checks(silver_checks(silver))  # raises DataQualityError on failure
