import pytest

pytest.importorskip("pyspark")  # Spark tests need requirements-spark.txt

from retail_analytics.bronze import SOURCE_COLUMNS, build_bronze

CSV_TEXT = """InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country
536365,85123A,"WHITE HANGING HEART, T-LIGHT HOLDER",6,2010-12-01 08:26:00,2.55,17850,United Kingdom
C536379,D,Discount,-1,2010-12-01 09:41:00,27.5,,United Kingdom
"""


def test_bronze_keeps_raw_strings_and_adds_metadata(spark, tmp_path):
    csv_path = tmp_path / "online_retail.csv"
    csv_path.write_text(CSV_TEXT)

    bronze = build_bronze(spark, str(csv_path))
    rows = bronze.orderBy("InvoiceNo").collect()

    assert bronze.columns == SOURCE_COLUMNS + ["_source_file", "_ingested_at"]
    assert all(dict(bronze.dtypes)[c] == "string" for c in SOURCE_COLUMNS)
    assert len(rows) == 2
    assert rows[0]["Description"] == "WHITE HANGING HEART, T-LIGHT HOLDER"  # quoted comma
    assert rows[1]["CustomerID"] is None
    assert rows[0]["_source_file"].endswith("online_retail.csv")
