import pytest


@pytest.fixture(scope="session")
def spark():
    # Spark tests are skipped when PySpark isn't installed (see requirements-spark.txt).
    pytest.importorskip("pyspark")
    from retail_analytics.storage import get_local_spark

    session = get_local_spark("ai-retail-analytics-tests")
    yield session
    session.stop()


@pytest.fixture
def bronze_rows():
    """A handful of raw lines covering each cleaning rule. Columns follow bronze.SOURCE_COLUMNS."""
    return [
        # valid lines: invoice 536365 on day 1, 536366 on day 3 (day 2 has no trading)
        ("536365", "85123A", "WHITE HANGING HEART", "6", "2010-12-01 08:26:00", "2.55", "17850", "United Kingdom"),
        ("536365", "71053", "WHITE METAL LANTERN", "6", "2010-12-01 08:26:00", "3.39", "17850.0", "United Kingdom"),
        ("536366", "22633", "HAND WARMER UNION JACK", "10", "2010-12-03 09:00:00", "1.85", "", "France"),
        # exact duplicate of the first line
        ("536365", "85123A", "WHITE HANGING HEART", "6", "2010-12-01 08:26:00", "2.55", "17850", "United Kingdom"),
        # rejected lines
        ("C536379", "D", "Discount", "-1", "2010-12-01 09:41:00", "27.50", "14527", "United Kingdom"),
        ("536380", "22961", "JAM MAKING SET", "-5", "2010-12-01 10:00:00", "1.45", "", "United Kingdom"),
        ("536381", "22139", "RETROSPOT TEA SET", "2", "2010-12-01 10:00:00", "0", "", "United Kingdom"),
        ("536382", "POST", "POSTAGE", "1", "2010-12-01 10:00:00", "18.00", "12583", "France"),
        ("536383", "22086", "PAPER CHAIN KIT", "abc", "2010-12-01 10:00:00", "2.95", "", "United Kingdom"),
        ("536384", "22086", "PAPER CHAIN KIT", "1", "not a date", "2.95", "", "United Kingdom"),
    ]
