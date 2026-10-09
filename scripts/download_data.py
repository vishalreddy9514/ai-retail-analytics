"""Download the UCI Online Retail dataset and convert it to data/raw/online_retail.csv.

UCI publishes the data as an Excel file inside a zip. Converting it once here gives the
pipeline a plain CSV with predictable formats (ISO timestamps, integer customer IDs).

Usage:
    python scripts/download_data.py                       # download from UCI
    python scripts/download_data.py --xlsx "Online Retail.xlsx"   # use a file you downloaded
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
XLSX_NAME = "Online Retail.xlsx"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
CSV_PATH = RAW_DIR / "online_retail.csv"


def download_xlsx(dest_dir: Path) -> Path:
    print(f"Downloading {UCI_ZIP_URL} ...")
    with urllib.request.urlopen(UCI_ZIP_URL, timeout=120) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    archive.extract(XLSX_NAME, dest_dir)
    return dest_dir / XLSX_NAME


def xlsx_to_csv(xlsx_path: Path, csv_path: Path) -> pd.DataFrame:
    print(f"Reading {xlsx_path} (takes ~1 minute) ...")
    df = pd.read_excel(xlsx_path, dtype={"InvoiceNo": str, "StockCode": str, "Description": str})
    df["CustomerID"] = df["CustomerID"].astype("Int64")  # 17850.0 -> 17850, NaN -> empty
    df.to_csv(csv_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, help="Use an already downloaded 'Online Retail.xlsx'")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    xlsx_path = args.xlsx or download_xlsx(RAW_DIR)
    df = xlsx_to_csv(xlsx_path, CSV_PATH)
    print(f"Wrote {len(df):,} rows and {len(df.columns)} columns to {CSV_PATH}")
    print(f"Invoice dates: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")


if __name__ == "__main__":
    main()
