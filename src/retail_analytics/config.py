"""Where the pipeline reads from and writes to, for Databricks and for local runs."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_TABLE = "bronze_online_retail"
SILVER_TABLE = "silver_online_retail"
QUARANTINE_TABLE = "quarantine_online_retail"
GOLD_TABLE = "gold_daily_sales"


@dataclass(frozen=True)
class Settings:
    is_databricks: bool
    raw_csv_path: str
    export_dir: str
    catalog: str = "workspace"
    schema: str = "retail"
    local_delta_dir: str = ""

    @classmethod
    def for_databricks(cls, catalog: str = "workspace", schema: str = "retail") -> "Settings":
        volume = f"/Volumes/{catalog}/{schema}/raw_files"
        return cls(
            is_databricks=True,
            raw_csv_path=f"{volume}/online_retail.csv",
            export_dir=f"{volume}/exports",
            catalog=catalog,
            schema=schema,
        )

    @classmethod
    def for_local(cls, data_dir: Path = PROJECT_ROOT / "data") -> "Settings":
        return cls(
            is_databricks=False,
            raw_csv_path=str(data_dir / "raw" / "online_retail.csv"),
            export_dir=str(data_dir / "exports"),
            local_delta_dir=str(data_dir / "delta"),
        )

    def table_name(self, table: str) -> str:
        """Unity Catalog three-part name, e.g. workspace.retail.gold_daily_sales."""
        return f"{self.catalog}.{self.schema}.{table}"

    def table_path(self, table: str) -> str:
        """Local Delta folder, e.g. data/delta/gold_daily_sales."""
        return str(Path(self.local_delta_dir) / table)
