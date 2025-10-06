"""Helper functions for data processing and analysis."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl


def save_dataframe(
    df: pd.DataFrame | pl.DataFrame,
    file_path: Path,
    format: str = "parquet"
) -> None:
    """Save DataFrame to file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df, pl.DataFrame):
        if format == "parquet":
            df.write_parquet(file_path)
        elif format == "csv":
            df.write_csv(file_path)
        elif format == "excel":
            df.to_pandas().to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format for Polars: {format}")
    else:
        if format == "parquet":
            df.to_parquet(file_path, index=False)
        elif format == "csv":
            df.to_csv(file_path, index=False)
        elif format == "excel":
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")


def load_dataframe(
    file_path: Path,
    use_polars: bool = True
) -> pd.DataFrame | pl.DataFrame:
    """Load DataFrame from file."""
    suffix = file_path.suffix.lower()

    if use_polars:
        if suffix == ".parquet":
            return pl.read_parquet(file_path)
        elif suffix == ".csv":
            return pl.read_csv(file_path)
        elif suffix in [".xlsx", ".xls"]:
            return pl.from_pandas(pd.read_excel(file_path))
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    else:
        if suffix == ".parquet":
            return pd.read_parquet(file_path)
        elif suffix == ".csv":
            return pd.read_csv(file_path)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


def calculate_occupancy_rate(
    census: float | int,
    beds: float | int
) -> float:
    """Calculate occupancy rate."""
    if beds <= 0:
        return 0.0
    return min(census / beds, 1.0)


def calculate_debt_to_assets(
    total_debt: float,
    total_assets: float
) -> float:
    """Calculate debt-to-assets ratio."""
    if total_assets <= 0:
        return 0.0
    return total_debt / total_assets


def calculate_roa(
    net_income: float,
    total_assets: float
) -> float:
    """Calculate Return on Assets (ROA)."""
    if total_assets <= 0:
        return 0.0
    return net_income / total_assets


def get_latest_quarter_date() -> str:
    """Get the latest completed fiscal quarter end date."""
    today = datetime.now()
    year = today.year
    month = today.month

    # Determine last completed quarter
    if month >= 10:
        quarter_end = f"{year}-09-30"  # Q3
    elif month >= 7:
        quarter_end = f"{year}-06-30"  # Q2
    elif month >= 4:
        quarter_end = f"{year}-03-31"  # Q1
    else:
        quarter_end = f"{year - 1}-12-31"  # Previous year Q4

    return quarter_end


def format_currency(value: float) -> str:
    """Format number as currency string."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"${value / 1e3:.2f}K"
    else:
        return f"${value:.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format decimal as percentage string."""
    return f"{value * 100:.{decimals}f}%"
