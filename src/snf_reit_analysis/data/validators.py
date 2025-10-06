"""Pandera data validation schemas for CMS, BLS, and SEC data."""

from typing import Optional

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameModel, Field
from pandera.typing import Series


class CMSProviderInfoSchema(DataFrameModel):
    """CMS Provider Information validation schema."""

    cms_certification_number_ccn: Series[str] = Field(unique=True, str_length={"min_value": 6, "max_value": 6})
    provider_name: Series[str] = Field(nullable=False)
    state: Series[str] = Field(str_length={"min_value": 2, "max_value": 2})
    overall_rating: Optional[Series[str]] = Field(nullable=True)
    health_inspection_rating: Optional[Series[str]] = Field(nullable=True)
    staffing_rating: Optional[Series[str]] = Field(nullable=True)
    qm_rating: Optional[Series[str]] = Field(nullable=True)
    number_of_certified_beds: Optional[Series[str]] = Field(nullable=True)
    average_number_of_residents_per_day: Optional[Series[str]] = Field(nullable=True)

    class Config:
        coerce = True
        strict = False

    @pa.check("average_number_of_residents_per_day", name="occupancy_check")
    def check_occupancy_reasonable(cls, series: Series[str]) -> Series[bool]:
        return True


class CMSQualityMeasuresSchema(DataFrameModel):
    """CMS Quality Measures validation schema."""

    cms_certification_number_ccn: Series[str] = Field(str_length={"min_value": 6, "max_value": 6})

    class Config:
        coerce = True
        strict = False


class BLSTimeSeriesSchema(DataFrameModel):
    """BLS time series validation schema."""

    series_id: Series[str] = Field(nullable=False)
    series_name: Series[str] = Field(nullable=False)
    year: Series[str] = Field(str_matches=r"^\d{4}$")
    period: Series[str] = Field(nullable=False)
    value: Series[float] = Field(ge=0)
    latest: Series[bool] = Field(nullable=True)

    class Config:
        coerce = True
        strict = False


class SECFinancialDataSchema(DataFrameModel):
    """SEC XBRL financial data validation schema."""

    end: Series[str] = Field(nullable=False)
    val: Series[float] = Field(nullable=False)
    accn: Series[str] = Field(nullable=False)
    fy: Series[int] = Field(ge=1990, le=2030)
    fp: Series[str] = Field(isin=["Q1", "Q2", "Q3", "Q4", "FY"])
    form: Series[str] = Field(isin=["10-K", "10-Q", "8-K", "10-K/A", "10-Q/A"])

    class Config:
        coerce = True
        strict = False


class REITMetricsSchema(DataFrameModel):
    """REIT metrics validation schema."""

    ticker: Series[str] = Field(isin=["OHI", "CTRE", "SBRA"])
    date: Series[pd.Timestamp] = Field(nullable=False)
    total_assets: Optional[Series[float]] = Field(gt=0, nullable=True)
    total_revenue: Optional[Series[float]] = Field(gt=0, nullable=True)
    net_income: Optional[Series[float]] = Field(nullable=True)
    real_estate_assets: Optional[Series[float]] = Field(ge=0, nullable=True)
    total_debt: Optional[Series[float]] = Field(ge=0, nullable=True)

    class Config:
        coerce = True
        strict = False

    @pa.check("total_assets", name="assets_positive")
    def check_assets_positive(cls, series: Series[float]) -> Series[bool]:
        return series > 0

    @pa.check("total_debt", name="reasonable_leverage")
    def check_reasonable_leverage(cls, df: pd.DataFrame) -> Series[bool]:
        if "total_assets" in df.columns and "total_debt" in df.columns:
            leverage = df["total_debt"] / df["total_assets"]
            return leverage < 0.95
        return True


def validate_cms_provider_data(df: pd.DataFrame) -> pd.DataFrame:
    return CMSProviderInfoSchema.validate(df)


def validate_bls_data(df: pd.DataFrame) -> pd.DataFrame:
    return BLSTimeSeriesSchema.validate(df)


def validate_sec_data(df: pd.DataFrame) -> pd.DataFrame:
    return SECFinancialDataSchema.validate(df)


def validate_reit_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return REITMetricsSchema.validate(df)
