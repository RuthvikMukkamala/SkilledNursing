"""ETL pipeline with Write-Audit-Publish pattern."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl
from pandera.errors import SchemaError

from ..config import config
from ..data.loaders import BLSDataLoader, CMSDataLoader, SECDataLoader
from ..data.validators import (
    validate_bls_data,
    validate_cms_provider_data,
    validate_sec_data,
)
from ..utils.helpers import save_dataframe
from ..utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class ValidationResult:
    """Data validation result."""

    def __init__(self, passed: bool, errors: Optional[List[str]] = None):
        self.passed = passed
        self.errors = errors or []

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"ValidationResult(status={status}, errors={len(self.errors)})"


class WAPPipeline:
    """Write-Audit-Publish pipeline."""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.staging_dir = config.interim_data_dir / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def write_to_staging(
        self,
        df: pd.DataFrame | pl.DataFrame,
        data_name: str
    ) -> str:
        """Write data to staging."""
        staging_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{data_name}_{timestamp}_{staging_id[:8]}.csv"
        staging_path = self.staging_dir / file_name

        logger.info(f"Writing {data_name} to staging: {staging_path}")
        save_dataframe(df, staging_path, format="csv")

        return staging_id

    def audit(
        self,
        df: pd.DataFrame,
        validator_func: Any,
        data_name: str
    ) -> ValidationResult:
        """Audit data using Pandera validation."""
        logger.info(f"Auditing {data_name}...")

        try:
            validated_df = validator_func(df)
            logger.info(f"✓ {data_name} validation passed")
            return ValidationResult(passed=True)

        except SchemaError as e:
            logger.error(f"✗ {data_name} validation failed: {e}")
            errors = [str(e)]
            return ValidationResult(passed=False, errors=errors)

        except Exception as e:
            logger.error(f"✗ Unexpected validation error for {data_name}: {e}")
            return ValidationResult(passed=False, errors=[str(e)])

    def publish_to_production(
        self,
        df: pd.DataFrame | pl.DataFrame,
        data_name: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """Publish validated data to production."""
        if output_dir is None:
            output_dir = config.processed_data_dir

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{data_name}_{timestamp}.csv"
        output_path = output_dir / file_name

        logger.info(f"Publishing {data_name} to production: {output_path}")
        save_dataframe(df, output_path, format="csv")

        # Also save latest version without timestamp
        latest_path = output_dir / f"{data_name}_latest.csv"
        save_dataframe(df, latest_path, format="csv")
        logger.info(f"Updated latest: {latest_path}")

        return output_path

    def run(
        self,
        df: pd.DataFrame | pl.DataFrame,
        validator_func: Any,
        data_name: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """Run full Write-Audit-Publish pipeline."""
        logger.info(f"Starting WAP pipeline for {data_name}")

        # Convert Polars to pandas for validation
        df_pandas = df.to_pandas() if isinstance(df, pl.DataFrame) else df

        # Write to staging
        staging_id = self.write_to_staging(df, data_name)

        # Audit
        validation_result = self.audit(df_pandas, validator_func, data_name)

        if not validation_result.passed:
            error_msg = f"Validation failed for {data_name}: {validation_result.errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Publish
        output_path = self.publish_to_production(df, data_name, output_dir)

        logger.info(f"✓ WAP pipeline completed for {data_name}")
        return output_path


def run_cms_pipeline(
    state: Optional[str] = None,
    min_rating: Optional[int] = None
) -> Dict[str, Path]:
    """Run CMS data pipeline."""
    logger.info("=" * 50)
    logger.info("Starting CMS Data Pipeline")
    logger.info("=" * 50)

    loader = CMSDataLoader()
    pipeline = WAPPipeline("cms")
    results = {}

    # Fetch provider information
    logger.info("Fetching CMS Provider Information...")
    provider_df = loader.get_provider_info(
        state=state,
        min_rating=min_rating,
        use_polars=False  # Need pandas for validation
    )
    logger.info(f"Fetched {len(provider_df):,} provider records")

    # Run WAP pipeline
    results["provider_info"] = pipeline.run(
        provider_df,
        validate_cms_provider_data,
        "cms_provider_info"
    )

    logger.info("✓ CMS Pipeline completed successfully")
    return results


def run_bls_pipeline(
    start_year: str = "2020",
    end_year: str = "2024"
) -> Dict[str, Path]:
    """Run BLS data pipeline."""
    logger.info("=" * 50)
    logger.info("Starting BLS Data Pipeline")
    logger.info("=" * 50)

    loader = BLSDataLoader()
    pipeline = WAPPipeline("bls")
    results = {}

    # Fetch PPI data
    logger.info("Fetching BLS PPI data...")
    ppi_df = loader.get_nursing_facility_ppi(
        start_year=start_year,
        end_year=end_year,
        as_dataframe=True
    )
    logger.info(f"Fetched {len(ppi_df):,} PPI records")

    results["ppi"] = pipeline.run(
        ppi_df,
        validate_bls_data,
        "bls_ppi"
    )

    # Fetch employment data
    logger.info("Fetching BLS employment data...")
    employment_df = loader.get_employment_data(
        start_year=start_year,
        end_year=end_year,
        as_dataframe=True
    )
    logger.info(f"Fetched {len(employment_df):,} employment records")

    results["employment"] = pipeline.run(
        employment_df,
        validate_bls_data,
        "bls_employment"
    )

    logger.info("✓ BLS Pipeline completed successfully")
    return results


def run_sec_pipeline(
    tickers: Optional[List[str]] = None
) -> Dict[str, Path]:
    """Run SEC EDGAR data pipeline."""
    logger.info("=" * 50)
    logger.info("Starting SEC EDGAR Data Pipeline")
    logger.info("=" * 50)

    if tickers is None:
        tickers = ["OHI", "CTRE", "SBRA"]

    loader = SECDataLoader()
    pipeline = WAPPipeline("sec")
    results = {}

    # Financial metrics to extract
    tags = [
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "Revenues",
        "NetIncomeLoss",
        "RealEstateInvestmentPropertyNet",
        "LongTermDebt",
        "OperatingIncomeLoss",
    ]

    for ticker in tickers:
        logger.info(f"Fetching financial data for {ticker}...")

        financials = loader.get_reit_financials(ticker, tags)

        # Process each financial metric
        for tag, df in financials.items():
            if not df.empty:
                data_name = f"sec_{ticker.lower()}_{tag.lower()}"

                results[data_name] = pipeline.run(
                    df,
                    validate_sec_data,
                    data_name
                )

    logger.info("✓ SEC Pipeline completed successfully")
    return results


def run_full_pipeline(
    cms_state: Optional[str] = None,
    bls_start_year: str = "2020",
    bls_end_year: str = "2024",
    sec_tickers: Optional[List[str]] = None
) -> Dict[str, Dict[str, Path]]:
    """Run complete ETL pipeline for all data sources."""
    setup_logging()
    logger.info("=" * 60)
    logger.info("STARTING FULL SNF REIT ANALYSIS PIPELINE")
    logger.info("=" * 60)

    results = {
        "cms": {},
        "bls": {},
        "sec": {}
    }

    try:
        # CMS Pipeline
        results["cms"] = run_cms_pipeline(state=cms_state)

        # BLS Pipeline
        results["bls"] = run_bls_pipeline(
            start_year=bls_start_year,
            end_year=bls_end_year
        )

        # SEC Pipeline
        results["sec"] = run_sec_pipeline(tickers=sec_tickers)

        logger.info("=" * 60)
        logger.info("✓ FULL PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"✗ Pipeline failed: {e}", exc_info=True)
        raise

    return results


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="SNF REIT Analysis ETL Pipeline")
    parser.add_argument(
        "--source",
        choices=["cms", "bls", "sec", "all"],
        default="all",
        help="Data source to process"
    )
    parser.add_argument("--cms-state", help="Filter CMS data by state")
    parser.add_argument("--bls-start-year", default="2020", help="BLS start year")
    parser.add_argument("--bls-end-year", default="2024", help="BLS end year")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    # Run pipelines
    if args.source == "cms":
        run_cms_pipeline(state=args.cms_state)
    elif args.source == "bls":
        run_bls_pipeline(
            start_year=args.bls_start_year,
            end_year=args.bls_end_year
        )
    elif args.source == "sec":
        run_sec_pipeline()
    else:  # all
        run_full_pipeline(
            cms_state=args.cms_state,
            bls_start_year=args.bls_start_year,
            bls_end_year=args.bls_end_year
        )


if __name__ == "__main__":
    main()
