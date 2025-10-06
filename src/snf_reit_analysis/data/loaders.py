"""Data loaders for CMS, BLS, and SEC APIs."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import config
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseAPILoader:
    """Base class for API data loaders with retry logic."""

    def __init__(self, timeout: int = 30):
        """Initialize with session and retry strategy."""
        self.timeout = timeout
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session


class CMSDataLoader(BaseAPILoader):
    """Loader for CMS Provider Data Catalog API."""

    def __init__(self):
        """Initialize CMS data loader."""
        super().__init__(timeout=config.cms.timeout)
        self.base_url = config.cms.base_url
        self.headers = {"Content-Type": "application/json"}

    def get_dataset_count(self, dataset_id: str) -> int:
        """Get total record count for a dataset."""
        url = f"{self.base_url}/datastore/query/{dataset_id}/0"
        params = {"limit": 1, "offset": 0}
        logger.info(f"Fetching count for dataset: {dataset_id}")

        response = self.session.get(url, params=params, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("count", 0)

    def fetch_dataset(
        self,
        dataset_id: str,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
        use_polars: bool = True
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Fetch complete dataset with pagination.

        Args:
            dataset_id: CMS dataset identifier
            filters: Dictionary of column filters (e.g., {'State': 'CA'})
            columns: List of columns to return
            use_polars: Return Polars DataFrame (faster) vs pandas

        Returns:
            DataFrame with complete dataset
        """
        logger.info(f"Fetching dataset: {dataset_id}")

        # Get total row count
        total_rows = self.get_dataset_count(dataset_id)
        logger.info(f"Total rows in dataset: {total_rows:,}")

        # Build query parameters
        params = {"limit": config.cms.page_size, "offset": 0}

        if filters:
            for col, val in filters.items():
                params[f"filter[{col}]"] = val

        if columns:
            params["properties"] = ",".join(columns)

        # Paginate through data
        all_data = []
        offset = 0
        url = f"{self.base_url}/datastore/query/{dataset_id}/0"

        while offset < total_rows:
            params["offset"] = offset

            logger.debug(f"Fetching rows {offset:,} to {offset + config.cms.page_size:,}")

            response = self.session.get(
                url, params=params, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            all_data.extend(results)
            offset += config.cms.page_size

        logger.info(f"Fetched {len(all_data):,} total rows")

        # Convert to DataFrame
        if use_polars:
            return pl.DataFrame(all_data)
        else:
            return pd.DataFrame(all_data)

    def get_provider_info(
        self,
        state: Optional[str] = None,
        min_rating: Optional[int] = None,
        use_polars: bool = True
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Fetch provider information dataset.

        Args:
            state: Filter by state (e.g., 'CA', 'TX')
            min_rating: Minimum overall rating (1-5)
            use_polars: Return Polars DataFrame

        Returns:
            DataFrame with provider information
        """
        filters = {}
        if state:
            filters["State"] = state
        if min_rating:
            filters["Overall_Rating"] = f">={min_rating}"

        return self.fetch_dataset(
            dataset_id=config.cms.provider_info_dataset,
            filters=filters if filters else None,
            use_polars=use_polars
        )

    def get_health_deficiencies(
        self,
        state: Optional[str] = None,
        use_polars: bool = True
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Fetch health deficiencies dataset.

        Args:
            state: Filter by state
            use_polars: Return Polars DataFrame

        Returns:
            DataFrame with health deficiency data
        """
        filters = {"State": state} if state else None

        return self.fetch_dataset(
            dataset_id=config.cms.deficiencies_dataset,
            filters=filters,
            use_polars=use_polars
        )

    def get_quality_measures(
        self, use_polars: bool = True
    ) -> pl.DataFrame | pd.DataFrame:
        """
        Fetch quality measures dataset.

        Args:
            use_polars: Return Polars DataFrame

        Returns:
            DataFrame with quality measures
        """
        return self.fetch_dataset(
            dataset_id=config.cms.quality_measures_dataset,
            use_polars=use_polars
        )


class BLSDataLoader(BaseAPILoader):
    """Loader for Bureau of Labor Statistics API."""

    def __init__(self):
        """Initialize BLS data loader."""
        super().__init__(timeout=config.bls.timeout)
        self.base_url = config.bls.base_url
        self.api_key = config.bls.api_key.get_secret_value()
        self.headers = {"Content-Type": "application/json"}

    def fetch_series(
        self,
        series_ids: List[str],
        start_year: str,
        end_year: str,
        catalog: bool = True,
        calculations: bool = True,
        annual_average: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch BLS time series data.

        Args:
            series_ids: List of BLS series IDs (max 50)
            start_year: Start year (YYYY)
            end_year: End year (YYYY)
            catalog: Include series catalog metadata
            calculations: Include percent change calculations
            annual_average: Include annual averages

        Returns:
            Dictionary with series data
        """
        if len(series_ids) > config.bls.max_series_per_request:
            raise ValueError(
                f"Maximum {config.bls.max_series_per_request} series per request"
            )

        logger.info(f"Fetching {len(series_ids)} BLS series from {start_year} to {end_year}")

        url = f"{self.base_url}/timeseries/data/"
        payload = {
            "seriesid": series_ids,
            "startyear": start_year,
            "endyear": end_year,
            "catalog": catalog,
            "calculations": calculations,
            "annualaverage": annual_average,
            "registrationkey": self.api_key
        }

        response = self.session.post(
            url, data=json.dumps(payload), headers=self.headers, timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()

        if data["status"] != "REQUEST_SUCCEEDED":
            logger.error(f"BLS API error: {data.get('message', 'Unknown error')}")
            raise ValueError(f"BLS API request failed: {data.get('message')}")

        logger.info(f"Successfully fetched {len(series_ids)} series")
        return data

    def get_nursing_facility_ppi(
        self,
        start_year: str = "2020",
        end_year: str = "2024",
        as_dataframe: bool = True
    ) -> pd.DataFrame | Dict[str, Any]:
        """
        Fetch PPI data for nursing care facilities.

        Args:
            start_year: Start year
            end_year: End year
            as_dataframe: Convert to pandas DataFrame

        Returns:
            DataFrame or raw JSON response
        """
        series_ids = [
            config.bls.ppi_overall,
            config.bls.ppi_medicare,
            config.bls.ppi_medicaid,
            config.bls.ppi_private,
        ]

        data = self.fetch_series(series_ids, start_year, end_year)

        if not as_dataframe:
            return data

        # Convert to DataFrame
        records = []
        for series in data["Results"]["series"]:
            series_id = series["seriesID"]
            series_name = self._get_series_name(series_id)

            for item in series["data"]:
                records.append({
                    "series_id": series_id,
                    "series_name": series_name,
                    "year": item["year"],
                    "period": item["period"],
                    "period_name": item["periodName"],
                    "value": float(item["value"]),
                    "latest": item.get("latest", False)
                })

        return pd.DataFrame(records)

    def get_employment_data(
        self,
        start_year: str = "2020",
        end_year: str = "2024",
        as_dataframe: bool = True
    ) -> pd.DataFrame | Dict[str, Any]:
        """
        Fetch employment data for nursing care facilities.

        Args:
            start_year: Start year
            end_year: End year
            as_dataframe: Convert to pandas DataFrame

        Returns:
            DataFrame or raw JSON response
        """
        series_ids = [
            config.bls.ces_all_employees,
            config.bls.ces_avg_hourly_earnings,
            config.bls.ces_avg_weekly_hours,
            config.bls.ces_avg_weekly_earnings,
        ]

        data = self.fetch_series(series_ids, start_year, end_year)

        if not as_dataframe:
            return data

        # Convert to DataFrame
        records = []
        for series in data["Results"]["series"]:
            series_id = series["seriesID"]
            series_name = self._get_series_name(series_id)

            for item in series["data"]:
                records.append({
                    "series_id": series_id,
                    "series_name": series_name,
                    "year": item["year"],
                    "period": item["period"],
                    "period_name": item["periodName"],
                    "value": float(item["value"]),
                    "latest": item.get("latest", False)
                })

        return pd.DataFrame(records)

    def _get_series_name(self, series_id: str) -> str:
        """Map series ID to friendly name."""
        series_map = {
            config.bls.ppi_overall: "PPI Overall",
            config.bls.ppi_medicare: "PPI Medicare",
            config.bls.ppi_medicaid: "PPI Medicaid",
            config.bls.ppi_private: "PPI Private Insurance",
            config.bls.ces_all_employees: "Total Employees",
            config.bls.ces_avg_hourly_earnings: "Avg Hourly Earnings",
            config.bls.ces_avg_weekly_hours: "Avg Weekly Hours",
            config.bls.ces_avg_weekly_earnings: "Avg Weekly Earnings",
        }
        return series_map.get(series_id, series_id)


class SECDataLoader(BaseAPILoader):
    """Loader for SEC EDGAR API."""

    def __init__(self):
        """Initialize SEC data loader."""
        super().__init__(timeout=config.sec.timeout)
        self.base_url = config.sec.base_url
        self.headers = {"User-Agent": config.sec.user_agent}
        self.rate_limit_delay = config.sec.rate_limit_delay

    def _rate_limit(self) -> None:
        """Enforce SEC rate limit of 10 requests per second."""
        time.sleep(self.rate_limit_delay)

    def get_company_submissions(self, cik: int) -> Dict[str, Any]:
        """
        Get complete filing history for a company.

        Args:
            cik: Company CIK code

        Returns:
            Dictionary with submission data
        """
        self._rate_limit()
        url = f"{self.base_url}/submissions/CIK{cik:010d}.json"
        logger.info(f"Fetching submissions for CIK {cik:010d}")

        response = self.session.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()

        return response.json()

    def get_company_facts(self, cik: int) -> Dict[str, Any]:
        """
        Get all XBRL facts for a company.

        Args:
            cik: Company CIK code

        Returns:
            Dictionary with all XBRL data
        """
        self._rate_limit()
        url = f"{self.base_url}/api/xbrl/companyfacts/CIK{cik:010d}.json"
        logger.info(f"Fetching company facts for CIK {cik:010d}")

        response = self.session.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()

        return response.json()

    def get_company_concept(
        self, cik: int, taxonomy: str, tag: str
    ) -> Dict[str, Any]:
        """
        Get specific XBRL concept across all filings.

        Args:
            cik: Company CIK code
            taxonomy: Taxonomy (e.g., 'us-gaap', 'dei')
            tag: XBRL tag name

        Returns:
            Dictionary with concept data
        """
        self._rate_limit()
        url = f"{self.base_url}/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{tag}.json"
        logger.info(f"Fetching {taxonomy}:{tag} for CIK {cik:010d}")

        response = self.session.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()

        return response.json()

    def extract_metric(
        self,
        facts_data: Dict[str, Any],
        tag: str,
        unit: str = "USD",
        form: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract specific financial metric from company facts.

        Args:
            facts_data: Company facts dictionary from get_company_facts()
            tag: XBRL tag name (e.g., 'Assets', 'Revenues')
            unit: Unit type (USD, shares, pure)
            form: Filing form type (10-K for annual, 10-Q for quarterly, None for all)

        Returns:
            DataFrame with metric values over time
        """
        try:
            data = facts_data["facts"]["us-gaap"][tag]["units"][unit]
            df = pd.DataFrame(data)

            # Filter by form type if specified
            if form:
                df_filtered = df[df["form"] == form].copy()
            else:
                # Include both 10-K and 10-Q filings
                df_filtered = df[df["form"].isin(["10-K", "10-Q"])].copy()

            df_filtered = df_filtered.sort_values("end", ascending=False)

            return df_filtered

        except KeyError as e:
            logger.warning(f"Metric not found: {tag} in unit {unit}")
            return pd.DataFrame()

    def get_reit_financials(
        self, ticker: str, tags: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get financial metrics for a REIT.

        Args:
            ticker: REIT ticker symbol ('OHI', 'CTRE', 'SBRA')
            tags: List of XBRL tags to extract

        Returns:
            Dictionary mapping tag names to DataFrames
        """
        if ticker not in config.sec.reit_ciks:
            raise ValueError(f"Unknown REIT ticker: {ticker}")

        cik = config.sec.reit_ciks[ticker]
        logger.info(f"Fetching financials for {ticker} (CIK {cik})")

        # Default tags for REIT analysis
        if tags is None:
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

        # Get all company facts
        facts = self.get_company_facts(cik)

        # Extract each metric
        results = {}
        for tag in tags:
            df = self.extract_metric(facts, tag)
            if not df.empty:
                results[tag] = df
                logger.info(f"Extracted {tag}: {len(df)} records")
            else:
                logger.warning(f"No data found for {tag}")

        return results

    def get_all_reits_financials(
        self, tags: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get financial metrics for all three REITs.

        Args:
            tags: List of XBRL tags to extract

        Returns:
            Nested dictionary: {ticker: {tag: DataFrame}}
        """
        results = {}

        for ticker in ["OHI", "CTRE", "SBRA"]:
            try:
                results[ticker] = self.get_reit_financials(ticker, tags)
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                results[ticker] = {}

        return results
