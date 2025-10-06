"""
REIT Property Matcher - Link REIT portfolios to CMS facility data.

Matches REIT properties to CMS facilities via chain names. Note: chain names
identify operators, not owners. Many operators serve multiple REITs.
For 100% accuracy, use facility-level address matching.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import polars as pl

from ..utils.logging import get_logger

logger = get_logger(__name__)


# REIT Chain Mappings
# Sources: SEC 10-K filings, fuzzy matching, investor materials
# Coverage: OHI ~37%, CTRE ~52%, SBRA ~75%
# Limitation: Operators serve multiple REITs (Ensign, PACS, Genesis)

REIT_CHAIN_MAPPINGS = {
    'OHI': [
        # Top operators by bed count
        'AMERICAN SENIOR COMMUNITIES',
        'SABER HEALTHCARE GROUP',
        'CIENA HEALTHCARE/LAUREL HEALTH CARE',
        'SIGNATURE HEALTHCARE',
        'GENESIS HEALTHCARE',
        'PRESTIGE ADMINISTRATIVE SERVICES',
        'CREATIVE SOLUTIONS IN HEALTHCARE',
        'PACS GROUP',
        'COMMUNICARE HEALTH',
        'HEARTHSTONE SENIOR COMMUNITIES',
        'THOMAS CHAMBERS & DAVID JOHNSON',
        'CONSULATE HEALTH CARE/INDEPENDENCE LIVING CENTERS/NSPIRE HEALTHCARE/RAYDIANT HEALTH CARE',
        'EXCEPTIONAL LIVING CENTERS',
        'FUNDAMENTAL HEALTHCARE',
        'GENERATIONS HEALTHCARE',
        'BROOKDALE SENIOR LIVING',
        'CASSENA CARE',
        'CCH HEALTHCARE',
        'EMPRES OPERATED BY EVERGREEN',
        'ENVIVE HEALTHCARE',
        'FLORIDA INSTITUTE FOR LONG-TERM CARE',
        'LEGACY HEALTH SERVICES',
        'NEXION HEALTH',
        'OPCO SKILLED MANAGEMENT',
        'PACS',
        'PRESTIGE CARE',
        'SENIOR HEALTH SOUTH',
        'SKLD',
        'SOUTHERN ADMINISTRATIVE SERVICES',
        'THE ENSIGN GROUP',
        'WELLINGTON HEALTH CARE SERVICES',
        'ANTHONY & BRYAN ADAMS',
        'INFINITY HEALTHCARE CONSULTING',
        'JAMES & JUDY LINCOLN',
        'KEY HEALTH MANAGEMENT',
        'BUCKNER RETIREMENT SERVICES',
        'COMMUNITY ELDERCARE SERVICES',
        'AKIKO IKE',
        'VENZA CARE MANAGEMENT',
        'SUMMITT CARE II, INC.',
    ],

    'CTRE': [
        'THE ENSIGN GROUP',
        'PACS GROUP',
        'PRIORITY MANAGEMENT',
        'CASCADIA HEALTHCARE',
        'CIENA HEALTHCARE/LAUREL HEALTH CARE',
        'FIVE STAR SENIOR LIVING',
        'PRISTINE SENIOR LIVING',
        'PENNANT GROUP',
    ],

    'SBRA': [
        'GENESIS HEALTHCARE',
        'AVAMERE',
        'CONSULATE HEALTH CARE/INDEPENDENCE LIVING CENTERS/NSPIRE HEALTHCARE/RAYDIANT HEALTH CARE',
        'CADIA HEALTHCARE',
        'DISCOVERY SENIOR LIVING',
        'INSPIRIT SENIOR LIVING',
        'SUNSHINE RETIREMENT LIVING',
        'HOLIDAY SENIOR LIVING',
        'AVAMERE HEALTH SERVICES',
        'LEGACY LIVING',
    ]
}


@dataclass
class REITPortfolio:
    """Portfolio metrics for a REIT."""
    reit_ticker: str
    total_facilities: int
    total_beds: int
    avg_occupancy_rate: float
    avg_quality_rating: float
    states: List[str]
    operators: List[str]
    geographic_distribution: Dict[str, int]
    quality_distribution: Dict[int, int]


class REITPropertyMatcher:
    """Match REIT properties to CMS facilities via chain names."""

    def __init__(self, cms_data_path: str = "data/processed/cms_provider_info_latest.csv"):
        """
        Initialize matcher with CMS data.

        Args:
            cms_data_path: Path to processed CMS provider info data
        """
        self.cms_data_path = Path(cms_data_path)
        self.cms_data: Optional[pl.DataFrame] = None
        self._load_cms_data()

    def _load_cms_data(self) -> None:
        """Load and prepare CMS provider data."""
        if not self.cms_data_path.exists():
            raise FileNotFoundError(f"CMS data not found at {self.cms_data_path}")

        logger.info(f"Loading CMS data from {self.cms_data_path}")

        # Use Polars for faster loading and processing
        self.cms_data = pl.read_csv(self.cms_data_path, infer_schema_length=50000)

        # Calculate occupancy rate if not already present
        if 'occupancy_rate' not in self.cms_data.columns:
            self.cms_data = self.cms_data.with_columns([
                (pl.col('average_number_of_residents_per_day') /
                 pl.col('number_of_certified_beds') * 100).alias('occupancy_rate')
            ])

        logger.info(f"Loaded {len(self.cms_data):,} facilities from CMS")

    def find_reit_facilities(
        self,
        reit_ticker: str,
        chain_patterns: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Find facilities owned by a specific REIT.

        Args:
            reit_ticker: REIT ticker symbol (OHI, CTRE, SBRA)
            chain_patterns: List of chain name patterns to match.
                          If None, uses REIT_CHAIN_MAPPINGS

        Returns:
            DataFrame of facilities owned by the REIT
        """
        if chain_patterns is None:
            chain_patterns = REIT_CHAIN_MAPPINGS.get(reit_ticker.upper(), [])

        if not chain_patterns:
            logger.warning(f"No chain patterns defined for {reit_ticker}")
            return pl.DataFrame()

        # Create regex pattern for matching (case-insensitive)
        pattern = '|'.join(chain_patterns)

        # Filter facilities by chain name
        reit_facilities = self.cms_data.filter(
            pl.col('chain_name').str.to_uppercase().str.contains(pattern.upper())
        )

        logger.info(f"Found {len(reit_facilities):,} facilities for {reit_ticker}")

        return reit_facilities

    def get_reit_portfolio_metrics(self, reit_ticker: str) -> REITPortfolio:
        """
        Calculate comprehensive portfolio metrics for a REIT.

        Args:
            reit_ticker: REIT ticker symbol

        Returns:
            REITPortfolio dataclass with aggregated metrics
        """
        facilities = self.find_reit_facilities(reit_ticker)

        if len(facilities) == 0:
            logger.warning(f"No facilities found for {reit_ticker}")
            return REITPortfolio(
                reit_ticker=reit_ticker,
                total_facilities=0,
                total_beds=0,
                avg_occupancy_rate=0.0,
                avg_quality_rating=0.0,
                states=[],
                operators=[],
                geographic_distribution={},
                quality_distribution={}
            )

        # Calculate metrics
        total_facilities = len(facilities)
        total_beds = facilities['number_of_certified_beds'].sum()

        # Bed-weighted occupancy rate
        facilities_with_occ = facilities.filter(pl.col('occupancy_rate').is_not_null())
        if len(facilities_with_occ) > 0:
            avg_occupancy = (
                facilities_with_occ['occupancy_rate'] *
                facilities_with_occ['number_of_certified_beds']
            ).sum() / facilities_with_occ['number_of_certified_beds'].sum()
        else:
            avg_occupancy = 0.0

        # Bed-weighted quality rating
        facilities_with_rating = facilities.filter(pl.col('overall_rating').is_not_null())
        if len(facilities_with_rating) > 0:
            avg_rating = (
                facilities_with_rating['overall_rating'] *
                facilities_with_rating['number_of_certified_beds']
            ).sum() / facilities_with_rating['number_of_certified_beds'].sum()
        else:
            avg_rating = 0.0

        # Geographic distribution
        geo_dist = (
            facilities
            .group_by('state')
            .agg(pl.len().alias('count'))
            .sort('count', descending=True)
        )
        geo_dict = dict(zip(geo_dist['state'].to_list(), geo_dist['count'].to_list()))

        # Quality distribution
        quality_dist = (
            facilities
            .filter(pl.col('overall_rating').is_not_null())
            .group_by('overall_rating')
            .agg(pl.len().alias('count'))
            .sort('overall_rating')
        )
        quality_dict = dict(zip(
            quality_dist['overall_rating'].to_list(),
            quality_dist['count'].to_list()
        ))

        # Unique operators and states
        operators = facilities['chain_name'].unique().to_list()
        states = facilities['state'].unique().to_list()

        return REITPortfolio(
            reit_ticker=reit_ticker,
            total_facilities=total_facilities,
            total_beds=total_beds,
            avg_occupancy_rate=float(avg_occupancy),
            avg_quality_rating=float(avg_rating),
            states=states,
            operators=operators,
            geographic_distribution=geo_dict,
            quality_distribution=quality_dict
        )

    def compare_reit_portfolios(self, reit_tickers: List[str]) -> pd.DataFrame:
        """
        Compare portfolio metrics across multiple REITs.

        Args:
            reit_tickers: List of REIT ticker symbols

        Returns:
            DataFrame with comparative metrics
        """
        portfolios = [self.get_reit_portfolio_metrics(ticker) for ticker in reit_tickers]

        comparison = pd.DataFrame([
            {
                'REIT': p.reit_ticker,
                'Facilities': p.total_facilities,
                'Total Beds': p.total_beds,
                'Avg Beds per Facility': p.total_beds / p.total_facilities if p.total_facilities > 0 else 0,
                'Avg Occupancy (%)': p.avg_occupancy_rate,
                'Avg Quality Rating': p.avg_quality_rating,
                'States': len(p.states),
                'Operators': len(p.operators),
                'Top State': list(p.geographic_distribution.keys())[0] if p.geographic_distribution else None,
                'Top State %': (
                    list(p.geographic_distribution.values())[0] / p.total_facilities * 100
                    if p.geographic_distribution and p.total_facilities > 0 else 0
                )
            }
            for p in portfolios
        ])

        return comparison

    def get_regional_overlap(self, reit_tickers: List[str]) -> pd.DataFrame:
        """
        Analyze geographic overlap between REITs.

        Args:
            reit_tickers: List of REIT ticker symbols

        Returns:
            DataFrame showing state-level overlap
        """
        portfolios = {
            ticker: self.get_reit_portfolio_metrics(ticker)
            for ticker in reit_tickers
        }

        # Get all states
        all_states = set()
        for portfolio in portfolios.values():
            all_states.update(portfolio.states)

        # Build overlap matrix
        overlap_data = []
        for state in sorted(all_states):
            row = {'State': state}
            for ticker in reit_tickers:
                geo_dist = portfolios[ticker].geographic_distribution
                row[ticker] = geo_dist.get(state, 0)
            overlap_data.append(row)

        return pd.DataFrame(overlap_data)

    def get_operator_tenant_mix(self, reit_ticker: str) -> pd.DataFrame:
        """
        Analyze operator/tenant mix for a REIT.

        Args:
            reit_ticker: REIT ticker symbol

        Returns:
            DataFrame with operator-level metrics
        """
        facilities = self.find_reit_facilities(reit_ticker)

        if len(facilities) == 0:
            return pd.DataFrame()

        # Group by operator (chain_name)
        operator_stats = (
            facilities
            .group_by('chain_name')
            .agg([
                pl.len().alias('facilities'),
                pl.sum('number_of_certified_beds').alias('total_beds'),
                pl.mean('occupancy_rate').alias('avg_occupancy'),
                pl.mean('overall_rating').alias('avg_quality_rating'),
                pl.col('state').n_unique().alias('states'),
            ])
            .sort('total_beds', descending=True)
        )

        # Calculate concentration
        total_facilities = facilities.select(pl.len()).item()
        operator_df = operator_stats.to_pandas()
        operator_df['facility_concentration_%'] = (
            operator_df['facilities'] / total_facilities * 100
        )

        return operator_df


def create_reit_property_database(
    output_path: str = "data/processed/reit_properties.parquet"
) -> pl.DataFrame:
    """
    Create comprehensive database linking all REITs to their properties.

    Args:
        output_path: Path to save the combined database

    Returns:
        Combined DataFrame with all REIT properties
    """
    matcher = REITPropertyMatcher()

    all_properties = []

    for reit_ticker in ['OHI', 'CTRE', 'SBRA']:
        facilities = matcher.find_reit_facilities(reit_ticker)

        if len(facilities) > 0:
            # Add REIT identifier
            facilities = facilities.with_columns(
                pl.lit(reit_ticker).alias('reit_owner')
            )
            all_properties.append(facilities)

    if all_properties:
        # Combine all properties
        combined = pl.concat(all_properties)

        # Save to parquet
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(output_path)

        logger.info(f"Saved REIT property database to {output_path}")
        logger.info(f"Total properties: {len(combined):,}")

        return combined

    return pl.DataFrame()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Initialize matcher
    matcher = REITPropertyMatcher()

    # Compare REITs
    comparison = matcher.compare_reit_portfolios(['OHI', 'CTRE', 'SBRA'])
    print("\n=== REIT Portfolio Comparison ===")
    print(comparison.to_string(index=False))

    # Geographic overlap
    overlap = matcher.get_regional_overlap(['OHI', 'CTRE', 'SBRA'])
    print("\n=== Geographic Overlap (Top 10 States) ===")
    print(overlap.head(10).to_string(index=False))

    # Create comprehensive database
    db = create_reit_property_database()
    print(f"\n=== Created property database with {len(db):,} facilities ===")
