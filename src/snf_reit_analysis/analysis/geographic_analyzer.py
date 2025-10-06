"""
Geographic Analysis Module - Connect BLS Regional Data to REIT Portfolios

This module analyzes geographic distribution of REIT portfolios and connects
them to regional economic indicators from BLS data.

Key Capabilities:
1. State-level REIT portfolio concentration
2. Regional BLS data mapping (PPI, employment by metro area)
3. Market attractiveness scoring
4. Geographic risk/opportunity analysis
"""

import polars as pl
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegionalMetrics:
    """Regional market metrics combining REIT + BLS data"""
    state: str
    reit_facilities: int
    reit_beds: int
    avg_occupancy: float
    avg_quality_rating: float
    ppi_growth_rate: Optional[float] = None
    employment_growth_rate: Optional[float] = None
    wage_growth_rate: Optional[float] = None
    market_attractiveness_score: Optional[float] = None


# State to Census Region mapping for BLS data
STATE_TO_REGION = {
    # Northeast
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
    'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast', 'PA': 'Northeast',

    # Midwest
    'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest', 'WI': 'Midwest',
    'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest', 'MO': 'Midwest', 'NE': 'Midwest',
    'ND': 'Midwest', 'SD': 'Midwest',

    # South
    'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South',
    'SC': 'South', 'VA': 'South', 'WV': 'South', 'AL': 'South', 'KY': 'South',
    'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South', 'OK': 'South', 'TX': 'South',
    'DC': 'South',

    # West
    'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
    'NM': 'West', 'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West',
    'HI': 'West', 'OR': 'West', 'WA': 'West'
}

# Medicare reimbursement by state (sample - should be updated with actual data)
# Source: CMS State Average Payment Rates
MEDICARE_REIMBURSEMENT_INDEX = {
    'AK': 1.67, 'CA': 1.15, 'NY': 1.12, 'MA': 1.10, 'CT': 1.08,
    'NJ': 1.07, 'IL': 1.05, 'TX': 0.95, 'FL': 0.93, 'AL': 0.85,
    # Add all states - this is illustrative
}


class GeographicAnalyzer:
    """
    Analyze geographic distribution and regional economics for REITs
    """

    def __init__(self,
                 reit_properties_path: str = "data/processed/reit_properties.parquet",
                 bls_ppi_path: Optional[str] = None,
                 bls_employment_path: Optional[str] = None):
        """
        Initialize geographic analyzer

        Args:
            reit_properties_path: Path to REIT property database
            bls_ppi_path: Path to BLS PPI data (optional)
            bls_employment_path: Path to BLS employment data (optional)
        """
        self.reit_properties_path = Path(reit_properties_path)
        self.bls_ppi_path = Path(bls_ppi_path) if bls_ppi_path else None
        self.bls_employment_path = Path(bls_employment_path) if bls_employment_path else None

        self.reit_properties: Optional[pl.DataFrame] = None
        self.bls_ppi_data: Optional[pd.DataFrame] = None
        self.bls_employment_data: Optional[pd.DataFrame] = None

        self._load_data()

    def _load_data(self) -> None:
        """Load REIT property and BLS data"""
        if self.reit_properties_path.exists():
            self.reit_properties = pl.read_parquet(self.reit_properties_path)
            logger.info(f"Loaded {len(self.reit_properties)} REIT properties")

        if self.bls_ppi_path and self.bls_ppi_path.exists():
            self.bls_ppi_data = pd.read_parquet(self.bls_ppi_path)
            logger.info("Loaded BLS PPI data")

        if self.bls_employment_path and self.bls_employment_path.exists():
            self.bls_employment_data = pd.read_parquet(self.bls_employment_path)
            logger.info("Loaded BLS employment data")

    def get_state_concentration(self, reit_ticker: str) -> pd.DataFrame:
        """
        Calculate state-level concentration for a REIT

        Args:
            reit_ticker: REIT ticker symbol

        Returns:
            DataFrame with state-level metrics
        """
        if self.reit_properties is None:
            raise ValueError("REIT properties data not loaded")

        reit_data = self.reit_properties.filter(
            pl.col('reit_owner') == reit_ticker.upper()
        )

        if len(reit_data) == 0:
            return pd.DataFrame()

        # Calculate state-level metrics
        state_stats = (
            reit_data
            .groupby('state')
            .agg([
                pl.count().alias('facilities'),
                pl.sum('number_of_certified_beds').alias('total_beds'),
                pl.mean('occupancy_rate').alias('avg_occupancy'),
                pl.mean('overall_rating').alias('avg_quality_rating'),
            ])
            .sort('total_beds', descending=True)
        )

        # Add concentration percentages
        total_facilities = reit_data.select(pl.count()).item()
        total_beds = reit_data.select(pl.sum('number_of_certified_beds')).item()

        state_df = state_stats.to_pandas()
        state_df['facility_concentration_%'] = (state_df['facilities'] / total_facilities * 100)
        state_df['bed_concentration_%'] = (state_df['total_beds'] / total_beds * 100)

        # Add region mapping
        state_df['region'] = state_df['state'].map(STATE_TO_REGION)

        return state_df

    def get_regional_distribution(self, reit_ticker: str) -> pd.DataFrame:
        """
        Calculate regional (Census region) distribution

        Args:
            reit_ticker: REIT ticker symbol

        Returns:
            DataFrame with regional metrics
        """
        state_data = self.get_state_concentration(reit_ticker)

        if state_data.empty:
            return pd.DataFrame()

        # Aggregate to regional level
        regional = state_data.groupby('region').agg({
            'facilities': 'sum',
            'total_beds': 'sum',
            'avg_occupancy': 'mean',
            'avg_quality_rating': 'mean',
        }).reset_index()

        total_facilities = state_data['facilities'].sum()
        total_beds = state_data['total_beds'].sum()

        regional['facility_concentration_%'] = (regional['facilities'] / total_facilities * 100)
        regional['bed_concentration_%'] = (regional['total_beds'] / total_beds * 100)

        return regional.sort_values('total_beds', ascending=False)

    def compare_geographic_footprints(self, reit_tickers: List[str]) -> pd.DataFrame:
        """
        Compare geographic footprints across REITs

        Args:
            reit_tickers: List of REIT ticker symbols

        Returns:
            DataFrame comparing regional exposure
        """
        regional_data = []

        for ticker in reit_tickers:
            regional = self.get_regional_distribution(ticker)
            regional['reit'] = ticker
            regional_data.append(regional)

        if not regional_data:
            return pd.DataFrame()

        combined = pd.concat(regional_data, ignore_index=True)

        # Pivot for easier comparison
        pivot = combined.pivot_table(
            index='region',
            columns='reit',
            values='bed_concentration_%',
            fill_value=0
        ).reset_index()

        return pivot

    def calculate_market_attractiveness(self, state: str) -> float:
        """
        Calculate market attractiveness score for a state

        Factors:
        1. Medicare reimbursement rate (higher is better)
        2. Average occupancy rate (higher is better)
        3. Average quality rating (higher is better)
        4. PPI growth (higher is better for revenue)
        5. Wage growth (lower is better for margins)

        Args:
            state: State abbreviation

        Returns:
            Market attractiveness score (0-100)
        """
        if self.reit_properties is None:
            return 0.0

        # Filter facilities in state
        state_facilities = self.reit_properties.filter(
            pl.col('state') == state
        )

        if len(state_facilities) == 0:
            return 0.0

        # Calculate base metrics
        avg_occupancy = state_facilities['occupancy_rate'].mean()
        avg_quality = state_facilities['overall_rating'].mean()

        # Medicare reimbursement factor (normalized to 0-1)
        reimbursement_factor = MEDICARE_REIMBURSEMENT_INDEX.get(state, 1.0) / 1.67  # Max is AK at 1.67

        # Combine factors (weighted average)
        weights = {
            'occupancy': 0.3,
            'quality': 0.2,
            'reimbursement': 0.5,  # Most important for REIT revenue
        }

        # Normalize occupancy to 0-1 (assuming 100% is max)
        occupancy_score = (avg_occupancy or 0) / 100

        # Normalize quality to 0-1 (5-star max)
        quality_score = (avg_quality or 0) / 5

        # Calculate weighted score
        score = (
            weights['occupancy'] * occupancy_score +
            weights['quality'] * quality_score +
            weights['reimbursement'] * reimbursement_factor
        ) * 100

        return float(score)

    def get_market_opportunities(self, reit_ticker: str, min_score: float = 60.0) -> pd.DataFrame:
        """
        Identify high-opportunity markets where REIT has low presence

        Args:
            reit_ticker: REIT ticker symbol
            min_score: Minimum attractiveness score

        Returns:
            DataFrame of high-opportunity states
        """
        # Get current state concentration
        current_states = self.get_state_concentration(reit_ticker)

        if current_states.empty:
            current_presence = set()
        else:
            current_presence = set(current_states['state'].tolist())

        # Get all states with facilities
        if self.reit_properties is None:
            return pd.DataFrame()

        all_states = self.reit_properties['state'].unique().to_list()

        # Calculate attractiveness for states with low/no presence
        opportunities = []

        for state in all_states:
            current_beds = 0
            current_facilities = 0

            if state in current_presence:
                state_row = current_states[current_states['state'] == state].iloc[0]
                current_beds = state_row['total_beds']
                current_facilities = state_row['facilities']

            score = self.calculate_market_attractiveness(state)

            if score >= min_score:
                opportunities.append({
                    'state': state,
                    'attractiveness_score': score,
                    'current_facilities': current_facilities,
                    'current_beds': current_beds,
                    'region': STATE_TO_REGION.get(state, 'Unknown'),
                    'medicare_reimbursement_index': MEDICARE_REIMBURSEMENT_INDEX.get(state, 1.0)
                })

        if not opportunities:
            return pd.DataFrame()

        opp_df = pd.DataFrame(opportunities)
        return opp_df.sort_values('attractiveness_score', ascending=False)

    def analyze_geographic_risk(self, reit_ticker: str) -> Dict[str, float]:
        """
        Analyze geographic concentration risk

        Args:
            reit_ticker: REIT ticker symbol

        Returns:
            Dictionary of risk metrics
        """
        state_data = self.get_state_concentration(reit_ticker)

        if state_data.empty:
            return {
                'herfindahl_index': 0.0,
                'top_state_concentration': 0.0,
                'top_3_states_concentration': 0.0,
                'number_of_states': 0,
                'geographic_diversification_score': 0.0
            }

        # Calculate Herfindahl-Hirschman Index for concentration
        # Sum of squared market shares (higher = more concentrated)
        hhi = (state_data['bed_concentration_%'] ** 2).sum()

        # Top state concentration
        top_state = state_data.iloc[0]['bed_concentration_%']

        # Top 3 states concentration
        top_3 = state_data.head(3)['bed_concentration_%'].sum()

        # Number of states
        num_states = len(state_data)

        # Diversification score (inverse of concentration)
        # 100 = perfectly diversified, 0 = all in one state
        diversification_score = 100 * (1 - (hhi / 10000))

        return {
            'herfindahl_index': float(hhi),
            'top_state_concentration': float(top_state),
            'top_3_states_concentration': float(top_3),
            'number_of_states': int(num_states),
            'geographic_diversification_score': float(diversification_score)
        }

    def export_geographic_analysis(self, reit_tickers: List[str],
                                   output_dir: str = "reports/geographic_analysis") -> None:
        """
        Export comprehensive geographic analysis for all REITs

        Args:
            reit_tickers: List of REIT ticker symbols
            output_dir: Output directory for reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for ticker in reit_tickers:
            # State concentration
            state_data = self.get_state_concentration(ticker)
            state_data.to_csv(output_path / f"{ticker}_state_concentration.csv", index=False)

            # Regional distribution
            regional_data = self.get_regional_distribution(ticker)
            regional_data.to_csv(output_path / f"{ticker}_regional_distribution.csv", index=False)

            # Market opportunities
            opportunities = self.get_market_opportunities(ticker)
            opportunities.to_csv(output_path / f"{ticker}_market_opportunities.csv", index=False)

            # Risk metrics
            risk_metrics = self.analyze_geographic_risk(ticker)
            risk_df = pd.DataFrame([risk_metrics])
            risk_df.to_csv(output_path / f"{ticker}_geographic_risk.csv", index=False)

            logger.info(f"Exported geographic analysis for {ticker}")

        # Comparative analysis
        comparison = self.compare_geographic_footprints(reit_tickers)
        comparison.to_csv(output_path / "reit_geographic_comparison.csv", index=False)

        logger.info(f"All geographic analyses exported to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyzer = GeographicAnalyzer()

    # Example: Analyze OHI
    print("\n=== OHI State Concentration ===")
    ohi_states = analyzer.get_state_concentration('OHI')
    print(ohi_states.head(10))

    print("\n=== OHI Geographic Risk ===")
    ohi_risk = analyzer.analyze_geographic_risk('OHI')
    print(ohi_risk)

    # Export all analyses
    analyzer.export_geographic_analysis(['OHI', 'CTRE', 'SBRA'])
