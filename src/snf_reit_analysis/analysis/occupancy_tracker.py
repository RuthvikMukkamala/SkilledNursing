"""
Occupancy Rate Tracker - Monitor REIT Portfolio Occupancy

This module tracks and analyzes occupancy rates across REIT portfolios:
1. Portfolio-level occupancy trends
2. Operator-level occupancy performance
3. Regional occupancy patterns
4. Occupancy impact on revenue

Key Insight: Occupancy directly impacts rental revenue for REITs.
Tracking occupancy trends helps predict revenue stability and growth.
"""

import polars as pl
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OccupancyMetrics:
    """Occupancy metrics for a portfolio"""
    reit_ticker: str
    portfolio_occupancy: float
    weighted_occupancy: float  # Weighted by beds
    occupancy_by_state: Dict[str, float]
    occupancy_by_operator: Dict[str, float]
    occupancy_distribution: Dict[str, int]  # Bin -> count
    low_occupancy_facilities: int  # <75%
    high_occupancy_facilities: int  # ≥85%


class OccupancyTracker:
    """
    Track and analyze occupancy rates for REIT portfolios
    """

    def __init__(self, reit_properties_path: str = "data/processed/reit_properties.parquet"):
        """
        Initialize occupancy tracker

        Args:
            reit_properties_path: Path to REIT property database
        """
        self.reit_properties_path = Path(reit_properties_path)
        self.reit_properties: Optional[pl.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """Load REIT property data"""
        if self.reit_properties_path.exists():
            self.reit_properties = pl.read_parquet(self.reit_properties_path)
            logger.info(f"Loaded {len(self.reit_properties)} REIT properties")
        else:
            logger.warning(f"REIT properties file not found: {self.reit_properties_path}")

    def get_portfolio_occupancy(self, reit_ticker: str) -> OccupancyMetrics:
        """
        Calculate comprehensive occupancy metrics for a REIT

        Args:
            reit_ticker: REIT ticker symbol

        Returns:
            OccupancyMetrics dataclass
        """
        if self.reit_properties is None:
            raise ValueError("REIT properties data not loaded")

        reit_data = self.reit_properties.filter(
            pl.col('reit_owner') == reit_ticker.upper()
        )

        if len(reit_data) == 0:
            return OccupancyMetrics(
                reit_ticker=reit_ticker,
                portfolio_occupancy=0.0,
                weighted_occupancy=0.0,
                occupancy_by_state={},
                occupancy_by_operator={},
                occupancy_distribution={},
                low_occupancy_facilities=0,
                high_occupancy_facilities=0
            )

        # Simple average occupancy
        avg_occupancy = reit_data['occupancy_rate'].mean()

        # Weighted average by beds
        weighted_occupancy = (
            reit_data['occupancy_rate'] * reit_data['number_of_certified_beds']
        ).sum() / reit_data['number_of_certified_beds'].sum()

        # Occupancy by state
        state_occ = (
            reit_data
            .groupby('state')
            .agg([
                (pl.col('occupancy_rate') * pl.col('number_of_certified_beds')).sum().alias('weighted_occ'),
                pl.sum('number_of_certified_beds').alias('total_beds')
            ])
        )
        state_occ_dict = {}
        for row in state_occ.iter_rows(named=True):
            state_occ_dict[row['state']] = (
                row['weighted_occ'] / row['total_beds'] if row['total_beds'] > 0 else 0
            )

        # Occupancy by operator
        operator_occ = (
            reit_data
            .groupby('chain_name')
            .agg([
                (pl.col('occupancy_rate') * pl.col('number_of_certified_beds')).sum().alias('weighted_occ'),
                pl.sum('number_of_certified_beds').alias('total_beds')
            ])
        )
        operator_occ_dict = {}
        for row in operator_occ.iter_rows(named=True):
            operator_occ_dict[row['chain_name']] = (
                row['weighted_occ'] / row['total_beds'] if row['total_beds'] > 0 else 0
            )

        # Occupancy distribution (bins)
        bins = {'<60%': 0, '60-70%': 0, '70-80%': 0, '80-90%': 0, '90-100%': 0}
        for occ_rate in reit_data['occupancy_rate'].to_list():
            if occ_rate is None:
                continue
            elif occ_rate < 60:
                bins['<60%'] += 1
            elif occ_rate < 70:
                bins['60-70%'] += 1
            elif occ_rate < 80:
                bins['70-80%'] += 1
            elif occ_rate < 90:
                bins['80-90%'] += 1
            else:
                bins['90-100%'] += 1

        # Low and high occupancy counts
        low_occ = len(reit_data.filter(pl.col('occupancy_rate') < 75))
        high_occ = len(reit_data.filter(pl.col('occupancy_rate') >= 85))

        return OccupancyMetrics(
            reit_ticker=reit_ticker,
            portfolio_occupancy=float(avg_occupancy) if avg_occupancy else 0.0,
            weighted_occupancy=float(weighted_occupancy) if weighted_occupancy else 0.0,
            occupancy_by_state=state_occ_dict,
            occupancy_by_operator=operator_occ_dict,
            occupancy_distribution=bins,
            low_occupancy_facilities=low_occ,
            high_occupancy_facilities=high_occ
        )

    def compare_occupancy_across_reits(self, reit_tickers: List[str]) -> pd.DataFrame:
        """
        Compare occupancy metrics across multiple REITs

        Args:
            reit_tickers: List of REIT ticker symbols

        Returns:
            DataFrame with comparative occupancy metrics
        """
        comparison_data = []

        for ticker in reit_tickers:
            metrics = self.get_portfolio_occupancy(ticker)

            if self.reit_properties is None:
                continue

            reit_data = self.reit_properties.filter(
                pl.col('reit_owner') == ticker.upper()
            )

            total_facilities = len(reit_data)

            comparison_data.append({
                'REIT': ticker,
                'Portfolio Occupancy (%)': metrics.portfolio_occupancy,
                'Weighted Occupancy (%)': metrics.weighted_occupancy,
                'Total Facilities': total_facilities,
                'Low Occupancy (<75%)': metrics.low_occupancy_facilities,
                'Low Occupancy %': (metrics.low_occupancy_facilities / total_facilities * 100)
                                   if total_facilities > 0 else 0,
                'High Occupancy (≥85%)': metrics.high_occupancy_facilities,
                'High Occupancy %': (metrics.high_occupancy_facilities / total_facilities * 100)
                                    if total_facilities > 0 else 0,
                'Occupancy Range': f"{reit_data['occupancy_rate'].min():.1f}% - {reit_data['occupancy_rate'].max():.1f}%"
            })

        return pd.DataFrame(comparison_data)

    def get_occupancy_by_quality_rating(self, reit_ticker: str) -> pd.DataFrame:
        """
        Analyze relationship between occupancy and quality rating

        Args:
            reit_ticker: REIT ticker symbol

        Returns:
            DataFrame showing occupancy by quality tier
        """
        if self.reit_properties is None:
            return pd.DataFrame()

        reit_data = self.reit_properties.filter(
            pl.col('reit_owner') == reit_ticker.upper()
        )

        if len(reit_data) == 0:
            return pd.DataFrame()

        # Group by quality rating
        quality_occ = (
            reit_data
            .groupby('overall_rating')
            .agg([
                pl.count().alias('facilities'),
                pl.mean('occupancy_rate').alias('avg_occupancy'),
                (pl.col('occupancy_rate') * pl.col('number_of_certified_beds')).sum().alias('weighted_occ'),
                pl.sum('number_of_certified_beds').alias('total_beds')
            ])
            .sort('overall_rating')
        )

        df = quality_occ.to_pandas()
        df['weighted_avg_occupancy'] = df['weighted_occ'] / df['total_beds']

        return df[['overall_rating', 'facilities', 'avg_occupancy', 'weighted_avg_occupancy', 'total_beds']]

    def identify_underperforming_facilities(self, reit_ticker: str,
                                           occupancy_threshold: float = 75.0) -> pd.DataFrame:
        """
        Identify facilities with occupancy below threshold

        Args:
            reit_ticker: REIT ticker symbol
            occupancy_threshold: Occupancy threshold (default 75%)

        Returns:
            DataFrame of underperforming facilities
        """
        if self.reit_properties is None:
            return pd.DataFrame()

        reit_data = self.reit_properties.filter(
            (pl.col('reit_owner') == reit_ticker.upper()) &
            (pl.col('occupancy_rate') < occupancy_threshold)
        )

        if len(reit_data) == 0:
            return pd.DataFrame()

        # Select key columns and sort by occupancy
        underperforming = (
            reit_data
            .select([
                'provider_name',
                'citytown',
                'state',
                'chain_name',
                'number_of_certified_beds',
                'occupancy_rate',
                'overall_rating',
            ])
            .sort('occupancy_rate')
        )

        df = underperforming.to_pandas()

        # Add potential revenue impact (beds not occupied)
        df['empty_beds'] = df['number_of_certified_beds'] * (1 - df['occupancy_rate'] / 100)

        return df

    def calculate_revenue_impact(self, reit_ticker: str,
                                 avg_daily_rate: float = 250.0) -> Dict[str, float]:
        """
        Estimate revenue impact of occupancy gaps

        Args:
            reit_ticker: REIT ticker symbol
            avg_daily_rate: Average daily rate per bed (default $250)

        Returns:
            Dictionary with revenue impact metrics
        """
        if self.reit_properties is None:
            return {}

        reit_data = self.reit_properties.filter(
            pl.col('reit_owner') == reit_ticker.upper()
        )

        if len(reit_data) == 0:
            return {}

        # Calculate total beds
        total_beds = reit_data['number_of_certified_beds'].sum()

        # Calculate occupied beds
        avg_occupancy = (
            reit_data['occupancy_rate'] * reit_data['number_of_certified_beds']
        ).sum() / total_beds

        occupied_beds = total_beds * (avg_occupancy / 100)
        empty_beds = total_beds - occupied_beds

        # Annual revenue potential
        annual_revenue_actual = occupied_beds * avg_daily_rate * 365
        annual_revenue_potential = total_beds * avg_daily_rate * 365
        revenue_gap = annual_revenue_potential - annual_revenue_actual

        # If occupancy improved to 85%
        revenue_at_85_pct = total_beds * 0.85 * avg_daily_rate * 365
        upside_to_85_pct = revenue_at_85_pct - annual_revenue_actual

        return {
            'total_beds': int(total_beds),
            'avg_occupancy_%': float(avg_occupancy),
            'occupied_beds': float(occupied_beds),
            'empty_beds': float(empty_beds),
            'annual_revenue_actual_$M': annual_revenue_actual / 1e6,
            'annual_revenue_potential_$M': annual_revenue_potential / 1e6,
            'revenue_gap_$M': revenue_gap / 1e6,
            'revenue_at_85%_occupancy_$M': revenue_at_85_pct / 1e6,
            'upside_to_85%_$M': upside_to_85_pct / 1e6
        }

    def get_regional_occupancy_comparison(self, reit_tickers: List[str]) -> pd.DataFrame:
        """
        Compare occupancy rates across REITs by region

        Args:
            reit_tickers: List of REIT ticker symbols

        Returns:
            DataFrame with regional occupancy comparison
        """
        regional_data = []

        for ticker in reit_tickers:
            metrics = self.get_portfolio_occupancy(ticker)

            for state, occupancy in metrics.occupancy_by_state.items():
                regional_data.append({
                    'REIT': ticker,
                    'State': state,
                    'Occupancy_%': occupancy
                })

        if not regional_data:
            return pd.DataFrame()

        df = pd.DataFrame(regional_data)

        # Pivot for easier comparison
        pivot = df.pivot_table(
            index='State',
            columns='REIT',
            values='Occupancy_%',
            fill_value=0
        ).reset_index()

        return pivot

    def export_occupancy_analysis(self, reit_tickers: List[str],
                                  output_dir: str = "reports/occupancy_analysis") -> None:
        """
        Export comprehensive occupancy analysis

        Args:
            reit_tickers: List of REIT ticker symbols
            output_dir: Output directory for reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # REIT comparison
        comparison = self.compare_occupancy_across_reits(reit_tickers)
        comparison.to_csv(output_path / "reit_occupancy_comparison.csv", index=False)

        for ticker in reit_tickers:
            # Occupancy metrics
            metrics = self.get_portfolio_occupancy(ticker)
            metrics_df = pd.DataFrame([{
                'REIT': ticker,
                'Portfolio_Occupancy_%': metrics.portfolio_occupancy,
                'Weighted_Occupancy_%': metrics.weighted_occupancy,
                'Low_Occupancy_Facilities': metrics.low_occupancy_facilities,
                'High_Occupancy_Facilities': metrics.high_occupancy_facilities
            }])
            metrics_df.to_csv(output_path / f"{ticker}_occupancy_metrics.csv", index=False)

            # Quality-occupancy relationship
            quality_occ = self.get_occupancy_by_quality_rating(ticker)
            if not quality_occ.empty:
                quality_occ.to_csv(output_path / f"{ticker}_occupancy_by_quality.csv", index=False)

            # Underperforming facilities
            underperforming = self.identify_underperforming_facilities(ticker)
            if not underperforming.empty:
                underperforming.to_csv(output_path / f"{ticker}_underperforming_facilities.csv", index=False)

            # Revenue impact
            revenue_impact = self.calculate_revenue_impact(ticker)
            if revenue_impact:
                rev_df = pd.DataFrame([revenue_impact])
                rev_df.to_csv(output_path / f"{ticker}_revenue_impact.csv", index=False)

            logger.info(f"Exported occupancy analysis for {ticker}")

        # Regional comparison
        regional = self.get_regional_occupancy_comparison(reit_tickers)
        if not regional.empty:
            regional.to_csv(output_path / "regional_occupancy_comparison.csv", index=False)

        logger.info(f"All occupancy analyses exported to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tracker = OccupancyTracker()

    # Analyze OHI
    print("\n=== OHI Occupancy Metrics ===")
    ohi_metrics = tracker.get_portfolio_occupancy('OHI')
    print(f"Portfolio Occupancy: {ohi_metrics.portfolio_occupancy:.1f}%")
    print(f"Weighted Occupancy: {ohi_metrics.weighted_occupancy:.1f}%")
    print(f"Low Occupancy Facilities: {ohi_metrics.low_occupancy_facilities}")
    print(f"High Occupancy Facilities: {ohi_metrics.high_occupancy_facilities}")

    # Revenue impact
    print("\n=== OHI Revenue Impact ===")
    revenue_impact = tracker.calculate_revenue_impact('OHI')
    for key, value in revenue_impact.items():
        print(f"{key}: {value}")

    # Compare all REITs
    print("\n=== REIT Occupancy Comparison ===")
    comparison = tracker.compare_occupancy_across_reits(['OHI', 'CTRE', 'SBRA'])
    print(comparison)

    # Export all analyses
    tracker.export_occupancy_analysis(['OHI', 'CTRE', 'SBRA'])
