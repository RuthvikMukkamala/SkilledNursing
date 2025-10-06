"""
Tenant Mix Analysis - Operator Quality and Concentration

This module analyzes the tenant/operator mix for each REIT, providing insights into:
1. Operator concentration and diversification
2. Operator quality metrics (from CMS data)
3. Revenue concentration risk
4. Tenant creditworthiness indicators

Key Insight: REITs generate revenue from operators (tenants) who run the facilities.
Understanding operator quality and concentration is critical for revenue stability.
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
class OperatorProfile:
    """Profile of a single operator"""
    operator_name: str
    facilities: int
    total_beds: int
    avg_occupancy: float
    avg_quality_rating: float
    states: List[str]
    revenue_share_pct: float
    quality_trend: Optional[str] = None  # 'improving', 'stable', 'declining'


class TenantMixAnalyzer:
    """
    Analyze tenant/operator mix for REIT portfolios
    """

    def __init__(self, reit_properties_path: str = "data/processed/reit_properties.parquet"):
        """
        Initialize tenant mix analyzer

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

    def get_operator_profiles(self, reit_ticker: str) -> List[OperatorProfile]:
        """
        Get detailed profiles for all operators of a REIT

        Args:
            reit_ticker: REIT ticker symbol

        Returns:
            List of OperatorProfile objects
        """
        if self.reit_properties is None:
            return []

        reit_data = self.reit_properties.filter(
            pl.col('reit_owner') == reit_ticker.upper()
        )

        if len(reit_data) == 0:
            return []

        # Calculate total beds for revenue share estimation
        total_reit_beds = reit_data.select(pl.sum('number_of_certified_beds')).item()

        # Group by operator (chain_name)
        operator_stats = (
            reit_data
            .groupby('chain_name')
            .agg([
                pl.count().alias('facilities'),
                pl.sum('number_of_certified_beds').alias('total_beds'),
                pl.mean('occupancy_rate').alias('avg_occupancy'),
                pl.mean('overall_rating').alias('avg_quality_rating'),
                pl.col('state').unique().alias('states_list'),
            ])
            .sort('total_beds', descending=True)
        )

        profiles = []

        for row in operator_stats.iter_rows(named=True):
            # Estimate revenue share based on beds (assumes similar rates)
            revenue_share = (row['total_beds'] / total_reit_beds * 100) if total_reit_beds > 0 else 0

            profile = OperatorProfile(
                operator_name=row['chain_name'],
                facilities=row['facilities'],
                total_beds=row['total_beds'],
                avg_occupancy=float(row['avg_occupancy']) if row['avg_occupancy'] else 0.0,
                avg_quality_rating=float(row['avg_quality_rating']) if row['avg_quality_rating'] else 0.0,
                states=row['states_list'],
                revenue_share_pct=float(revenue_share)
            )
            profiles.append(profile)

        return profiles

    def calculate_tenant_concentration(self, reit_ticker: str) -> Dict[str, float]:
        """
        Calculate tenant concentration metrics

        Args:
            reit_ticker: REIT ticker symbol

        Returns:
            Dictionary with concentration metrics
        """
        profiles = self.get_operator_profiles(reit_ticker)

        if not profiles:
            return {
                'herfindahl_index': 0.0,
                'top_tenant_revenue_share': 0.0,
                'top_3_tenants_revenue_share': 0.0,
                'top_5_tenants_revenue_share': 0.0,
                'number_of_operators': 0,
                'tenant_diversification_score': 0.0
            }

        # Sort by revenue share
        profiles_sorted = sorted(profiles, key=lambda x: x.revenue_share_pct, reverse=True)

        # Herfindahl-Hirschman Index (sum of squared market shares)
        hhi = sum(p.revenue_share_pct ** 2 for p in profiles)

        # Top tenant concentrations
        top_1 = profiles_sorted[0].revenue_share_pct if len(profiles_sorted) >= 1 else 0
        top_3 = sum(p.revenue_share_pct for p in profiles_sorted[:3])
        top_5 = sum(p.revenue_share_pct for p in profiles_sorted[:5])

        # Diversification score (100 = perfectly diversified, 0 = one tenant)
        diversification_score = 100 * (1 - (hhi / 10000))

        return {
            'herfindahl_index': float(hhi),
            'top_tenant_revenue_share': float(top_1),
            'top_3_tenants_revenue_share': float(top_3),
            'top_5_tenants_revenue_share': float(top_5),
            'number_of_operators': len(profiles),
            'tenant_diversification_score': float(diversification_score)
        }

    def get_operator_quality_distribution(self, reit_ticker: str) -> pd.DataFrame:
        """
        Analyze quality distribution across operators

        Args:
            reit_ticker: REIT ticker symbol

        Returns:
            DataFrame with operator quality metrics
        """
        profiles = self.get_operator_profiles(reit_ticker)

        if not profiles:
            return pd.DataFrame()

        data = []
        for profile in profiles:
            data.append({
                'operator': profile.operator_name,
                'facilities': profile.facilities,
                'total_beds': profile.total_beds,
                'revenue_share_%': profile.revenue_share_pct,
                'avg_occupancy_%': profile.avg_occupancy,
                'avg_quality_rating': profile.avg_quality_rating,
                'states': len(profile.states),
                'quality_category': self._categorize_quality(profile.avg_quality_rating),
                'occupancy_category': self._categorize_occupancy(profile.avg_occupancy)
            })

        df = pd.DataFrame(data)
        return df.sort_values('revenue_share_%', ascending=False)

    def _categorize_quality(self, rating: float) -> str:
        """Categorize quality rating"""
        if rating >= 4.0:
            return 'High Quality (4-5 stars)'
        elif rating >= 3.0:
            return 'Medium Quality (3 stars)'
        else:
            return 'Low Quality (<3 stars)'

    def _categorize_occupancy(self, occupancy: float) -> str:
        """Categorize occupancy rate"""
        if occupancy >= 85.0:
            return 'High Occupancy (≥85%)'
        elif occupancy >= 75.0:
            return 'Medium Occupancy (75-85%)'
        else:
            return 'Low Occupancy (<75%)'

    def identify_credit_risks(self, reit_ticker: str, risk_threshold: float = 75.0) -> pd.DataFrame:
        """
        Identify operators with potential credit risk

        Risk factors:
        1. Low occupancy (<75%)
        2. Low quality rating (<3 stars)
        3. Declining quality trend (if available)

        Args:
            reit_ticker: REIT ticker symbol
            risk_threshold: Occupancy threshold for risk (default 75%)

        Returns:
            DataFrame of at-risk operators
        """
        operator_df = self.get_operator_quality_distribution(reit_ticker)

        if operator_df.empty:
            return pd.DataFrame()

        # Identify risk factors
        at_risk = operator_df[
            (operator_df['avg_occupancy_%'] < risk_threshold) |
            (operator_df['avg_quality_rating'] < 3.0)
        ].copy()

        if not at_risk.empty:
            # Add risk score (weighted by revenue share)
            at_risk['risk_score'] = (
                (100 - at_risk['avg_occupancy_%']) * 0.6 +
                (5 - at_risk['avg_quality_rating']) * 10 * 0.4
            ) * (at_risk['revenue_share_%'] / 100)

            at_risk = at_risk.sort_values('risk_score', ascending=False)

        return at_risk

    def compare_tenant_mix(self, reit_tickers: List[str]) -> pd.DataFrame:
        """
        Compare tenant mix quality across REITs

        Args:
            reit_tickers: List of REIT ticker symbols

        Returns:
            DataFrame comparing tenant mix metrics
        """
        comparison_data = []

        for ticker in reit_tickers:
            concentration = self.calculate_tenant_concentration(ticker)
            operator_df = self.get_operator_quality_distribution(ticker)

            if not operator_df.empty:
                # Calculate portfolio-weighted averages
                weighted_occupancy = (
                    operator_df['avg_occupancy_%'] * operator_df['revenue_share_%']
                ).sum() / 100

                weighted_quality = (
                    operator_df['avg_quality_rating'] * operator_df['revenue_share_%']
                ).sum() / 100

                # Quality distribution
                high_quality_share = operator_df[
                    operator_df['quality_category'] == 'High Quality (4-5 stars)'
                ]['revenue_share_%'].sum()

                comparison_data.append({
                    'REIT': ticker,
                    'Number of Operators': concentration['number_of_operators'],
                    'Top Tenant Share (%)': concentration['top_tenant_revenue_share'],
                    'Top 3 Tenants Share (%)': concentration['top_3_tenants_revenue_share'],
                    'HHI (Concentration)': concentration['herfindahl_index'],
                    'Diversification Score': concentration['tenant_diversification_score'],
                    'Weighted Avg Occupancy (%)': weighted_occupancy,
                    'Weighted Avg Quality': weighted_quality,
                    'High Quality Revenue (%)': high_quality_share
                })
            else:
                comparison_data.append({
                    'REIT': ticker,
                    'Number of Operators': 0,
                    'Top Tenant Share (%)': 0,
                    'Top 3 Tenants Share (%)': 0,
                    'HHI (Concentration)': 0,
                    'Diversification Score': 0,
                    'Weighted Avg Occupancy (%)': 0,
                    'Weighted Avg Quality': 0,
                    'High Quality Revenue (%)': 0
                })

        return pd.DataFrame(comparison_data)

    def analyze_operator_overlap(self, reit_tickers: List[str]) -> pd.DataFrame:
        """
        Identify operators that work with multiple REITs

        Args:
            reit_tickers: List of REIT ticker symbols

        Returns:
            DataFrame showing shared operators
        """
        if self.reit_properties is None:
            return pd.DataFrame()

        # Get all operators for each REIT
        reit_operators = {}
        for ticker in reit_tickers:
            profiles = self.get_operator_profiles(ticker)
            reit_operators[ticker] = {p.operator_name for p in profiles}

        # Find overlaps
        all_operators = set()
        for operators in reit_operators.values():
            all_operators.update(operators)

        overlap_data = []
        for operator in all_operators:
            reits_with_operator = [
                ticker for ticker, operators in reit_operators.items()
                if operator in operators
            ]

            if len(reits_with_operator) > 1:  # Only show shared operators
                overlap_data.append({
                    'operator': operator,
                    'reits': ', '.join(reits_with_operator),
                    'reit_count': len(reits_with_operator)
                })

        if not overlap_data:
            return pd.DataFrame()

        df = pd.DataFrame(overlap_data)
        return df.sort_values('reit_count', ascending=False)

    def export_tenant_analysis(self, reit_tickers: List[str],
                               output_dir: str = "reports/tenant_mix") -> None:
        """
        Export comprehensive tenant mix analysis

        Args:
            reit_tickers: List of REIT ticker symbols
            output_dir: Output directory for reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for ticker in reit_tickers:
            # Operator quality distribution
            operator_df = self.get_operator_quality_distribution(ticker)
            operator_df.to_csv(output_path / f"{ticker}_operator_quality.csv", index=False)

            # Concentration metrics
            concentration = self.calculate_tenant_concentration(ticker)
            conc_df = pd.DataFrame([concentration])
            conc_df.to_csv(output_path / f"{ticker}_tenant_concentration.csv", index=False)

            # Credit risk analysis
            at_risk = self.identify_credit_risks(ticker)
            if not at_risk.empty:
                at_risk.to_csv(output_path / f"{ticker}_credit_risks.csv", index=False)

            logger.info(f"Exported tenant analysis for {ticker}")

        # Comparative analysis
        comparison = self.compare_tenant_mix(reit_tickers)
        comparison.to_csv(output_path / "reit_tenant_mix_comparison.csv", index=False)

        # Operator overlap
        overlap = self.analyze_operator_overlap(reit_tickers)
        if not overlap.empty:
            overlap.to_csv(output_path / "operator_overlap_analysis.csv", index=False)

        logger.info(f"All tenant analyses exported to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    analyzer = TenantMixAnalyzer()

    # Analyze OHI
    print("\n=== OHI Tenant Concentration ===")
    ohi_concentration = analyzer.calculate_tenant_concentration('OHI')
    for key, value in ohi_concentration.items():
        print(f"{key}: {value:.2f}")

    print("\n=== OHI Operator Quality ===")
    ohi_operators = analyzer.get_operator_quality_distribution('OHI')
    print(ohi_operators.head(10))

    # Compare all REITs
    print("\n=== REIT Tenant Mix Comparison ===")
    comparison = analyzer.compare_tenant_mix(['OHI', 'CTRE', 'SBRA'])
    print(comparison)

    # Export all analyses
    analyzer.export_tenant_analysis(['OHI', 'CTRE', 'SBRA'])
