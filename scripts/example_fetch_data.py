"""
Example script to fetch data from all three sources.

This script demonstrates how to use the data loaders programmatically.
"""

from snf_reit_analysis.data.loaders import BLSDataLoader, CMSDataLoader, SECDataLoader
from snf_reit_analysis.utils.logging import get_logger, setup_logging

setup_logging(log_level="INFO")
logger = get_logger(__name__)


def main():
    logger.info("Starting data fetch example...")
    logger.info("Fetching CMS Provider Information...")
    cms = CMSDataLoader()

    # Get high-rated facilities in California
    ca_providers = cms.get_provider_info(state="CA", min_rating=4, use_polars=False)
    logger.info(f"Fetched {len(ca_providers):,} high-rated CA facilities")
    logger.info(f"Sample: {ca_providers['Provider_Name'].head(3).tolist()}")

    logger.info("Fetching BLS Economic Data...")
    bls = BLSDataLoader()

    # Get PPI data for last 2 years
    ppi_data = bls.get_nursing_facility_ppi(start_year="2023", end_year="2024")
    logger.info(f"Fetched {len(ppi_data):,} PPI records")

    latest_ppi = ppi_data[ppi_data["series_name"] == "PPI Overall"].iloc[0]
    logger.info(
        f"Latest PPI Overall: {latest_ppi['value']} ({latest_ppi['year']}-{latest_ppi['period']})"
    )

    logger.info("Fetching SEC REIT Financials...")
    sec = SECDataLoader()

    ohi_financials = sec.get_reit_financials("OHI", tags=["Assets", "Revenues"])
    logger.info(f"Fetched {len(ohi_financials)} metrics for OHI")

    if "Assets" in ohi_financials and not ohi_financials["Assets"].empty:
        latest_assets = ohi_financials["Assets"].iloc[0]
        logger.info(
            f"Latest OHI Total Assets: ${latest_assets['val']:,.0f} ({latest_assets['end']})"
        )

if __name__ == "__main__":
    main()
