# Processed Data - CSV Files

This directory contains processed data extracted from CMS, BLS, and SEC APIs, saved as CSV files.

## Available Datasets

### CMS (Centers for Medicare & Medicaid Services)
- **cms_provider_info_latest.csv** - Nursing home provider information with ratings, staffing, and quality metrics
  - 14,752 records
  - Columns: cms_certification_number_ccn, provider_name, state, overall_rating, health_inspection_rating, staffing_rating, qm_rating, number_of_certified_beds, etc.

### SEC EDGAR (3 REITs: OHI, CTRE, SBRA)

#### Omega Healthcare Investors (OHI)
- sec_ohi_assets_latest.csv - Total assets over time
- sec_ohi_liabilities_latest.csv - Total liabilities 
- sec_ohi_stockholdersequity_latest.csv - Stockholders equity
- sec_ohi_revenues_latest.csv - Total revenues
- sec_ohi_netincomeloss_latest.csv - Net income/loss
- sec_ohi_realestateinvestmentpropertynet_latest.csv - Real estate property (net)
- sec_ohi_longtermdebt_latest.csv - Long-term debt
- sec_ohi_operatingincomeloss_latest.csv - Operating income/loss

#### CareTrust REIT (CTRE)
- sec_ctre_assets_latest.csv
- sec_ctre_liabilities_latest.csv
- sec_ctre_stockholdersequity_latest.csv
- sec_ctre_revenues_latest.csv
- sec_ctre_netincomeloss_latest.csv
- sec_ctre_realestateinvestmentpropertynet_latest.csv
- sec_ctre_longtermdebt_latest.csv

#### Sabra Health Care REIT (SBRA)
- sec_sbra_assets_latest.csv
- sec_sbra_liabilities_latest.csv
- sec_sbra_stockholdersequity_latest.csv
- sec_sbra_revenues_latest.csv
- sec_sbra_netincomeloss_latest.csv
- sec_sbra_realestateinvestmentpropertynet_latest.csv
- sec_sbra_longtermdebt_latest.csv

### SEC Data Structure
All SEC CSV files contain:
- `start` - Period start date (for flow metrics like revenues)
- `end` - Period end date
- `val` - Metric value
- `accn` - SEC accession number
- `fy` - Fiscal year
- `fp` - Fiscal period (FY, Q1, Q2, Q3, Q4)
- `form` - Filing form type (10-K for annual, 10-Q for quarterly)
- `filed` - Filing date
- `frame` - CIK frame identifier

## Data Refresh

To refresh the data:
```bash
make data-cms  # Refresh CMS data
make data-sec  # Refresh SEC data
make data-bls  # Refresh BLS data (requires API key)
make data      # Refresh all data sources
```

## File Naming Convention
- `{dataset}_latest.csv` - Most recent version of the dataset
- `{dataset}_YYYYMMDD_HHMMSS.csv` - Timestamped versions for historical tracking
