# Omega Healthcare Investors (OHI) 2024 Data

## Files Downloaded

### 1. SEC 10-K Filing (Fiscal Year 2024)
- **Source**: SEC EDGAR
- **Accession Number**: 0000888491-25-000006
- **Filed**: February 13, 2025
- **Period**: Fiscal year ended December 31, 2024
- **Files**:
  - `ohi_10k_2024.pdf` (13 MB) - Full 10-K filing in PDF format
  - `ohi_10k_2024.htm` (6.2 MB) - Full 10-K filing in HTML/XBRL format
  - `ohi_10k_2024.txt` (1.5 MB) - Text extraction from PDF

### 2. Facilities List
- **Source**: Omega Healthcare Investors website (https://www.omegahealthcare.com/portfolio)
- **File**: `ohi_facilities_list_q2_2025.pdf` (439 KB)
- **Date**: Q2 2025 (as of June 30, 2025)
- **Description**: Detailed facility-by-facility listing with addresses

### 3. Extracted Schedule III Data
- **File**: `ohi_schedule_iii_2024.csv` (98 KB)
- **Records**: 1,032 total facilities
  - 743 US properties
  - 289 UK properties (England, Scotland, Wales)
- **Columns**:
  - `state`: US state or UK country
  - `facility_name`: Name of the facility
  - `facility_type`: Type of healthcare facility
  - `street_address`: Street address
  - `city`: City
  - `county`: County (US) or equivalent (UK)
  - `zip`: ZIP/postal code

## Schedule III Details

### What is Schedule III?
Schedule III is a required SEC filing schedule titled "Real Estate and Accumulated Depreciation" that provides information about a REIT's real estate investments. For Omega Healthcare Investors, the 10-K filing contains:

1. **Aggregated Schedule III** (in 10-K, pages F-71 to F-72):
   - Summary by state/country
   - Initial cost, improvements, carrying value
   - Accumulated depreciation
   - Construction dates and acquisition dates
   - Does NOT include property-by-property listings

2. **Detailed Facilities List** (from company website):
   - Property-by-property data with addresses
   - Facility names, types, and locations
   - Available as a supplemental document

### Data Quality Notes

**What We Have**:
- Complete facility names and addresses for all 1,032 properties
- Facility types (SNF, ALF, ILF, SF, MOB, etc.)
- Geographic information (state, city, county, zip)

**What We Don't Have** (not in the facilities list):
- **Operator/Tenant names**: This information is not included in the public facilities list
- **Lease terms**: Specific lease details are not disclosed property-by-property
- **Revenue by property**: Financial data is aggregated by state in Schedule III

The operator/tenant information may be available in:
- Quarterly supplements (operator summaries)
- SEC filings (major operator disclosures)
- Direct requests to investor relations

### Facility Type Breakdown

```
Skilled Nursing Facility         608
Assisted Living Facility         373
Independent Living Facility       21
Traumatic Brain Injury            12
Behaviorial Health - SUD           6
Long Term Acute Care Hospital      3
Specialty Hospital                 2
Behaviorial Health - Psych         2
Medical Office Building            1
Inpatient Rehab Facility           1
Other (UK-specific types)          3
```

### Geographic Distribution (Top 10 States)

```
Texas                             98
Indiana                           68
California                        52
Florida                           50
North Carolina                    45
Ohio                              39
Pennsylvania                      38
Michigan                          35
Tennessee                         34
England (UK)                     235
```

## Usage

The CSV file can be loaded in Python:

```python
import pandas as pd

# Load the data
df = pd.read_csv('data/raw/ohi_schedule_iii_2024.csv')

# Filter to US properties only
us_properties = df[~df['state'].isin(['England', 'Scotland', 'Wales'])]

# Group by state
by_state = df.groupby('state').size()
```

## Data Limitations

1. **Operator Information**: The facilities list does not include operator/tenant names for individual properties. This data may require:
   - Reviewing quarterly earnings supplements
   - Analyzing disclosure in Note sections of the 10-K
   - Contacting Omega investor relations

2. **Date Discrepancy**: The facilities list is dated Q2 2025 (June 30, 2025), while the 10-K covers the fiscal year 2024 (ending December 31, 2024). There may be slight differences in the portfolio between these dates.

3. **Financial Data**: Property-level financial data (rents, revenues, valuations) is aggregated by state in the SEC Schedule III and not available at the individual property level.

## Sources & References

- **SEC Filing**: https://www.sec.gov/Archives/edgar/data/888491/000088849125000006/ohi-20241231x10k.htm
- **Omega Website**: https://www.omegahealthcare.com/
- **Facilities List**: https://www.omegahealthcare.com/portfolio
- **CIK**: 0000888491

## Extraction Scripts

- `scripts/parse_facilities_list_v2.py`: Python script to parse the facilities PDF into CSV format
