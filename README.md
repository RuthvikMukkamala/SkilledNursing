# Healthcare REIT Analysis Platform

**Version 1.0.0**

A comprehensive data-driven analysis platform for evaluating skilled nursing facility (SNF) REITs using multi-source public data integration. Combines SEC financial statements, BLS economic indicators, CMS quality ratings, and REIT property disclosures to generate differentiated investment insights.

---

## Overview

This platform analyzes three major healthcare REITs:

| Ticker | Company | Total Assets | Properties | Focus |
|--------|---------|--------------|------------|-------|
| **OHI** | Omega Healthcare Investors | $10.55B | 872 | Market leader, scale advantages |
| **CTRE** | CareTrust REIT | $4.66B | 239 | High-growth, nimble operator |
| **SBRA** | Sabra Healthcare REIT | $5.33B | 418 | Diversified mid-cap |


### 1. Initial Setup

```bash
# Clone repository
git clone https://github.com/ruthvik/NursingREITs.git
cd NursingREITs

# Setup project directories and configuration
make setup
```

This creates:
- Data directories (`data/raw`, `data/interim`, `data/processed`, `data/external`)
- Model directories (`models/production`, `models/experiments`)
- Report directories (`reports/figures`, `reports/outputs`)
- `.env` configuration file template

### 2. Configure API Keys

Edit `.env` file with your credentials:

```bash
# Bureau of Labor Statistics API Key (free registration)
BLS_API_KEY=your_bls_key_here

# SEC EDGAR User-Agent (required, must include email)
SEC_USER_AGENT="Your-Company contact@example.com"
```

**Get API Keys:**
- **BLS API**: Register at https://data.bls.gov/registrationEngine/
- **SEC EDGAR**: Just use format `"CompanyName contact@email.com"`
- **CMS API**: No key required (public API)

### 3. Install Dependencies

```bash
# Install with development tools
make install

# OR for production only (no testing/linting tools)
make install-prod
```

### 4. Fetch Data

```bash
# Fetch all data sources (CMS + BLS + SEC)
make data

# OR fetch individually:
make data-cms    # CMS nursing home quality data (~15K facilities)
make data-bls    # BLS economic indicators (PPI, employment)
make data-sec    # SEC financial statements (OHI, CTRE, SBRA)
```

**Note**: Full data fetch takes ~5-10 minutes depending on connection.

### 5. Launch Dashboard

```bash
# Launch interactive dashboard
make dashboard

# OR with auto-reload for development
make dev
```

Dashboard opens at: `http://localhost:8501`

---

## Available Commands

### Setup & Installation

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make setup` | Initial project setup (directories, .env template) |
| `make install` | Install package with dev dependencies |
| `make install-prod` | Install production dependencies only |

### Data Pipeline

| Command | Description |
|---------|-------------|
| `make data` | **Fetch all data sources** (CMS + BLS + SEC) |
| `make data-cms` | Fetch CMS nursing home provider data only |
| `make data-bls` | Fetch BLS economic indicators only |
| `make data-sec` | Fetch SEC EDGAR financials only |
| `make pipeline` | Run full ETL pipeline (alternative entry point) |

### Dashboard

| Command | Description |
|---------|-------------|
| `make dashboard` | Launch Streamlit dashboard (production mode) |
| `make dev` | Launch dashboard with auto-reload on file changes |

### Code Quality

| Command | Description |
|---------|-------------|
| `make test` | Run pytest with coverage report |
| `make test-fast` | Run tests without coverage (faster) |
| `make lint` | Run linting checks (ruff + mypy) |
| `make format` | Auto-format code (black + ruff) |
| `make check` | Run all quality checks (format + lint + test) |

### Cleanup

| Command | Description |
|---------|-------------|
| `make clean` | Clean build artifacts and cache |
| `make clean-data` | **⚠️ Delete all data files** (prompts for confirmation) |

---

## Data Sources

### 1. CMS Provider Data Catalog
- **Source**: https://data.cms.gov/provider-data/
- **Data**: 15,000+ nursing home facilities
- **Metrics**: Quality ratings (1-5 stars), ownership, bed counts, deficiencies
- **Update Frequency**: Annual (Q1)
- **Authentication**: None required

### 2. Bureau of Labor Statistics (BLS)
- **Source**: https://api.bls.gov/publicAPI/v2
- **Data**: Producer Price Index (PPI), employment statistics
- **Metrics**: Nursing facility pricing, wage inflation, staffing levels
- **Update Frequency**: Monthly (2nd Friday)
- **Authentication**: Free API key required

### 3. SEC EDGAR
- **Source**: https://data.sec.gov/api/xbrl/companyfacts/
- **Data**: XBRL financial statements (10-Q, 10-K)
- **Metrics**: Revenue, assets, liabilities, equity, debt
- **Update Frequency**: Quarterly (45 days after quarter end)
- **Authentication**: User-Agent header required

### 4. REIT Investor Relations
- **Source**: Company investor relations websites
- **Data**: Official property schedules (Q2 2025)
- **Metrics**: Facility names, addresses, bed counts, property types
- **Update Frequency**: Quarterly
- **Authentication**: None required

---

## Project Structure

```
NursingREITs/
├── data/
│   ├── raw/                    # Original API responses
│   ├── interim/                # Staging area with UUID versioning
│   │   └── staging/           # Write-Audit-Publish staging
│   ├── processed/              # Validated, production-ready data
│   ├── analysis/               # Derived metrics and statistics
│   └── visualizations/         # Saved charts and maps
│
├── src/snf_reit_analysis/
│   ├── __init__.py
│   ├── config.py              # Pydantic configuration management
│   ├── dashboard.py           # Main Streamlit dashboard (1900+ lines)
│   │
│   ├── data/
│   │   ├── loaders.py         # API data loaders (CMS, BLS, SEC)
│   │   └── validators.py     # Pandera schema validation
│   │
│   ├── pipelines/
│   │   └── etl.py             # ETL orchestration with WAP pattern
│   │
│   ├── analysis/
│   │   ├── geographic_analyzer.py
│   │   ├── occupancy_tracker.py
│   │   └── tenant_mix_analyzer.py
│   │
│   ├── pages/
│   │   └── actual_reit_portfolios.py  # Geographic property analysis
│   │
│   └── utils/
│       ├── helpers.py         # Utility functions
│       └── logging.py         # Structured logging
│
├── tests/
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
│
├── notebooks/
│   └── exploratory/            # Jupyter notebooks for ad-hoc analysis
│
├── models/                     # Saved models and experiments
├── reports/                    # Generated reports and figures
│
├── Makefile                    # Build automation
├── pyproject.toml             # Package configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Dashboard Pages

### 1. **Overview**
Portfolio summary with key metrics and latest REIT performance.

### 2. **CMS Nursing Homes**
- Quality rating distribution (1-5 stars)
- State-level filtering and analysis
- Ownership type breakdown
- Top states by facility count

### 3. **REIT Financials**
- Individual REIT selection
- Quarterly revenue trends (2020-2025)
- Total assets evolution
- YoY growth comparisons

### 4. **REIT Comparison**
- Side-by-side revenue comparison
- Market share analysis
- Latest quarter bar charts

### 5. **Actual REIT Properties**
- Geographic choropleth maps
- Property type distribution
- Regional breakdown (Northeast, South, Midwest, West)
- State-level concentration analysis

### 6. **Deep Dive Analysis**
- Size & scale comparison
- Growth & momentum analysis
- Profitability & returns
- Balance sheet strength
- Comprehensive 14-metric scorecard

### 7. **BLS Economic Data**
- Producer Price Index trends
- Employment metrics (wages, staffing)
- Cost pressure analysis
- Leading indicators for revenue forecasting

### 8. **Valuation Framework**
- Multi-source integration methodology
- Key valuation drivers (expandable sections)
- Cap rate recommendations
- Scenario analysis

### 9. **Data Explorer**
- Raw data table viewer
- CSV download functionality
- Dataset selection and filtering

---

## Technology Stack

### Core Technologies
- **Python 3.11+**: Primary language
- **Polars 0.20+**: Fast data processing (10-100x faster than pandas)
- **Pandas 2.2+**: Data manipulation and analysis
- **Streamlit 1.31+**: Interactive dashboard framework

### Data Engineering
- **Pydantic 2.6+**: Configuration management with type validation
- **Pandera 0.18+**: Data schema validation
- **Requests**: HTTP API client with retry logic
- **python-dotenv**: Environment variable management

### Visualization
- **Plotly 5.19+**: Interactive charts with hover tooltips
- **Matplotlib 3.8+**: Static publication-quality figures
- **Seaborn 0.13+**: Statistical visualizations

### Development Tools
- **pytest 8.0+**: Unit testing framework
- **Black**: Code formatting
- **Ruff**: Fast Python linting
- **MyPy**: Static type checking

---

## ETL Pipeline

### Write-Audit-Publish (WAP) Pattern

```python
# 1. WRITE: Save to staging with UUID
staging_id = write_to_staging(df, "cms_provider_info")

# 2. AUDIT: Validate schema and business rules
validation = audit(df, validate_cms_provider_data)

# 3. PUBLISH: Deploy to production if validation passes
if validation.passed:
    publish_to_production(df, "cms_provider_info")
```

**Benefits:**
- Atomic deployments
- Data quality gates
- Version history
- Rollback capability

---

## Sample Workflow

### Analyzing a New REIT Quarter

```bash
# 1. Fetch latest SEC filings (after earnings release)
make data-sec

# 2. Update BLS data (monthly on 2nd Friday)
make data-bls

# 3. Refresh dashboard
make dashboard

# 4. Navigate to "Deep Dive Analysis" page
# 5. Review comprehensive scorecard
# 6. Check "Valuation Framework" for updated cap rates
# 7. Export data via "Data Explorer" for further modeling
```

---

## Testing

```bash
# Run full test suite with coverage
make test

# Fast tests without coverage report
make test-fast

# Run specific test file
pytest tests/unit/test_validators.py -v

# Run with specific marker
pytest -m "not slow" -v
```

**Coverage Target**: 80%+ for core modules

---

## Development

### Code Quality Standards

```bash
# Before committing, run:
make format    # Auto-format code
make lint      # Check for issues
make test      # Ensure tests pass

# Or run all checks at once:
make check
```

### Adding New Data Sources

1. Create loader class in `src/snf_reit_analysis/data/loaders.py`
2. Add validation schema in `src/snf_reit_analysis/data/validators.py`
3. Create pipeline function in `src/snf_reit_analysis/pipelines/etl.py`
4. Add dashboard page in `src/snf_reit_analysis/dashboard.py`

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- **CMS**: Provider data and quality ratings
- **BLS**: Economic indicators and employment statistics
- **SEC**: XBRL financial data access
- **Streamlit**: Dashboard framework

---

## Contact

**Ruthvik C. Mukkamala**  
**Verition Fund Management Case Study**  
**Portfolio Manager**: Matt Adams

For questions or collaboration:
- GitHub: [@ruthvik](https://github.com/ruthvik)
- Email: contact@example.com

---

## Resources

- [CMS Provider Data Catalog](https://data.cms.gov/provider-data/)
- [BLS API Documentation](https://www.bls.gov/developers/)
- [SEC EDGAR API Guide](https://www.sec.gov/edgar/sec-api-documentation)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Built for data-driven healthcare real estate investment**