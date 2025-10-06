"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import List

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CMSConfig(BaseSettings):
    """CMS API configuration for nursing home data."""

    base_url: str = "https://data.cms.gov/provider-data/api/1"
    provider_info_dataset: str = "4pq5-n9py"
    deficiencies_dataset: str = "r5ix-sfxw"
    quality_measures_dataset: str = "djen-97ju"
    page_size: int = 1000
    timeout: int = 30

    model_config = SettingsConfigDict(env_prefix="CMS_", env_file=".env", extra="ignore")


class BLSConfig(BaseSettings):
    """BLS API configuration for economic data."""

    api_key: SecretStr
    base_url: str = "https://api.bls.gov/publicAPI/v2"

    # PPI Series IDs for Nursing Care Facilities (NAICS 623110)
    ppi_overall: str = "PCU623110623110"
    ppi_medicare: str = "PCU623110623110101"
    ppi_medicaid: str = "PCU623110623110102"
    ppi_private: str = "PCU623110623110103"

    # Employment Series IDs (NAICS 6231 - Nursing care facilities)
    ces_all_employees: str = "CES6562310001"
    ces_avg_hourly_earnings: str = "CES6562310003"
    ces_avg_weekly_hours: str = "CES6562310006"
    ces_avg_weekly_earnings: str = "CES6562310011"

    timeout: int = 30
    max_series_per_request: int = 50
    max_years_per_request: int = 20

    model_config = SettingsConfigDict(env_prefix="BLS_", env_file=".env", extra="ignore")


class SECConfig(BaseSettings):
    """SEC EDGAR API configuration for REIT financial data."""

    base_url: str = "https://data.sec.gov"
    user_agent: str = Field(
        default="SNF-REIT-Analysis contact@example.com",
        description="Required User-Agent header per SEC API policy"
    )

    # REIT CIK codes
    ohi_cik: int = 888491  # Omega Healthcare Investors
    ctre_cik: int = 1590717  # CareTrust REIT
    sbra_cik: int = 1492298  # Sabra Health Care REIT

    rate_limit_delay: float = 0.11  # 10 requests/second = 0.1s + buffer
    timeout: int = 30

    @field_validator("user_agent")
    @classmethod
    def validate_user_agent(cls, v: str) -> str:
        """Ensure user agent contains email per SEC requirements."""
        if "@" not in v:
            raise ValueError("User-Agent must contain an email address")
        return v

    model_config = SettingsConfigDict(env_prefix="SEC_", env_file=".env", extra="ignore")

    @property
    def reit_ciks(self) -> dict[str, int]:
        """Get dictionary of REIT tickers to CIK codes."""
        return {
            "OHI": self.ohi_cik,
            "CTRE": self.ctre_cik,
            "SBRA": self.sbra_cik,
        }


class AppConfig(BaseSettings):
    """Main application configuration."""

    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # Data directories
    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    raw_data_dir: Path = Field(default_factory=lambda: Path("data/raw"))
    interim_data_dir: Path = Field(default_factory=lambda: Path("data/interim"))
    processed_data_dir: Path = Field(default_factory=lambda: Path("data/processed"))
    external_data_dir: Path = Field(default_factory=lambda: Path("data/external"))

    # Models and reports
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    reports_dir: Path = Field(default_factory=lambda: Path("reports"))

    # API configurations
    cms: CMSConfig = Field(default_factory=CMSConfig)
    bls: BLSConfig = Field(default_factory=BLSConfig)
    sec: SECConfig = Field(default_factory=SECConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

    def __init__(self, **kwargs):
        """Initialize and ensure data directories exist."""
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self) -> None:
        """Create all required data directories."""
        for dir_path in [
            self.data_dir,
            self.raw_data_dir,
            self.interim_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.models_dir,
            self.reports_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = AppConfig()
