"""Tests for configuration module."""

import pytest
from pydantic import ValidationError

from snf_reit_analysis.config import (
    AppConfig,
    BLSConfig,
    CMSConfig,
    SECConfig,
)


def test_cms_config_defaults():
    config = CMSConfig(BLS_API_KEY="dummy")
    assert config.base_url == "https://data.cms.gov/data-api/v1"
    assert config.provider_info_dataset == "4pq5-n9py"
    assert config.deficiencies_dataset == "r5ix-sfxw"
    assert config.page_size == 5000


def test_bls_config_requires_api_key():
    with pytest.raises(ValidationError):
        BLSConfig()


def test_bls_config_with_api_key(mock_env_vars):
    config = BLSConfig()
    assert config.api_key.get_secret_value() == "test_api_key_12345"
    assert config.ppi_overall == "PCU623110623110"


def test_sec_config_validates_user_agent():
    with pytest.raises(ValidationError, match="email"):
        SECConfig(user_agent="CompanyName")


def test_sec_config_valid_user_agent():
    config = SECConfig(user_agent="CompanyName test@example.com")
    assert "@" in config.user_agent


def test_sec_config_reit_ciks():
    config = SECConfig(user_agent="Test test@example.com")
    ciks = config.reit_ciks

    assert ciks["OHI"] == 888491
    assert ciks["CTRE"] == 1590717
    assert ciks["SBRA"] == 1492298
    assert len(ciks) == 3


def test_app_config_creates_directories(tmp_path, mock_env_vars):
    config = AppConfig(
        data_dir=tmp_path / "data",
        raw_data_dir=tmp_path / "data" / "raw",
        interim_data_dir=tmp_path / "data" / "interim",
        processed_data_dir=tmp_path / "data" / "processed",
        external_data_dir=tmp_path / "data" / "external",
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports"
    )

    assert config.data_dir.exists()
    assert config.raw_data_dir.exists()
    assert config.processed_data_dir.exists()
