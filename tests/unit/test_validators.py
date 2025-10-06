"""Tests for data validation schemas."""

import pandas as pd
import pytest
from pandera.errors import SchemaError

from snf_reit_analysis.data.validators import (
    validate_bls_data,
    validate_cms_provider_data,
    validate_sec_data,
)


def test_cms_provider_validation_success(sample_cms_provider_data):
    df = pd.DataFrame(sample_cms_provider_data)
    validated_df = validate_cms_provider_data(df)

    assert len(validated_df) == 2
    assert validated_df["Overall_Rating"].dtype == "int64"


def test_cms_provider_validation_invalid_rating():
    invalid_data = [{
        "Federal_Provider_Number": "015001",
        "Provider_Name": "Test Facility",
        "State": "CA",
        "Overall_Rating": 6,  # Invalid: must be 1-5
        "Number_of_Certified_Beds": 100,
    }]

    df = pd.DataFrame(invalid_data)

    with pytest.raises(SchemaError):
        validate_cms_provider_data(df)


def test_bls_validation_success():
    valid_data = [
        {
            "series_id": "PCU623110623110",
            "series_name": "PPI Overall",
            "year": "2024",
            "period": "M01",
            "value": 150.5,
            "latest": True
        }
    ]

    df = pd.DataFrame(valid_data)
    validated_df = validate_bls_data(df)

    assert len(validated_df) == 1
    assert validated_df["value"].dtype == "float64"


def test_bls_validation_invalid_year():
    invalid_data = [{
        "series_id": "PCU623110623110",
        "series_name": "PPI Overall",
        "year": "24",  # Invalid: must be YYYY
        "period": "M01",
        "value": 150.5,
        "latest": True
    }]

    df = pd.DataFrame(invalid_data)

    with pytest.raises(SchemaError):
        validate_bls_data(df)


def test_sec_validation_success():
    valid_data = [
        {
            "end": "2024-09-30",
            "val": 10000000000,
            "accn": "0001193125-24-123456",
            "fy": 2024,
            "fp": "Q3",
            "form": "10-Q"
        }
    ]

    df = pd.DataFrame(valid_data)
    validated_df = validate_sec_data(df)

    assert len(validated_df) == 1
    assert validated_df["fy"].dtype == "int64"


def test_sec_validation_invalid_form():
    invalid_data = [{
        "end": "2024-09-30",
        "val": 10000000000,
        "accn": "0001193125-24-123456",
        "fy": 2024,
        "fp": "Q3",
        "form": "INVALID"  # Not in allowed forms
    }]

    df = pd.DataFrame(invalid_data)

    with pytest.raises(SchemaError):
        validate_sec_data(df)
