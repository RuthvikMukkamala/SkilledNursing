"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def mock_env_vars():
    """Mock environment variables."""
    env_vars = {
        "BLS_API_KEY": "test_api_key_12345",
        "SEC_USER_AGENT": "TestCompany test@example.com",
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG"
    }

    # Set env vars
    for key, value in env_vars.items():
        os.environ[key] = value

    yield env_vars

    # Cleanup
    for key in env_vars.keys():
        os.environ.pop(key, None)


@pytest.fixture
def sample_cms_provider_data():
    return [
        {
            "Federal_Provider_Number": "015001",
            "Provider_Name": "Test Nursing Home 1",
            "State": "CA",
            "Overall_Rating": 5,
            "Health_Inspection_Rating": 5,
            "Staffing_Rating": 4,
            "QM_Rating": 5,
            "Number_of_Certified_Beds": 100,
            "Average_Number_of_Residents_per_Day": 85.5
        },
        {
            "Federal_Provider_Number": "015002",
            "Provider_Name": "Test Nursing Home 2",
            "State": "TX",
            "Overall_Rating": 3,
            "Health_Inspection_Rating": 3,
            "Staffing_Rating": 3,
            "QM_Rating": 4,
            "Number_of_Certified_Beds": 75,
            "Average_Number_of_Residents_per_Day": 60.0
        }
    ]


@pytest.fixture
def sample_bls_data():
    return {
        "status": "REQUEST_SUCCEEDED",
        "responseTime": 100,
        "message": [],
        "Results": {
            "series": [
                {
                    "seriesID": "PCU623110623110",
                    "data": [
                        {
                            "year": "2024",
                            "period": "M01",
                            "periodName": "January",
                            "value": "150.5",
                            "latest": True
                        },
                        {
                            "year": "2023",
                            "period": "M12",
                            "periodName": "December",
                            "value": "148.2",
                            "latest": False
                        }
                    ]
                }
            ]
        }
    }


@pytest.fixture
def sample_sec_company_facts():
    return {
        "cik": 888491,
        "entityName": "Omega Healthcare Investors Inc",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "label": "Assets",
                    "description": "Total assets",
                    "units": {
                        "USD": [
                            {
                                "end": "2024-09-30",
                                "val": 10000000000,
                                "accn": "0001193125-24-123456",
                                "fy": 2024,
                                "fp": "Q3",
                                "form": "10-Q",
                                "filed": "2024-11-01"
                            },
                            {
                                "end": "2023-12-31",
                                "val": 9500000000,
                                "accn": "0001193125-24-000001",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-15"
                            }
                        ]
                    }
                }
            }
        }
    }
