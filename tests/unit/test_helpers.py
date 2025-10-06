"""Tests for utility helper functions."""

import pytest

from snf_reit_analysis.utils.helpers import (
    calculate_debt_to_assets,
    calculate_occupancy_rate,
    calculate_roa,
    format_currency,
    format_percentage,
)


def test_calculate_occupancy_rate():
    assert calculate_occupancy_rate(85, 100) == 0.85
    assert calculate_occupancy_rate(100, 100) == 1.0
    assert calculate_occupancy_rate(105, 100) == 1.0  # Capped at 100%
    assert calculate_occupancy_rate(50, 0) == 0.0  # Handle zero beds


def test_calculate_debt_to_assets():
    assert calculate_debt_to_assets(500000, 1000000) == 0.5
    assert calculate_debt_to_assets(750000, 1000000) == 0.75
    assert calculate_debt_to_assets(100000, 0) == 0.0  # Handle zero assets


def test_calculate_roa():
    assert calculate_roa(100000, 1000000) == 0.1
    assert calculate_roa(-50000, 1000000) == -0.05  # Negative income
    assert calculate_roa(100000, 0) == 0.0  # Handle zero assets


def test_format_currency():
    assert format_currency(1500000000) == "$1.50B"
    assert format_currency(2500000) == "$2.50M"
    assert format_currency(5000) == "$5.00K"
    assert format_currency(150) == "$150.00"
    assert format_currency(-1000000) == "$-1.00M"


def test_format_percentage():
    assert format_percentage(0.15) == "15.00%"
    assert format_percentage(0.156) == "15.60%"
    assert format_percentage(0.156, decimals=1) == "15.6%"
    assert format_percentage(1.0) == "100.00%"
    assert format_percentage(-0.05) == "-5.00%"
