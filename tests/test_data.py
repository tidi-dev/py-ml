"""Tests for the dataset loader.

These tests check that the data looks the way the course describes it.
If one of them fails, something about the data changed and the lessons
may no longer match what you see.
"""

import pandas as pd
import pytest

from house_price.data import FEATURE_COLUMNS, TARGET_COLUMN, load_housing_data


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    # Load once and share between tests; loading is the slow part.
    return load_housing_data()


def test_returns_a_dataframe(df: pd.DataFrame) -> None:
    assert isinstance(df, pd.DataFrame)


def test_has_one_row_per_block_group(df: pd.DataFrame) -> None:
    assert df.shape == (20_640, 9)


def test_columns_are_the_eight_features_then_the_target(df: pd.DataFrame) -> None:
    assert list(df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]


def test_has_no_missing_values(df: pd.DataFrame) -> None:
    assert df.isna().sum().sum() == 0


def test_all_columns_are_numeric(df: pd.DataFrame) -> None:
    assert all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)


def test_target_is_in_units_of_100k_dollars(df: pd.DataFrame) -> None:
    # Values run from about 0.15 ($15,000) to about 5.0 ($500,000).
    # The dataset caps high values at 5.0, which is why the max is not larger.
    assert df[TARGET_COLUMN].min() == pytest.approx(0.15, abs=0.01)
    assert df[TARGET_COLUMN].max() == pytest.approx(5.0, abs=0.01)
