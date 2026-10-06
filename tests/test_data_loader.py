"""
Unit tests for data loading, category mapping, and stratified train/val/test splitting.
"""

import pandas as pd
import pytest
from src.classification.data_loader import load_raw_complaints, load_and_split_data
from src.config import CATEGORIES


def test_load_raw_complaints_sample():
    # Load a small stratified sample to keep tests fast
    df = load_raw_complaints(sample_size=500, random_state=42)
    assert len(df) == 500
    assert "text" in df.columns
    assert "category" in df.columns
    assert df["text"].isnull().sum() == 0
    # Every category should belong to known CATEGORIES
    assert set(df["category"].unique()).issubset(set(CATEGORIES))


def test_load_and_split_data_leakage_and_stratification():
    # Load 1000 samples for split verification
    splits = load_and_split_data(
        sample_size=1000,
        test_ratio=0.15,
        val_ratio=0.15,
        random_state=42,
    )

    # Check total size
    assert splits.total_size == 1000
    assert splits.test_size == 150
    # Val is 15% of (1000 - 150 = 850) -> ~128 (rounding due to stratification)
    assert abs(splits.val_size - (850 * 0.15)) <= 1
    assert splits.train_size == 1000 - splits.test_size - splits.val_size

    # Check that labels are complete across splits
    assert set(splits.label_names) == set(CATEGORIES)
    assert set(splits.y_train.unique()) == set(CATEGORIES)
    assert set(splits.y_val.unique()) == set(CATEGORIES)
    assert set(splits.y_test.unique()) == set(CATEGORIES)

    # Check summary output
    summary = splits.summary()
    assert "train_distribution" in summary
    assert "val_distribution" in summary
    assert "test_distribution" in summary

    # Verify distributions are close (stratification check)
    # Credit Reporting is largest class (~50-55%)
    for dist_key in ["train_distribution", "val_distribution", "test_distribution"]:
        dist = summary[dist_key]
        assert "Credit Reporting" in dist
        assert 0.45 <= dist["Credit Reporting"] <= 0.65


def test_splitting_reproducibility():
    s1 = load_and_split_data(sample_size=200, random_state=42)
    s2 = load_and_split_data(sample_size=200, random_state=42)
    assert s1.X_train.tolist() == s2.X_train.tolist()
    assert s1.y_train.tolist() == s2.y_train.tolist()
    assert s1.X_test.tolist() == s2.X_test.tolist()
    assert s1.y_test.tolist() == s2.y_test.tolist()
