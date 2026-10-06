"""
ComplaintsPulse — Data Loading & Stratified Splitting Pipeline

Loads the raw CFPB dataset, maps product labels to standardized business
categories, handles missing values, and creates strictly isolated
train / validation / test splits with zero data leakage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    RAW_TEXT_COL,
    RAW_LABEL_COL,
    CATEGORY_MAP,
    TEST_RATIO,
    VAL_RATIO,
    RANDOM_STATE,
)
from src.preprocessing.cleaner import ComplaintCleaner


@dataclass
class DatasetSplits:
    """Container holding strictly isolated train, validation, and test splits."""
    X_train: pd.Series
    y_train: pd.Series
    X_val: pd.Series
    y_val: pd.Series
    X_test: pd.Series
    y_test: pd.Series
    label_names: list[str]

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def val_size(self) -> int:
        return len(self.X_val)

    @property
    def test_size(self) -> int:
        return len(self.X_test)

    @property
    def total_size(self) -> int:
        return self.train_size + self.val_size + self.test_size

    def summary(self) -> dict:
        """Returns split sizes and class distribution summaries."""
        return {
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "total_size": self.total_size,
            "train_distribution": self.y_train.value_counts(normalize=True).to_dict(),
            "val_distribution": self.y_val.value_counts(normalize=True).to_dict(),
            "test_distribution": self.y_test.value_counts(normalize=True).to_dict(),
        }


def load_raw_complaints(
    file_path: Optional[Path] = None,
    sample_size: Optional[int] = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Load raw complaints CSV and validate required columns.
    
    Args:
        file_path: Path to CSV. Defaults to DATASET_PATH in config.
        sample_size: If specified, returns a stratified sample for faster runs.
        random_state: Seed for sampling.
        
    Returns:
        pd.DataFrame with standardized columns 'text' and 'category'.
    """
    path = Path(file_path) if file_path else DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(f"Complaint dataset not found at: {path}")

    # Read required columns
    df = pd.read_csv(path, usecols=[RAW_TEXT_COL, RAW_LABEL_COL])

    # Drop nulls
    df = df.dropna(subset=[RAW_TEXT_COL, RAW_LABEL_COL]).copy()

    # Map product categories to clean business names
    df = df[df[RAW_LABEL_COL].isin(CATEGORY_MAP.keys())].copy()
    df["category"] = df[RAW_LABEL_COL].map(CATEGORY_MAP)
    df["text"] = df[RAW_TEXT_COL].astype(str)

    # Filter out empty or whitespace-only texts
    df = df[df["text"].str.strip().str.len() > 0].copy()

    # Optional stratified sampling for quick experiments
    if sample_size is not None and sample_size < len(df):
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["category"],
            random_state=random_state,
        )
        df = df.reset_index(drop=True)

    return df[["text", "category"]]


def load_and_split_data(
    file_path: Optional[Path] = None,
    test_ratio: float = TEST_RATIO,
    val_ratio: float = VAL_RATIO,
    sample_size: Optional[int] = None,
    clean_text_flag: bool = False,
    random_state: int = RANDOM_STATE,
) -> DatasetSplits:
    """
    Loads complaints data and creates strictly isolated train/val/test splits.
    
    Splitting is performed with stratification to maintain class ratios.
    
    Args:
        file_path: Path to dataset CSV.
        test_ratio: Fraction reserved for final holdout test set (e.g. 0.15).
        val_ratio: Fraction reserved for validation set from remainder (e.g. 0.15).
        sample_size: Optional cap on dataset size for fast runs/tests.
        clean_text_flag: If True, applies ComplaintCleaner to the texts.
        random_state: Random state for reproducibility.
        
    Returns:
        DatasetSplits dataclass.
    """
    df = load_raw_complaints(
        file_path=file_path,
        sample_size=sample_size,
        random_state=random_state,
    )

    if clean_text_flag:
        cleaner = ComplaintCleaner()
        df["text"] = df["text"].map(cleaner.clean)
        # Drop any that became empty after cleaning
        df = df[df["text"].str.strip().str.len() > 0].copy()

    X = df["text"]
    y = df["category"]

    # First split: Hold out Test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        stratify=y,
        random_state=random_state,
    )

    # Second split: Hold out Validation set from train_val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=random_state,
    )

    label_names = sorted(y.unique().tolist())

    return DatasetSplits(
        X_train=X_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        X_val=X_val.reset_index(drop=True),
        y_val=y_val.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        label_names=label_names,
    )
