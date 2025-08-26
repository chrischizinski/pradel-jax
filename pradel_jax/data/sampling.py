"""
Stratified sampling utilities for pradel-jax.

Provides functions for stratified random sampling based on tier status and other criteria.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path

from .adapters import DataContext, load_data
from ..utils.logging import get_logger


logger = get_logger(__name__)


def determine_tier_status(
    data: pd.DataFrame, tier_columns: Optional[List[str]] = None
) -> pd.Series:
    """
    Determine if each individual was ever a tier 2 person.

    Args:
        data: DataFrame with tier information
        tier_columns: List of column names containing tier information.
                     If None, auto-detects columns starting with 'tier'

    Returns:
        Series with boolean values: True if ever tier 2, False if only tier 1
    """
    if tier_columns is None:
        # Auto-detect tier columns
        tier_columns = [col for col in data.columns if "tier" in col.lower()]

        if not tier_columns:
            logger.warning(
                "No tier columns found - checking for common tier column names"
            )
            # Check for common variations
            common_tier_names = [
                "tier",
                "tier_1",
                "tier_2",
                "tier_status",
                "license_tier",
            ]
            tier_columns = [
                col
                for col in data.columns
                if any(name in col.lower() for name in common_tier_names)
            ]

    if not tier_columns:
        raise ValueError(
            "No tier columns found. Please specify tier_columns explicitly or ensure "
            "tier information is available in columns with 'tier' in the name."
        )

    logger.info(f"Using tier columns: {tier_columns}")

    # Check if any tier column contains value >= 2 for each individual
    ever_tier_2 = pd.Series(False, index=data.index)

    for col in tier_columns:
        if col in data.columns:
            # Convert to numeric, handling any string values
            tier_values = pd.to_numeric(data[col], errors="coerce")
            ever_tier_2 |= tier_values >= 2

    return ever_tier_2


def stratified_sample(
    data: pd.DataFrame,
    n_samples: int,
    tier_columns: Optional[List[str]] = None,
    tier_2_proportion: float = 0.5,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Perform stratified sampling based on tier 2 status.

    Args:
        data: DataFrame to sample from
        n_samples: Total number of samples to draw
        tier_columns: List of tier column names (auto-detected if None)
        tier_2_proportion: Target proportion of tier 2 individuals (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        Stratified sample of the data
    """
    if n_samples >= len(data):
        logger.info(
            f"Requested {n_samples} samples from {len(data)} rows - returning all data"
        )
        return data.copy()

    # Determine tier 2 status
    ever_tier_2 = determine_tier_status(data, tier_columns)

    # Split into strata
    tier_2_data = data[ever_tier_2]
    tier_1_only_data = data[~ever_tier_2]

    logger.info(
        f"Population composition: {len(tier_2_data)} ever tier 2, {len(tier_1_only_data)} tier 1 only"
    )

    # Calculate target sample sizes
    n_tier_2_target = int(n_samples * tier_2_proportion)
    n_tier_1_target = n_samples - n_tier_2_target

    # Adjust if we don't have enough in either stratum
    n_tier_2_available = len(tier_2_data)
    n_tier_1_available = len(tier_1_only_data)

    if n_tier_2_target > n_tier_2_available:
        logger.warning(
            f"Not enough tier 2 individuals ({n_tier_2_available} available, {n_tier_2_target} requested)"
        )
        n_tier_2_actual = n_tier_2_available
        n_tier_1_actual = min(n_samples - n_tier_2_actual, n_tier_1_available)
    elif n_tier_1_target > n_tier_1_available:
        logger.warning(
            f"Not enough tier 1 individuals ({n_tier_1_available} available, {n_tier_1_target} requested)"
        )
        n_tier_1_actual = n_tier_1_available
        n_tier_2_actual = min(n_samples - n_tier_1_actual, n_tier_2_available)
    else:
        n_tier_2_actual = n_tier_2_target
        n_tier_1_actual = n_tier_1_target

    logger.info(
        f"Sampling {n_tier_2_actual} tier 2 and {n_tier_1_actual} tier 1 individuals"
    )

    # Sample from each stratum
    sampled_data = []

    if n_tier_2_actual > 0:
        tier_2_sample = tier_2_data.sample(n=n_tier_2_actual, random_state=random_state)
        sampled_data.append(tier_2_sample)

    if n_tier_1_actual > 0:
        tier_1_sample = tier_1_only_data.sample(
            n=n_tier_1_actual, random_state=random_state
        )
        sampled_data.append(tier_1_sample)

    if not sampled_data:
        raise ValueError("No data could be sampled")

    # Combine samples
    result = pd.concat(sampled_data, ignore_index=True)

    # Shuffle the result to mix tier 1 and tier 2 individuals
    if random_state is not None:
        result = result.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        )
    else:
        result = result.sample(frac=1.0).reset_index(drop=True)

    # Log final composition
    final_tier_2 = determine_tier_status(result, tier_columns)
    final_tier_2_count = final_tier_2.sum()
    final_tier_1_count = len(result) - final_tier_2_count
    final_tier_2_prop = final_tier_2_count / len(result) if len(result) > 0 else 0

    logger.info(
        f"Final sample: {len(result)} individuals ({final_tier_2_count} tier 2, {final_tier_1_count} tier 1, {final_tier_2_prop:.1%} tier 2)"
    )

    return result


def train_validation_split(
    data: pd.DataFrame,
    validation_size: float = 0.2,
    tier_columns: Optional[List[str]] = None,
    random_state: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets with stratification by tier status.

    Args:
        data: DataFrame to split
        validation_size: Proportion of data for validation (0.0 to 1.0)
        tier_columns: List of tier column names (auto-detected if None)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (training_data, validation_data)
    """
    if validation_size <= 0 or validation_size >= 1:
        raise ValueError("validation_size must be between 0 and 1")

    # Determine tier 2 status for stratification
    ever_tier_2 = determine_tier_status(data, tier_columns)

    # Split each stratum
    tier_2_data = data[ever_tier_2]
    tier_1_data = data[~ever_tier_2]

    train_data_parts = []
    val_data_parts = []

    # Split tier 2 data
    if len(tier_2_data) > 0:
        n_val_tier_2 = max(1, int(len(tier_2_data) * validation_size))
        tier_2_val = tier_2_data.sample(n=n_val_tier_2, random_state=random_state)
        tier_2_train = tier_2_data.drop(tier_2_val.index)

        train_data_parts.append(tier_2_train)
        val_data_parts.append(tier_2_val)

    # Split tier 1 data
    if len(tier_1_data) > 0:
        n_val_tier_1 = max(1, int(len(tier_1_data) * validation_size))
        tier_1_val = tier_1_data.sample(n=n_val_tier_1, random_state=random_state)
        tier_1_train = tier_1_data.drop(tier_1_val.index)

        train_data_parts.append(tier_1_train)
        val_data_parts.append(tier_1_val)

    # Combine and shuffle
    train_data = pd.concat(train_data_parts, ignore_index=True)
    val_data = pd.concat(val_data_parts, ignore_index=True)

    if random_state is not None:
        train_data = train_data.sample(frac=1.0, random_state=random_state).reset_index(
            drop=True
        )
        val_data = val_data.sample(frac=1.0, random_state=random_state + 1).reset_index(
            drop=True
        )

    logger.info(f"Data split: {len(train_data)} training, {len(val_data)} validation")

    return train_data, val_data


def load_data_with_sampling(
    file_path: Union[str, Path],
    n_samples: Optional[int] = None,
    tier_columns: Optional[List[str]] = None,
    tier_2_proportion: float = 0.5,
    validation_split: Optional[float] = None,
    random_state: Optional[int] = None,
    **load_kwargs,
) -> Union[DataContext, Tuple[DataContext, DataContext]]:
    """
    Load data with optional stratified sampling and train/validation split.

    Args:
        file_path: Path to data file
        n_samples: Number of samples to draw (None for no sampling)
        tier_columns: List of tier column names (auto-detected if None)
        tier_2_proportion: Target proportion of tier 2 individuals
        validation_split: Proportion for validation set (None for no split)
        random_state: Random seed for reproducibility
        **load_kwargs: Additional arguments for load_data()

    Returns:
        DataContext if no validation split, else (train_context, val_context)
    """
    # First load the raw data to check format
    if Path(file_path).suffix.lower() == ".csv":
        # Peek at the data to determine format
        peek_data = pd.read_csv(file_path, nrows=100)
    else:
        raise ValueError(f"Unsupported file format: {Path(file_path).suffix}")

    if n_samples is None:
        # No sampling - load normally
        logger.info("Loading data without sampling")
        return load_data(file_path, **load_kwargs)

    # Load full data for sampling
    logger.info(f"Loading data for stratified sampling (target: {n_samples} samples)")
    full_data = pd.read_csv(file_path)

    # Perform stratified sampling
    sampled_data = stratified_sample(
        full_data,
        n_samples=n_samples,
        tier_columns=tier_columns,
        tier_2_proportion=tier_2_proportion,
        random_state=random_state,
    )

    # Optionally split into train/validation
    if validation_split is not None:
        train_data, val_data = train_validation_split(
            sampled_data,
            validation_size=validation_split,
            tier_columns=tier_columns,
            random_state=random_state,
        )
    else:
        train_data = sampled_data
        val_data = None

    # Helper function to load data from DataFrame
    def load_from_dataframe(df: pd.DataFrame, split_type: str = "full") -> DataContext:
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            temp_path = tmp_file.name

        try:
            # Load through normal pipeline
            context = load_data(temp_path, **load_kwargs)

            # Add metadata
            if context.metadata is None:
                context.metadata = {}

            context.metadata.update(
                {
                    "original_file": str(file_path),
                    "sampling_applied": True,
                    "split_type": split_type,
                    "original_n_individuals": len(full_data),
                    "sampled_n_individuals": len(sampled_data),
                    "final_n_individuals": len(df),
                    "tier_2_proportion_target": tier_2_proportion,
                    "validation_split": validation_split,
                    "random_state": random_state,
                    "tier_columns_used": tier_columns,
                }
            )

            return context

        finally:
            Path(temp_path).unlink(missing_ok=True)

    # Load training data
    train_context = load_from_dataframe(train_data, "training")

    # Return single context or tuple
    if val_data is not None:
        val_context = load_from_dataframe(val_data, "validation")
        return train_context, val_context
    else:
        return train_context


def get_sampling_summary(data_context: DataContext) -> Dict[str, any]:
    """
    Get summary of sampling applied to data.

    Args:
        data_context: DataContext to analyze

    Returns:
        Dictionary with sampling summary information
    """
    if data_context.metadata is None or not data_context.metadata.get(
        "sampling_applied", False
    ):
        return {"sampling_applied": False}

    return {
        "sampling_applied": True,
        "original_file": data_context.metadata.get("original_file"),
        "original_n_individuals": data_context.metadata.get("original_n_individuals"),
        "sampled_n_individuals": data_context.metadata.get("sampled_n_individuals"),
        "sampling_ratio": (
            data_context.metadata.get("sampled_n_individuals", 0)
            / data_context.metadata.get("original_n_individuals", 1)
        ),
        "tier_2_proportion_target": data_context.metadata.get(
            "tier_2_proportion_target"
        ),
        "tier_columns_used": data_context.metadata.get("tier_columns_used"),
        "random_state": data_context.metadata.get("random_state"),
    }
