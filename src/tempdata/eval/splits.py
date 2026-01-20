"""Time-based splitting strategies for evaluation.

Provides train/val/test splitting with no data leakage:
- StaticSplit: Fixed train/val/test fractions by time
- WalkForwardSplit: Rolling window cross-validation

All splits respect temporal ordering to prevent leakage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tempdata.eval.config import SplitConfig


class Split(ABC):
    """Abstract base class for splitting strategies."""

    @abstractmethod
    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.

        Args:
            df: DataFrame to split (must be sorted by time)

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        pass


@dataclass
class StaticSplit(Split):
    """Static train/val/test split by time fraction.

    Data is split by temporal order, with no shuffling.

    Attributes:
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
        test_frac: Fraction of data for testing
    """
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    def __post_init__(self) -> None:
        """Validate fractions sum to 1."""
        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Fractions must sum to 1, got {total}")

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by time fractions.

        Args:
            df: DataFrame sorted by time

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        train_end = int(n * self.train_frac)
        val_end = int(n * (self.train_frac + self.val_frac))

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)

        return train_df, val_df, test_df


@dataclass
class WalkForwardSplit(Split):
    """Walk-forward (rolling window) cross-validation split.

    Creates expanding or rolling training windows with a fixed test period.
    Useful for time-series evaluation where you want to simulate
    real-world model updates.

    Attributes:
        window_size: Number of days for training window
        step_size: Number of days to step forward each iteration
        expanding: If True, training window expands; if False, it rolls
    """
    window_size: int
    step_size: int
    expanding: bool = False

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split for walk-forward validation.

        For simplicity, returns the final fold:
        - Train: all data up to window_size from end
        - Val: empty (walk-forward doesn't use separate validation)
        - Test: final step_size days

        For full walk-forward iteration, use generate_folds().

        Args:
            df: DataFrame sorted by time

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        test_start = n - self.step_size
        train_end = test_start

        if self.expanding:
            train_start = 0
        else:
            train_start = max(0, train_end - self.window_size)

        train_df = df.iloc[train_start:train_end].reset_index(drop=True)
        val_df = pd.DataFrame(columns=df.columns)  # Empty
        test_df = df.iloc[test_start:].reset_index(drop=True)

        return train_df, val_df, test_df

    def generate_folds(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Generate all walk-forward folds.

        Yields (train_df, test_df) pairs for each fold, advancing
        by step_size each iteration.

        Args:
            df: DataFrame sorted by time

        Returns:
            List of (train_df, test_df) tuples for each fold
        """
        n = len(df)
        folds = []

        start = self.window_size
        while start + self.step_size <= n:
            if self.expanding:
                train_start = 0
            else:
                train_start = start - self.window_size

            train_df = df.iloc[train_start:start].reset_index(drop=True)
            test_df = df.iloc[start:start + self.step_size].reset_index(drop=True)

            folds.append((train_df, test_df))
            start += self.step_size

        return folds


def create_split(config: SplitConfig) -> Split:
    """Create a Split instance from configuration.

    Args:
        config: SplitConfig with split parameters

    Returns:
        Configured Split instance
    """
    if config.type == "static":
        return StaticSplit(
            train_frac=config.train_frac,
            val_frac=config.val_frac,
            test_frac=config.test_frac,
        )
    elif config.type == "walk_forward":
        if config.window_size is None or config.step_size is None:
            raise ValueError("walk_forward requires window_size and step_size")
        return WalkForwardSplit(
            window_size=config.window_size,
            step_size=config.step_size,
        )
    else:
        raise ValueError(f"Unknown split type: {config.type}")
