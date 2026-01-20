"""Evaluation configuration management.

This module defines the EvalConfig dataclass for temperature evaluation runs.
All configuration is frozen at run start and dumped to runs/<run_id>/config.json.

No trading/market fields are included - this is pure temperature evaluation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal


@dataclass
class SplitConfig:
    """Configuration for train/val/test splitting.

    Attributes:
        type: Split strategy ("static" or "walk_forward")
        train_frac: Fraction for training (static split)
        val_frac: Fraction for validation (static split)
        test_frac: Fraction for testing (static split)
        window_size: Window size in days (walk_forward)
        step_size: Step size in days (walk_forward)
    """
    type: Literal["static", "walk_forward"] = "static"
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    # Walk-forward params
    window_size: int | None = None
    step_size: int | None = None


@dataclass
class ModelConfig:
    """Configuration for the forecasting model.

    Attributes:
        type: Model type ("passthrough" or "ridge")
        alpha: Ridge regularization parameter
        features: List of feature column names
    """
    type: Literal["passthrough", "ridge"] = "ridge"
    alpha: float = 1.0
    features: list[str] = field(default_factory=lambda: [
        "tmax_pred_f", "sin_doy", "cos_doy", "bias_7d", "bias_14d"
    ])


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation.

    Attributes:
        type: Uncertainty model type ("global", "bucketed", "rolling")
        buckets: Lead hour buckets for bucketed sigma
        sigma_floor: Minimum sigma value to prevent overconfidence
    """
    type: Literal["global", "bucketed", "rolling"] = "bucketed"
    buckets: list[tuple[int, int]] | None = None
    sigma_floor: float = 1.0


@dataclass
class EvalConfig:
    """Configuration for a temperature evaluation run.

    All parameters are frozen at run start to ensure reproducibility.
    No trading/market concepts - pure temperature evaluation only.

    Attributes:
        run_name: Human-readable name for this run
        station_ids: List of station identifiers (e.g., ["KLGA"])
        start_date_local: First target date to include (inclusive)
        end_date_local: Last target date to include (inclusive)
        min_coverage_hours: Minimum truth coverage for valid day (default 18)
        lead_hours_allowed: If set, only include these lead hours
        split: Train/val/test split configuration
        model: Forecasting model configuration
        uncertainty: Uncertainty estimation configuration
        random_seed: Random seed for reproducibility
    """

    # Required fields
    run_name: str
    station_ids: list[str]
    start_date_local: date
    end_date_local: date

    # Data quality
    min_coverage_hours: int = 18
    lead_hours_allowed: list[int] | None = None

    # Split configuration
    split: SplitConfig = field(default_factory=SplitConfig)

    # Model configuration
    model: ModelConfig = field(default_factory=ModelConfig)

    # Uncertainty configuration
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)

    # Reproducibility
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Convert nested dicts to dataclasses if needed
        if isinstance(self.split, dict):
            self.split = SplitConfig(**self.split)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.uncertainty, dict):
            self.uncertainty = UncertaintyConfig(**self.uncertainty)
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration parameters."""
        errors = []

        # Station validation
        if not self.station_ids:
            errors.append("station_ids must not be empty")

        # Date validation
        if self.start_date_local >= self.end_date_local:
            errors.append(
                f"start_date_local ({self.start_date_local}) must be before "
                f"end_date_local ({self.end_date_local})"
            )

        # Coverage validation
        if not 0 <= self.min_coverage_hours <= 24:
            errors.append(
                f"min_coverage_hours must be in [0, 24], got {self.min_coverage_hours}"
            )

        # Split validation
        if self.split.type == "static":
            total = self.split.train_frac + self.split.val_frac + self.split.test_frac
            if abs(total - 1.0) > 1e-6:
                errors.append(
                    f"Split fractions must sum to 1, got {total}"
                )
        elif self.split.type == "walk_forward":
            if self.split.window_size is None or self.split.window_size <= 0:
                errors.append("walk_forward requires positive window_size")
            if self.split.step_size is None or self.split.step_size <= 0:
                errors.append("walk_forward requires positive step_size")

        if errors:
            raise ValueError("EvalConfig validation failed:\n  - " + "\n  - ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert dates to ISO format strings
        d["start_date_local"] = self.start_date_local.isoformat()
        d["end_date_local"] = self.end_date_local.isoformat()
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Path | str) -> Path:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalConfig:
        """Create config from dictionary."""
        d = d.copy()
        # Parse dates
        if isinstance(d.get("start_date_local"), str):
            d["start_date_local"] = date.fromisoformat(d["start_date_local"])
        if isinstance(d.get("end_date_local"), str):
            d["end_date_local"] = date.fromisoformat(d["end_date_local"])
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> EvalConfig:
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path | str) -> EvalConfig:
        """Load config from JSON file."""
        path = Path(path)
        return cls.from_json(path.read_text())


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_git_commit() -> str | None:
    """Get the current git commit hash, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def create_run_metadata() -> dict[str, Any]:
    """Create metadata for the current run."""
    return {
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }
