"""Backtest configuration management.

This module defines the BacktestConfig dataclass and validation logic.
All configuration is frozen at run start and dumped to runs/<run_id>/config.json.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _default_bins() -> list[tuple[float, float]]:
    """Default Polymarket-style temperature bins in Fahrenheit.
    
    These are example bins; real bins should match market definitions.
    """
    return [
        (-float("inf"), 32.0),  # Below freezing
        (32.0, 50.0),
        (50.0, 60.0),
        (60.0, 70.0),
        (70.0, 80.0),
        (80.0, 90.0),
        (90.0, 100.0),
        (100.0, float("inf")),  # Extreme heat
    ]


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.
    
    All parameters are frozen at run start to ensure reproducibility.
    
    Attributes:
        station_ids: List of station identifiers (e.g., ["KLGA"])
        start_date_local: First target date to include (inclusive)
        end_date_local: Last target date to include (inclusive)
        min_coverage_hours: Minimum truth coverage for valid day (default 18)
        lead_hours_allowed: If set, only include these lead hours (e.g., [28, 29])
        bins_f: Temperature bin boundaries in Fahrenheit
        train_frac: Fraction of data for training (default 0.70)
        val_frac: Fraction of data for validation (default 0.15)
        model_type: Model type identifier (default "ridge")
        model_alpha: Ridge regularization parameter (default 1.0)
        sigma_type: Uncertainty model type (default "bucketed")
        sigma_buckets: Lead hour buckets for sigma estimation
        price_type: Price provider type (default "synthetic")
        price_noise: Noise std for synthetic prices (default 0.05)
        price_spread: Bid-ask spread for synthetic prices (default 0.02)
        edge_min: Minimum edge to trade (default 0.03)
        max_per_market_pct: Max bankroll % per market (default 0.02)
        max_total_pct: Max total exposure % (default 0.25)
        slippage: Execution slippage (default 0.01)
        initial_bankroll: Starting bankroll (default 10000.0)
        random_seed: Random seed for reproducibility (default 42)
    """
    
    # Required fields
    station_ids: list[str]
    start_date_local: date
    end_date_local: date
    
    # Data quality
    min_coverage_hours: int = 18
    lead_hours_allowed: list[int] | None = None
    
    # Market/bins
    bins_f: list[tuple[float, float]] = field(default_factory=_default_bins)
    
    # Train/val/test splits (test = 1 - train - val)
    train_frac: float = 0.70
    val_frac: float = 0.15
    
    # Model configuration
    model_type: str = "ridge"
    model_alpha: float = 1.0
    model_features: list[str] = field(default_factory=lambda: [
        "tmax_pred_f", "sin_doy", "cos_doy", "bias_7d", "bias_14d"
    ])
    
    # Uncertainty/sigma configuration
    sigma_type: str = "bucketed"
    sigma_buckets: list[tuple[int, int]] | None = None
    sigma_floor: float = 1.0  # Minimum sigma to prevent overconfidence
    
    # Pricing configuration
    price_type: str = "synthetic"
    price_noise: float = 0.05
    price_spread: float = 0.02
    
    # Strategy configuration
    edge_min: float = 0.03
    max_per_market_pct: float = 0.02
    max_total_pct: float = 0.25
    slippage: float = 0.01
    
    # Simulation
    initial_bankroll: float = 10000.0
    random_seed: int = 42
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
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
        if not 0 < self.train_frac < 1:
            errors.append(f"train_frac must be in (0, 1), got {self.train_frac}")
        if not 0 < self.val_frac < 1:
            errors.append(f"val_frac must be in (0, 1), got {self.val_frac}")
        if self.train_frac + self.val_frac >= 1:
            errors.append(
                f"train_frac + val_frac must be < 1, got {self.train_frac + self.val_frac}"
            )
        
        # Bins validation
        if not self.bins_f:
            errors.append("bins_f must not be empty")
        else:
            for i, (lo, hi) in enumerate(self.bins_f):
                if lo >= hi:
                    errors.append(f"Bin {i}: low ({lo}) must be < high ({hi})")
        
        # Strategy validation
        if not 0 <= self.edge_min < 1:
            errors.append(f"edge_min must be in [0, 1), got {self.edge_min}")
        if not 0 < self.max_per_market_pct <= 1:
            errors.append(
                f"max_per_market_pct must be in (0, 1], got {self.max_per_market_pct}"
            )
        if not 0 < self.max_total_pct <= 1:
            errors.append(f"max_total_pct must be in (0, 1], got {self.max_total_pct}")
        if not 0 <= self.slippage < 1:
            errors.append(f"slippage must be in [0, 1), got {self.slippage}")
        
        # Bankroll validation
        if self.initial_bankroll <= 0:
            errors.append(
                f"initial_bankroll must be positive, got {self.initial_bankroll}"
            )
        
        if errors:
            raise ValueError("BacktestConfig validation failed:\n  - " + "\n  - ".join(errors))
    
    @property
    def test_frac(self) -> float:
        """Fraction of data for testing."""
        return 1.0 - self.train_frac - self.val_frac
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert dates to ISO format strings
        d["start_date_local"] = self.start_date_local.isoformat()
        d["end_date_local"] = self.end_date_local.isoformat()
        # Handle infinity in bins
        d["bins_f"] = [
            (
                "-inf" if lo == float("-inf") else lo,
                "inf" if hi == float("inf") else hi,
            )
            for lo, hi in self.bins_f
        ]
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
    def from_dict(cls, d: dict[str, Any]) -> BacktestConfig:
        """Create config from dictionary."""
        d = d.copy()
        # Parse dates
        if isinstance(d.get("start_date_local"), str):
            d["start_date_local"] = date.fromisoformat(d["start_date_local"])
        if isinstance(d.get("end_date_local"), str):
            d["end_date_local"] = date.fromisoformat(d["end_date_local"])
        # Parse infinity in bins
        if "bins_f" in d:
            d["bins_f"] = [
                (
                    float("-inf") if lo == "-inf" else lo,
                    float("inf") if hi == "inf" else hi,
                )
                for lo, hi in d["bins_f"]
            ]
        return cls(**d)
    
    @classmethod
    def from_json(cls, json_str: str) -> BacktestConfig:
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, path: Path | str) -> BacktestConfig:
        """Load config from JSON file."""
        path = Path(path)
        return cls.from_json(path.read_text())


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
