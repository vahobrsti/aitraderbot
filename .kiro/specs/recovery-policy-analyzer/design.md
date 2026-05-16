# Technical Design Document: Recovery Policy Analyzer

## Overview

The Recovery Policy Analyzer is a Django management command that extends the existing `analyze_loser_recovery.py` to provide policy-driven analysis of trades that fail to hit their target by the TTH p75 checkpoint day. The analyzer uses signal-specific parameters from `policy.py` for checkpoint timing and adverse thresholds, outputting a decision matrix to guide hold vs flip decisions for adverse trades.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Recovery Policy Analyzer                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │  PolicyConfig    │    │   TradeBuilder   │    │ RecoveryAnalyzer │       │
│  │  Adapter         │    │   (reused from   │    │                  │       │
│  │                  │    │   hit_rate.py)   │    │                  │       │
│  └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘       │
│           │                       │                       │                  │
│           ▼                       ▼                       ▼                  │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                      RecoveryCandidate                            │       │
│  │  (dataclass holding trade + checkpoint + recovery metrics)        │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                      DecisionMatrix                               │       │
│  │  (categorizes: both_win, hold_wins, flip_wins, both_lose)         │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                   │                                          │
│                                   ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │                      ConsoleOutput                                │       │
│  │  (summary tables, decision matrix, recommendations)               │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           External Dependencies                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  execution.services.policy    │  PolicyVersion, get_policy()                 │
│  datafeed.models              │  RawDailyData (price data)                   │
│  signals.fusion               │  fuse_signals, add_fusion_features           │
│  signals.overlays             │  apply_overlays, get_size_multiplier         │
│  signals.tactical_puts        │  tactical_put_inside_bull                    │
│  signals.mvrv_short           │  check_mvrv_short_signal                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
1. CLI Arguments Parsed
         │
         ▼
2. PolicyConfigAdapter.get_checkpoint_day(signal_type)
   PolicyConfigAdapter.get_adverse_threshold(signal_type)
         │
         ▼
3. TradeBuilder._build_trades_df() → DataFrame of trades
         │
         ▼
4. Filter: exclude IRON_CONDOR, apply --type/--year filters
         │
         ▼
5. RecoveryAnalyzer._analyze_recovery()
   For each trade:
   ├── Get checkpoint day from policy
   ├── Calculate checkpoint return
   ├── Check if adverse (direction-specific)
   ├── If adverse: calculate recovery metrics
   └── Create RecoveryCandidate
         │
         ▼
6. DecisionMatrix.categorize(candidates)
   ├── both_win: original_hit AND recovery_hit
   ├── hold_wins: original_hit AND NOT recovery_hit
   ├── flip_wins: NOT original_hit AND recovery_hit
   └── both_lose: NOT original_hit AND NOT recovery_hit
         │
         ▼
7. ConsoleOutput.print_results()
   ├── Summary by trade type
   ├── Overall statistics
   ├── Decision matrix
   ├── Recommendation (based on net edge)
   └── Detailed trade list (sorted by recovery potential)
```

## Components

### 1. PolicyConfigAdapter

Adapts the `PolicyVersion` from `policy.py` to provide checkpoint and threshold values.

```python
from dataclasses import dataclass
from typing import Optional
from execution.services.policy import get_policy, PolicyVersion


@dataclass
class PolicyConfigAdapter:
    """
    Adapts PolicyVersion to provide recovery analysis parameters.
    
    Extracts TTH p75 for checkpoint days and MAE(W) p75 for adverse thresholds
    from the path_profiles in policy.py.
    """
    policy: PolicyVersion
    
    # Default values when signal type not found in policy
    DEFAULT_CHECKPOINT_DAY: int = 7
    DEFAULT_ADVERSE_THRESHOLD: float = 0.02  # 2%
    
    # TTH p75 values derived from DTE targets in policy
    # These map signal types to their TTH p75 checkpoint days
    TTH_P75_MAP: dict[str, int] = {
        "CALL": 7,
        "LONG": 7,  # Alias for CALL
        "PUT": 6,
        "PRIMARY_SHORT": 6,  # Alias for PUT
        "OPTION_CALL": 5,
        "OPTION_PUT": 3,
        "TACTICAL_PUT": 6,  # 5.5 rounded up
        "BULL_PROBE": 5,
        "BEAR_PROBE": 9,  # 8.5 rounded up
        "MVRV_SHORT": 10,
        "IRON_CONDOR": 7,
    }
    
    @classmethod
    def from_policy(cls, version: Optional[str] = None) -> "PolicyConfigAdapter":
        """Create adapter from policy version."""
        return cls(policy=get_policy(version))
    
    def get_checkpoint_day(self, signal_type: str) -> int:
        """
        Get checkpoint day for a signal type based on TTH p75.
        
        The checkpoint day is when we evaluate if a trade is adverse.
        Derived from TTH p75 values in the policy's DTE targets.
        
        Args:
            signal_type: Signal type (e.g., "CALL", "PUT", "MVRV_SHORT")
            
        Returns:
            Checkpoint day (1-indexed from trade entry)
        """
        return self.TTH_P75_MAP.get(signal_type, self.DEFAULT_CHECKPOINT_DAY)
    
    def get_adverse_threshold(
        self, 
        signal_type: str, 
        override: Optional[float] = None
    ) -> float:
        """
        Get adverse threshold for a signal type.
        
        Calculated as MAE(W) p75 / 2 from path_profiles.
        
        Args:
            signal_type: Signal type
            override: User-provided override value (takes precedence)
            
        Returns:
            Adverse threshold as decimal (e.g., 0.0235 for 2.35%)
        """
        if override is not None:
            return override
            
        profile = self.policy.get_path_profile(signal_type)
        mae_p75 = profile.get("mae_p75", self.DEFAULT_ADVERSE_THRESHOLD * 2)
        return mae_p75 / 2
    
    def get_direction_for_type(self, signal_type: str) -> str:
        """
        Get trade direction for a signal type.
        
        Args:
            signal_type: Signal type
            
        Returns:
            "LONG" or "SHORT"
        """
        long_types = {"CALL", "LONG", "OPTION_CALL", "BULL_PROBE"}
        return "LONG" if signal_type in long_types else "SHORT"
```

### 2. RecoveryCandidate (Data Model)

```python
from dataclasses import dataclass


@dataclass
class RecoveryCandidate:
    """
    A trade that qualifies for recovery analysis.
    
    A trade qualifies when it is adverse at the checkpoint day
    (price moved against the original direction by more than threshold).
    """
    date: str                    # Trade date (YYYY-MM-DD)
    trade_type: str              # Signal type (CALL, PUT, MVRV_SHORT, etc.)
    direction: str               # Original direction (LONG or SHORT)
    entry_price: float           # BTC price at entry
    checkpoint_day: int          # Day at which trade was evaluated
    checkpoint_price: float      # BTC price at checkpoint
    checkpoint_return: float     # Return at checkpoint (negative = adverse)
    remaining_days: int          # Days remaining after checkpoint
    forward_return: float        # Return from checkpoint to horizon end
    original_hit: bool           # Did original trade eventually hit target?
    recovery_hit: bool           # Would flipping have hit recovery target?
    recovery_return: float       # MFE in flipped direction from checkpoint
    
    @property
    def outcome_category(self) -> str:
        """Categorize this trade for the decision matrix."""
        if self.original_hit and self.recovery_hit:
            return "both_win"
        elif self.original_hit and not self.recovery_hit:
            return "hold_wins"
        elif not self.original_hit and self.recovery_hit:
            return "flip_wins"
        else:
            return "both_lose"
```

### 3. DecisionMatrix

```python
from dataclasses import dataclass, field
from typing import List


@dataclass
class DecisionMatrix:
    """
    Decision matrix for recovery analysis.
    
    Categorizes adverse trades into four outcomes:
    - both_win: Original eventually hits AND recovery would hit
    - hold_wins: Original eventually hits AND recovery would miss
    - flip_wins: Original misses AND recovery would hit
    - both_lose: Original misses AND recovery would miss
    """
    candidates: List[RecoveryCandidate] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return len(self.candidates)
    
    @property
    def both_win_count(self) -> int:
        return sum(1 for c in self.candidates if c.outcome_category == "both_win")
    
    @property
    def hold_wins_count(self) -> int:
        return sum(1 for c in self.candidates if c.outcome_category == "hold_wins")
    
    @property
    def flip_wins_count(self) -> int:
        return sum(1 for c in self.candidates if c.outcome_category == "flip_wins")
    
    @property
    def both_lose_count(self) -> int:
        return sum(1 for c in self.candidates if c.outcome_category == "both_lose")
    
    def get_percentage(self, category: str) -> float:
        """Get percentage for a category."""
        if self.total == 0:
            return 0.0
        count_map = {
            "both_win": self.both_win_count,
            "hold_wins": self.hold_wins_count,
            "flip_wins": self.flip_wins_count,
            "both_lose": self.both_lose_count,
        }
        return count_map.get(category, 0) / self.total * 100
    
    @property
    def net_edge(self) -> float:
        """
        Calculate net edge: flip_wins_pct - hold_wins_pct.
        
        Positive = flipping has advantage
        Negative = holding has advantage
        """
        return self.get_percentage("flip_wins") - self.get_percentage("hold_wins")
    
    @property
    def recommendation(self) -> str:
        """
        Get recommendation based on net edge.
        
        Returns:
            "flip" if net_edge > 5%
            "hold" if net_edge < -5%
            "neutral" otherwise
        """
        if self.net_edge > 5.0:
            return "flip"
        elif self.net_edge < -5.0:
            return "hold"
        else:
            return "neutral"
```

### 4. RecoveryAnalyzer

```python
from typing import List, Optional
import numpy as np
import pandas as pd


class RecoveryAnalyzer:
    """
    Analyzes trades for recovery potential.
    
    For each trade:
    1. Determines checkpoint day from policy
    2. Checks if trade is adverse at checkpoint
    3. If adverse, calculates recovery metrics
    """
    
    def __init__(
        self,
        policy_adapter: PolicyConfigAdapter,
        horizon: int = 14,
        original_target: float = 0.05,
        recovery_target: float = 0.03,
        adverse_threshold_override: Optional[float] = None,
    ):
        self.policy = policy_adapter
        self.horizon = horizon
        self.original_target = original_target
        self.recovery_target = recovery_target
        self.adverse_override = adverse_threshold_override
    
    def analyze(
        self,
        trades_df: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> List[RecoveryCandidate]:
        """
        Analyze trades for recovery potential.
        
        Args:
            trades_df: DataFrame with columns [date, type, direction]
            price_df: DataFrame with columns [btc_close, btc_high, btc_low]
                      indexed by date
        
        Returns:
            List of RecoveryCandidate for trades that are adverse at checkpoint
        """
        candidates = []
        
        for _, trade in trades_df.iterrows():
            candidate = self._analyze_single_trade(trade, price_df)
            if candidate is not None:
                candidates.append(candidate)
        
        return candidates
    
    def _analyze_single_trade(
        self,
        trade: pd.Series,
        price_df: pd.DataFrame,
    ) -> Optional[RecoveryCandidate]:
        """Analyze a single trade for recovery potential."""
        dt = pd.Timestamp(trade["date"])
        if dt not in price_df.index:
            return None
        
        entry_price = float(price_df.loc[dt, "btc_close"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            return None
        
        trade_type = trade["type"]
        direction = trade["direction"]
        is_long = direction == "LONG"
        
        # Get checkpoint day from policy
        checkpoint_day = self.policy.get_checkpoint_day(trade_type)
        if checkpoint_day >= self.horizon:
            checkpoint_day = self.horizon - 2
        
        # Get price path
        path = price_df.loc[dt:].iloc[1:self.horizon + 1]
        if len(path) < self.horizon:
            return None
        
        closes = path["btc_close"].to_numpy(dtype=float)
        highs = path["btc_high"].to_numpy(dtype=float)
        lows = path["btc_low"].to_numpy(dtype=float)
        
        if np.isnan(closes).any() or np.isnan(highs).any() or np.isnan(lows).any():
            return None
        
        # Check if original trade hit target
        if is_long:
            favorable = highs / entry_price - 1.0
        else:
            favorable = 1.0 - lows / entry_price
        original_hit = bool(np.any(favorable >= self.original_target))
        
        # Get checkpoint metrics
        checkpoint_price = closes[checkpoint_day - 1]
        if is_long:
            checkpoint_return = checkpoint_price / entry_price - 1.0
        else:
            checkpoint_return = 1.0 - checkpoint_price / entry_price
        
        # Get adverse threshold
        threshold = self.policy.get_adverse_threshold(
            trade_type, 
            self.adverse_override
        )
        
        # Check if adverse at checkpoint
        # checkpoint_return < -threshold means price moved against direction
        is_adverse = checkpoint_return < -threshold
        
        if not is_adverse:
            return None  # Not a recovery candidate
        
        # Calculate recovery metrics
        remaining_days = self.horizon - checkpoint_day
        remaining_closes = closes[checkpoint_day:]
        remaining_highs = highs[checkpoint_day:]
        remaining_lows = lows[checkpoint_day:]
        
        if len(remaining_closes) == 0:
            return None
        
        # Forward return in original direction
        forward_return = remaining_closes[-1] / checkpoint_price - 1.0
        if not is_long:
            forward_return = -forward_return
        
        # Recovery return: MFE in flipped direction
        if is_long:
            # Original was LONG, flip to SHORT
            recovery_favorable = 1.0 - remaining_lows / checkpoint_price
        else:
            # Original was SHORT, flip to LONG
            recovery_favorable = remaining_highs / checkpoint_price - 1.0
        
        recovery_hit = bool(np.any(recovery_favorable >= self.recovery_target))
        recovery_return = float(np.max(recovery_favorable))
        
        return RecoveryCandidate(
            date=dt.strftime("%Y-%m-%d"),
            trade_type=trade_type,
            direction=direction,
            entry_price=entry_price,
            checkpoint_day=checkpoint_day,
            checkpoint_price=checkpoint_price,
            checkpoint_return=checkpoint_return,
            remaining_days=remaining_days,
            forward_return=forward_return,
            original_hit=original_hit,
            recovery_hit=recovery_hit,
            recovery_return=recovery_return,
        )
```

### 5. TradeBuilder (Reused Logic)

The trade building logic is extracted from `analyze_hit_rate.py` and reused. The key method `_build_trades_df` handles:

- Loading ML models for fusion signal generation
- Applying fusion signal logic
- Applying overlay filtering (unless `--no-overlay`)
- Applying cooldown logic (unless `--no-cooldown`)
- Filtering by type and year

```python
def _build_trades_df(
    self,
    csv_path: Path,
    options: dict,
    year_filter: Optional[int],
    no_overlay: bool,
    no_cooldown: bool,
) -> pd.DataFrame:
    """
    Build trades DataFrame using same logic as analyze_hit_rate.py.
    
    This ensures consistency between analysis tools.
    """
    # Implementation mirrors analyze_loser_recovery._build_trades_df
    # See existing implementation for details
    ...
```

## Data Models

### Input Data

| Field | Type | Description |
|-------|------|-------------|
| CSV features file | DataFrame | Features with fusion signals, indexed by date |
| Price data | DataFrame | BTC OHLC from RawDailyData model |
| ML models | joblib | Long/short models for fusion signal generation |

### Output Data

| Field | Type | Description |
|-------|------|-------------|
| RecoveryCandidate | dataclass | Individual trade with recovery metrics |
| DecisionMatrix | dataclass | Aggregated outcomes for all candidates |
| Summary statistics | dict | Per-type and overall metrics |

## Interfaces

### Command-Line Interface

```
python manage.py analyze_recovery_policy [OPTIONS]

Options:
  --csv PATH              Input features CSV (default: features_14d_5pct.csv)
  --year INT              Filter to specific year
  --no-overlay            Disable overlay filtering
  --no-cooldown           Disable cooldown logic
  --long-model PATH       Path to long model (default: models/long_model.joblib)
  --short-model PATH      Path to short model (default: models/short_model.joblib)
  --type TYPE             Filter by trade type
  --horizon INT           Forward path horizon in days (default: 14)
  --target FLOAT          Original target threshold (default: 0.05)
  --adverse-threshold FLOAT  Override adverse threshold
  --recovery-target FLOAT Recovery target threshold (default: 0.03)
```

### PolicyVersion Interface (from policy.py)

```python
def get_path_profile(self, signal_type: str) -> dict:
    """
    Get path profile for a signal type.
    
    Returns dict with:
    - shakeout_pct: % of winners with shakeout path
    - invalidation_pct: % of winners invalidated before hit
    - mae_p75: 75th percentile MAE for winners
    - clean_win_pct: % of winners with clean paths
    """
```

## Error Handling

| Error Condition | Handling |
|-----------------|----------|
| CSV file not found | Display error message, exit gracefully |
| No trades match filters | Display informative message |
| Missing price data for trade date | Skip trade, continue analysis |
| Invalid price values (NaN, <= 0) | Skip trade, continue analysis |
| Unknown signal type | Use default checkpoint (7d) and threshold (2%) |

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Checkpoint Day Derivation from Policy

*For any* signal type with a defined TTH p75 value in the policy's path_profiles, the checkpoint day returned by `PolicyConfigAdapter.get_checkpoint_day()` SHALL equal the TTH p75 value (rounded up for fractional values). *For any* unknown signal type, the checkpoint day SHALL be the default value of 7.

**Validates: Requirements 1.1, 1.2, 1.3**

### Property 2: Adverse Threshold Calculation

*For any* signal type with a defined MAE(W) p75 value in the policy's path_profiles, the adverse threshold returned by `PolicyConfigAdapter.get_adverse_threshold()` SHALL equal MAE(W) p75 / 2. *For any* unknown signal type, the threshold SHALL be the default value of 0.02 (2%).

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 3: Direction-Specific Adverse Classification

*For any* LONG trade, the trade is classified as adverse if and only if `checkpoint_price < entry_price * (1 - threshold)`. *For any* SHORT trade, the trade is classified as adverse if and only if `checkpoint_price > entry_price * (1 + threshold)`.

**Validates: Requirements 7.1, 7.2, 7.3**

### Property 4: Recovery Return Calculation

*For any* recovery candidate, the `recovery_return` SHALL equal the maximum favorable excursion in the flipped direction from checkpoint to horizon end. For a LONG trade flipped to SHORT: `max(1 - lows[checkpoint:] / checkpoint_price)`. For a SHORT trade flipped to LONG: `max(highs[checkpoint:] / checkpoint_price - 1)`.

**Validates: Requirements 7.4**

### Property 5: Decision Matrix Categorization Completeness

*For any* set of recovery candidates, each candidate SHALL be classified into exactly one of the four outcome categories (both_win, hold_wins, flip_wins, both_lose), and the sum of counts across all categories SHALL equal the total number of candidates.

**Validates: Requirements 5.1, 5.2**

### Property 6: Net Edge Calculation

*For any* decision matrix, the net edge SHALL equal `flip_wins_percentage - hold_wins_percentage`, where each percentage is calculated as `count / total * 100`.

**Validates: Requirements 5.3**

### Property 7: Recommendation Based on Edge

*For any* decision matrix with net_edge > 5%, the recommendation SHALL be "flip". *For any* decision matrix with net_edge < -5%, the recommendation SHALL be "hold". *For any* decision matrix with -5% <= net_edge <= 5%, the recommendation SHALL be "neutral".

**Validates: Requirements 5.4, 5.5, 5.6**

### Property 8: IRON_CONDOR Exclusion

*For any* analysis output, no trade with type "IRON_CONDOR" SHALL be present in the recovery candidates list.

**Validates: Requirements 4.6**

### Property 9: Type Filter Correctness

*For any* analysis with a `--type` filter specified, all trades in the output SHALL have the specified trade type.

**Validates: Requirements 4.4**

### Property 10: Year Filter Correctness

*For any* analysis with a `--year` filter specified, all trades in the output SHALL have dates within the specified year.

**Validates: Requirements 4.5**

### Property 11: Trade List Sorting

*For any* detailed trade list output, the trades SHALL be sorted by `recovery_return` in descending order.

**Validates: Requirements 6.3**

### Property 12: User Override Takes Precedence

*For any* analysis where `--adverse-threshold` is provided, the adverse threshold used for all trades SHALL equal the user-provided value, regardless of the policy-derived value.

**Validates: Requirements 2.4**

## File Structure

```
signals/
└── management/
    └── commands/
        └── analyze_recovery_policy.py    # Main command implementation

execution/
└── services/
    └── policy.py                         # Existing policy module (read-only)
```

## Dependencies

| Dependency | Purpose |
|------------|---------|
| `execution.services.policy` | Policy configuration (TTH p75, MAE p75) |
| `datafeed.models.RawDailyData` | BTC price data |
| `signals.fusion` | Fusion signal generation |
| `signals.overlays` | Overlay filtering |
| `signals.tactical_puts` | Tactical put signal |
| `signals.mvrv_short` | MVRV short signal |
| `pandas` | Data manipulation |
| `numpy` | Numerical calculations |
| `joblib` | ML model loading |
