"""
Option Pricer Module

Provides multiple pricing modes for backtesting option strategies:
- linear: Legacy fixed leverage model (backward compatible)
- payoff_only: Path-exact payoff geometry (bounded P&L from structure)
- synthetic_iv: Black-Scholes with regime-mapped synthetic IV
- learned: ML-based leverage prediction from OptionSnapshot data

The learned mode supports two model types:
- bucket: Bucket-based quantile lookup (interpretable, robust with limited data)
- gbm: Gradient Boosting quantile regression (better generalization)

Usage:
    from execution.services.option_pricer import OptionPricer, PricingMode
    
    pricer = OptionPricer(mode=PricingMode.SYNTHETIC_IV)
    
    # Price at entry
    entry = pricer.price_at_entry(
        spot=67000, strike=68000, dte=14,
        option_type='call', iv_estimate=0.65
    )
    
    # Price at exit
    exit_result = pricer.price_at_exit(
        spot=69000, strike=68000, dte=7,
        option_type='call', iv_estimate=0.60,
        entry_price=entry['mid']
    )
    
    # Leverage prediction (learned mode - bucket)
    from execution.services.option_pricer import LeveragePredictor
    predictor = LeveragePredictor(model_path='models/option_response_predictor.json')
    
    # Leverage prediction (learned mode - GBM)
    from execution.services.option_pricer import GBMPredictor
    predictor = GBMPredictor(model_path='models/option_response_gbm.joblib')
    
    quantiles = predictor.predict(
        dte=14, moneyness=-0.05, iv=0.65, spread_pct=0.02,
        btc_change_pct=0.03, option_type='call'
    )
    # Returns: LeveragePrediction(p10=-0.15, p50=0.12, p90=0.45, ...)
"""
import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


class PricingMode(Enum):
    LINEAR = "linear"           # Legacy: fixed leverage * underlying move
    PAYOFF_ONLY = "payoff_only" # Path-exact payoff geometry
    SYNTHETIC_IV = "synthetic_iv"  # Black-Scholes with synthetic IV
    LEARNED = "learned"         # ML-based leverage prediction


@dataclass
class PricingConfig:
    """Configuration for option pricing."""
    # Synthetic IV parameters
    base_rv_window: int = 20  # Days for realized volatility
    regime_mult_range: Tuple[float, float] = (0.8, 1.4)  # IV/RV ratio by regime
    moneyness_adj_range: Tuple[float, float] = (0.85, 1.25)  # ATM vs OTM adjustment
    term_structure_adj_range: Tuple[float, float] = (0.9, 1.15)  # Short vs long DTE
    
    # Spread/slippage assumptions
    base_spread_pct: float = 0.02  # 2% bid-ask spread
    spread_dte_mult: Dict[str, float] = None  # DTE bucket -> spread multiplier
    spread_moneyness_mult: Dict[str, float] = None  # Moneyness bucket -> spread multiplier
    
    # Payoff structure parameters
    spread_width_pct: float = 0.10  # 10% spread width for debit spreads
    condor_wing_width_pct: float = 0.05  # 5% wing width for condors
    
    # Leverage predictor parameters
    predictor_model_path: Optional[Path] = None  # Path to saved predictor model
    min_training_samples: int = 100  # Minimum samples to train predictor
    
    def __post_init__(self):
        if self.spread_dte_mult is None:
            self.spread_dte_mult = {
                "7_10": 1.3,   # Wider spreads near expiry
                "11_14": 1.0,
                "15_21": 0.9,
            }
        if self.spread_moneyness_mult is None:
            self.spread_moneyness_mult = {
                "atm": 0.8,        # Tighter at ATM
                "slightly_itm": 1.0,
                "slightly_otm": 1.2,
                "deep_itm": 1.5,   # Wider deep ITM
                "deep_otm": 1.8,   # Widest deep OTM
            }


# =============================================================================
# LEVERAGE PREDICTOR - ML-based option return prediction
# =============================================================================

@dataclass
class LeveragePrediction:
    """Quantile predictions for option return."""
    p10: float  # 10th percentile (stressed)
    p25: float  # 25th percentile (conservative)
    p50: float  # 50th percentile (base)
    p75: float  # 75th percentile
    p90: float  # 90th percentile (optimistic)
    
    # Metadata
    dte_bucket: str
    moneyness_bucket: str
    regime: str
    n_samples: int  # Number of training samples in this bucket
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "p10": self.p10,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "dte_bucket": self.dte_bucket,
            "moneyness_bucket": self.moneyness_bucket,
            "regime": self.regime,
            "n_samples": self.n_samples,
        }


@dataclass
class TrainingRow:
    """Single training row for leverage predictor."""
    # Features at entry
    dte: float
    moneyness: float  # (strike - spot) / spot
    iv: float
    spread_pct: float
    option_type: str  # 'call' or 'put'
    
    # Path features
    btc_change_pct: float  # (spot_exit - spot_entry) / spot_entry
    iv_change: float  # iv_exit - iv_entry
    days_held: int
    
    # Target
    option_return_pct: float  # (option_exit - option_entry) / option_entry


class LeveragePredictor:
    """
    ML-based leverage/return predictor trained on OptionSnapshot data.
    
    Predicts quantiles of option return given:
    - Entry conditions (DTE, moneyness, IV, spread)
    - Path conditions (BTC change, IV change, time held)
    
    Uses bucket-based quantile estimation for interpretability and
    robustness with limited data.
    """
    
    # Bucket definitions
    DTE_BUCKETS = {
        "0_3": (0, 3),
        "4_7": (4, 7),
        "7_10": (7, 10),
        "11_14": (11, 14),
        "15_21": (15, 21),
        "22_30": (22, 30),
        "30_plus": (30, 999),
    }
    
    MONEYNESS_BUCKETS = {
        "deep_itm": (-999, -0.10),
        "itm": (-0.10, -0.03),
        "atm": (-0.03, 0.03),
        "otm": (0.03, 0.10),
        "deep_otm": (0.10, 999),
    }
    
    # V2: Tighter IV buckets for better differentiation
    IV_BUCKETS = {
        "very_low": (0, 0.35),
        "low": (0.35, 0.45),
        "normal": (0.45, 0.60),
        "elevated": (0.60, 0.80),
        "high": (0.80, 999),
    }
    
    SPREAD_BUCKETS = {
        "tight": (0, 0.02),
        "normal": (0.02, 0.05),
        "wide": (0.05, 0.10),
        "illiquid": (0.10, 999),
    }
    
    # V2: Tighter BTC move buckets for better granularity
    BTC_MOVE_BUCKETS = {
        "large_down": (-999, -0.05),
        "mod_down": (-0.05, -0.03),
        "small_down": (-0.03, -0.01),
        "flat": (-0.01, 0.01),
        "small_up": (0.01, 0.03),
        "mod_up": (0.03, 0.05),
        "large_up": (0.05, 999),
    }
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to saved model (optional, will train if not provided)
        """
        self.model_path = model_path
        self.bucket_stats: Dict[str, Dict[str, Any]] = {}
        self.global_stats: Dict[str, float] = {}
        self.is_fitted = False
        self._training_data: List[TrainingRow] = []
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def _get_bucket(self, value: float, buckets: Dict[str, Tuple[float, float]]) -> str:
        """Get bucket name for a value."""
        for name, (lo, hi) in buckets.items():
            if lo <= value < hi:
                return name
        # Fallback to last bucket
        return list(buckets.keys())[-1]
    
    def _get_dte_bucket(self, dte: float) -> str:
        return self._get_bucket(dte, self.DTE_BUCKETS)
    
    def _get_moneyness_bucket(self, moneyness: float) -> str:
        return self._get_bucket(moneyness, self.MONEYNESS_BUCKETS)
    
    def _get_iv_bucket(self, iv: float) -> str:
        return self._get_bucket(iv, self.IV_BUCKETS)
    
    def _get_spread_bucket(self, spread_pct: float) -> str:
        return self._get_bucket(spread_pct, self.SPREAD_BUCKETS)
    
    def _get_btc_move_bucket(self, btc_change_pct: float) -> str:
        return self._get_bucket(btc_change_pct, self.BTC_MOVE_BUCKETS)
    
    def _make_bucket_key(
        self,
        dte_bucket: str,
        moneyness_bucket: str,
        option_type: str,
        btc_move_bucket: str,
        iv_bucket: str = None,
    ) -> str:
        """Create composite bucket key. V2 includes IV bucket."""
        if iv_bucket:
            return f"{dte_bucket}|{moneyness_bucket}|{option_type}|{btc_move_bucket}|{iv_bucket}"
        return f"{dte_bucket}|{moneyness_bucket}|{option_type}|{btc_move_bucket}"
    
    def add_training_row(self, row: TrainingRow) -> None:
        """Add a training row."""
        self._training_data.append(row)
    
    def build_training_data_from_snapshots(
        self,
        horizon_days: int = 1,
        min_snapshots_per_symbol: int = 2,
    ) -> int:
        """
        Build training data from OptionSnapshot pairs.
        
        For each option, pairs entry snapshot at time t with exit snapshot
        at time t+horizon to compute actual option returns.
        
        Args:
            horizon_days: Days between entry and exit snapshots
            min_snapshots_per_symbol: Minimum snapshots needed per symbol
        
        Returns:
            Number of training rows created
        """
        try:
            from datafeed.models import OptionSnapshot
        except ImportError:
            logger.warning("Cannot import OptionSnapshot - Django not configured")
            return 0
        
        from django.db.models import Count
        from datetime import timedelta
        
        # Find symbols with enough snapshots
        symbol_counts = (
            OptionSnapshot.objects
            .values('symbol')
            .annotate(count=Count('id'))
            .filter(count__gte=min_snapshots_per_symbol)
        )
        
        symbols = [s['symbol'] for s in symbol_counts]
        logger.info(f"Found {len(symbols)} symbols with >= {min_snapshots_per_symbol} snapshots")
        
        rows_created = 0
        
        for symbol in symbols:
            snapshots = list(
                OptionSnapshot.objects
                .filter(symbol=symbol)
                .order_by('timestamp')
                .values(
                    'timestamp', 'spot_price', 'strike', 'option_type',
                    'mid_price', 'iv', 'spread_pct', 'dte'
                )
            )
            
            if len(snapshots) < 2:
                continue
            
            # Pair snapshots by horizon
            for i, entry in enumerate(snapshots[:-1]):
                entry_ts = entry['timestamp']
                target_ts = entry_ts + timedelta(days=horizon_days)
                
                # Find closest exit snapshot
                exit_snap = None
                min_diff = timedelta(days=999)
                
                for j in range(i + 1, len(snapshots)):
                    diff = abs(snapshots[j]['timestamp'] - target_ts)
                    if diff < min_diff:
                        min_diff = diff
                        exit_snap = snapshots[j]
                    if snapshots[j]['timestamp'] > target_ts + timedelta(days=1):
                        break
                
                if exit_snap is None or min_diff > timedelta(days=2):
                    continue
                
                # Compute features and target
                entry_price = float(entry['mid_price'] or 0)
                exit_price = float(exit_snap['mid_price'] or 0)
                
                if entry_price <= 0 or exit_price <= 0:
                    continue
                
                entry_spot = float(entry['spot_price'] or 0)
                exit_spot = float(exit_snap['spot_price'] or 0)
                
                if entry_spot <= 0 or exit_spot <= 0:
                    continue
                
                btc_change_pct = (exit_spot - entry_spot) / entry_spot
                option_return_pct = (exit_price - entry_price) / entry_price
                
                entry_iv = float(entry['iv'] or 0.6)
                exit_iv = float(exit_snap['iv'] or entry_iv)
                iv_change = exit_iv - entry_iv
                
                days_held = (exit_snap['timestamp'] - entry_ts).days
                
                row = TrainingRow(
                    dte=float(entry['dte'] or 14),
                    moneyness=float((entry['strike'] - entry_spot) / entry_spot),
                    iv=entry_iv,
                    spread_pct=float(entry['spread_pct'] or 0.02),
                    option_type=entry['option_type'],
                    btc_change_pct=btc_change_pct,
                    iv_change=iv_change,
                    days_held=days_held,
                    option_return_pct=option_return_pct,
                )
                
                self.add_training_row(row)
                rows_created += 1
        
        logger.info(f"Created {rows_created} training rows from snapshots")
        return rows_created
    
    def fit(self, min_samples_per_bucket: int = 5, include_iv: bool = True) -> None:
        """
        Fit the predictor on accumulated training data.
        
        Computes quantile statistics for each bucket combination.
        
        Args:
            min_samples_per_bucket: Minimum samples to compute bucket stats
            include_iv: Whether to include IV in bucket key (V2 feature)
        """
        if not self._training_data:
            logger.warning("No training data - predictor will use fallback estimates")
            self.is_fitted = True
            return
        
        # Group by bucket - V2: include IV bucket
        bucket_data: Dict[str, List[float]] = {}
        
        for row in self._training_data:
            dte_bucket = self._get_dte_bucket(row.dte)
            moneyness_bucket = self._get_moneyness_bucket(row.moneyness)
            btc_move_bucket = self._get_btc_move_bucket(row.btc_change_pct)
            iv_bucket = self._get_iv_bucket(row.iv) if include_iv else None
            
            key = self._make_bucket_key(
                dte_bucket, moneyness_bucket, row.option_type, btc_move_bucket, iv_bucket
            )
            
            if key not in bucket_data:
                bucket_data[key] = []
            bucket_data[key].append(row.option_return_pct)
        
        # Compute quantiles for each bucket
        for key, returns in bucket_data.items():
            if len(returns) < min_samples_per_bucket:
                continue
            
            arr = np.array(returns)
            self.bucket_stats[key] = {
                "p10": float(np.percentile(arr, 10)),
                "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "p90": float(np.percentile(arr, 90)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n_samples": len(returns),
            }
        
        # Compute global stats as fallback
        all_returns = [r.option_return_pct for r in self._training_data]
        arr = np.array(all_returns)
        self.global_stats = {
            "p10": float(np.percentile(arr, 10)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n_samples": len(all_returns),
        }
        
        self.is_fitted = True
        logger.info(
            f"Fitted predictor with {len(self.bucket_stats)} buckets, "
            f"{len(self._training_data)} total samples"
        )
    
    def fit_from_snapshots(
        self,
        horizon_days: int = 1,
        min_samples_per_bucket: int = 5,
    ) -> None:
        """
        Convenience method to build training data and fit in one call.
        
        Args:
            horizon_days: Days between entry and exit snapshots
            min_samples_per_bucket: Minimum samples per bucket
        """
        self.build_training_data_from_snapshots(horizon_days=horizon_days)
        self.fit(min_samples_per_bucket=min_samples_per_bucket)
    
    def predict(
        self,
        dte: float,
        moneyness: float,
        option_type: str,
        btc_change_pct: float,
        iv: Optional[float] = None,
        spread_pct: Optional[float] = None,
        iv_change: Optional[float] = None,
        days_held: Optional[int] = None,
    ) -> LeveragePrediction:
        """
        Predict option return quantiles.
        
        Args:
            dte: Days to expiry at entry
            moneyness: (strike - spot) / spot
            option_type: 'call' or 'put'
            btc_change_pct: Underlying price change
            iv: Implied volatility (optional, for regime detection)
            spread_pct: Bid-ask spread % (optional)
            iv_change: IV change from entry (optional)
            days_held: Days position held (optional)
        
        Returns:
            LeveragePrediction with quantile estimates
        """
        dte_bucket = self._get_dte_bucket(dte)
        moneyness_bucket = self._get_moneyness_bucket(moneyness)
        btc_move_bucket = self._get_btc_move_bucket(btc_change_pct)
        iv_bucket = self._get_iv_bucket(iv) if iv else "normal"
        
        # V2: Try with IV bucket first (most specific)
        key_with_iv = self._make_bucket_key(
            dte_bucket, moneyness_bucket, option_type, btc_move_bucket, iv_bucket
        )
        
        if key_with_iv in self.bucket_stats:
            stats = self.bucket_stats[key_with_iv]
            return LeveragePrediction(
                p10=stats["p10"],
                p25=stats["p25"],
                p50=stats["p50"],
                p75=stats["p75"],
                p90=stats["p90"],
                dte_bucket=dte_bucket,
                moneyness_bucket=moneyness_bucket,
                regime=iv_bucket,
                n_samples=stats["n_samples"],
            )
        
        # Fallback: Try without IV bucket
        key = self._make_bucket_key(
            dte_bucket, moneyness_bucket, option_type, btc_move_bucket
        )
        
        # Try exact bucket match
        if key in self.bucket_stats:
            stats = self.bucket_stats[key]
            return LeveragePrediction(
                p10=stats["p10"],
                p25=stats["p25"],
                p50=stats["p50"],
                p75=stats["p75"],
                p90=stats["p90"],
                dte_bucket=dte_bucket,
                moneyness_bucket=moneyness_bucket,
                regime=iv_bucket,
                n_samples=stats["n_samples"],
            )
        
        # Try partial matches (relax btc_move)
        for btc_bucket in self.BTC_MOVE_BUCKETS.keys():
            partial_key = self._make_bucket_key(
                dte_bucket, moneyness_bucket, option_type, btc_bucket
            )
            if partial_key in self.bucket_stats:
                stats = self.bucket_stats[partial_key]
                # Adjust for actual BTC move direction
                adjustment = self._compute_btc_adjustment(
                    btc_change_pct, btc_bucket, option_type
                )
                return LeveragePrediction(
                    p10=stats["p10"] + adjustment,
                    p25=stats["p25"] + adjustment,
                    p50=stats["p50"] + adjustment,
                    p75=stats["p75"] + adjustment,
                    p90=stats["p90"] + adjustment,
                    dte_bucket=dte_bucket,
                    moneyness_bucket=moneyness_bucket,
                    regime=iv_bucket,
                    n_samples=stats["n_samples"],
                )
        
        # Fall back to global stats or synthetic estimate
        if self.global_stats:
            return LeveragePrediction(
                p10=self.global_stats["p10"],
                p25=self.global_stats["p25"],
                p50=self.global_stats["p50"],
                p75=self.global_stats["p75"],
                p90=self.global_stats["p90"],
                dte_bucket=dte_bucket,
                moneyness_bucket=moneyness_bucket,
                regime=iv_bucket,
                n_samples=self.global_stats.get("n_samples", 0),
            )
        
        # Ultimate fallback: synthetic estimate based on delta approximation
        return self._synthetic_estimate(
            dte, moneyness, option_type, btc_change_pct, iv or 0.6
        )
    
    def _compute_btc_adjustment(
        self,
        actual_btc_change: float,
        bucket_btc_change: str,
        option_type: str,
    ) -> float:
        """
        Compute adjustment for BTC move difference.
        
        Uses approximate delta to estimate impact.
        """
        # Get bucket midpoint
        bucket_ranges = self.BTC_MOVE_BUCKETS[bucket_btc_change]
        bucket_mid = (bucket_ranges[0] + bucket_ranges[1]) / 2
        bucket_mid = max(-0.10, min(0.10, bucket_mid))  # Clamp
        
        diff = actual_btc_change - bucket_mid
        
        # Approximate delta effect (calls gain on up, puts gain on down)
        if option_type == 'call':
            return diff * 5  # ~5x leverage approximation
        else:
            return -diff * 5
    
    def _synthetic_estimate(
        self,
        dte: float,
        moneyness: float,
        option_type: str,
        btc_change_pct: float,
        iv: float,
    ) -> LeveragePrediction:
        """
        Generate synthetic estimate when no data available.
        
        Uses Black-Scholes delta approximation.
        """
        # Estimate delta based on moneyness
        if abs(moneyness) < 0.03:
            delta = 0.50
        elif moneyness < -0.03:  # ITM
            delta = 0.60 + min(abs(moneyness) * 2, 0.35)
        else:  # OTM
            delta = 0.40 - min(moneyness * 2, 0.30)
        
        if option_type == 'put':
            delta = -delta
        
        # Estimate leverage from delta
        leverage = abs(delta) * 10  # Rough approximation
        
        # Base return estimate
        if option_type == 'call':
            base_return = btc_change_pct * leverage
        else:
            base_return = -btc_change_pct * leverage
        
        # Add theta decay estimate
        theta_decay = 0.02 * (1 if dte <= 7 else 0.5)  # Higher decay near expiry
        base_return -= theta_decay
        
        # Generate quantiles with uncertainty
        std = abs(base_return) * 0.5 + 0.10  # Higher uncertainty for larger moves
        
        return LeveragePrediction(
            p10=base_return - 1.28 * std,
            p25=base_return - 0.67 * std,
            p50=base_return,
            p75=base_return + 0.67 * std,
            p90=base_return + 1.28 * std,
            dte_bucket=self._get_dte_bucket(dte),
            moneyness_bucket=self._get_moneyness_bucket(moneyness),
            regime="synthetic",
            n_samples=0,
        )
    
    def save(self, path: Path) -> None:
        """Save predictor to disk."""
        import json
        
        data = {
            "bucket_stats": self.bucket_stats,
            "global_stats": self.global_stats,
            "is_fitted": self.is_fitted,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved predictor to {path}")
    
    def load(self, path: Path) -> None:
        """Load predictor from disk."""
        import json
        
        path = Path(path)
        if not path.exists():
            logger.warning(f"Predictor file not found: {path}")
            return
        
        data = json.loads(path.read_text())
        self.bucket_stats = data.get("bucket_stats", {})
        self.global_stats = data.get("global_stats", {})
        self.is_fitted = data.get("is_fitted", False)
        logger.info(f"Loaded predictor from {path} with {len(self.bucket_stats)} buckets")


# =============================================================================
# GBM PREDICTOR - Gradient Boosting Quantile Regression
# =============================================================================

class GBMPredictor:
    """
    Gradient Boosting quantile regression predictor for option returns.
    
    Trains separate GBM models for each quantile (p10, p25, p50, p75, p90)
    using sklearn's GradientBoostingRegressor with quantile loss.
    
    Features:
    - dte: Days to expiry
    - moneyness: (strike - spot) / spot
    - iv: Implied volatility
    - spread_pct: Bid-ask spread percentage
    - is_call: 1 for call, 0 for put
    - btc_change_pct: Underlying price change
    - iv_change: IV change from entry
    - days_held: Days position held
    
    Usage:
        predictor = GBMPredictor()
        predictor.fit(training_rows)
        predictor.save('models/option_response_gbm.joblib')
        
        # Later:
        predictor = GBMPredictor(model_path='models/option_response_gbm.joblib')
        pred = predictor.predict(dte=14, moneyness=-0.05, ...)
    """
    
    FEATURE_NAMES = [
        'dte', 'moneyness', 'iv', 'spread_pct', 'is_call',
        'btc_change_pct', 'iv_change', 'days_held'
    ]
    
    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize GBM predictor.
        
        Args:
            model_path: Path to saved .joblib model (optional)
        """
        self.model_path = model_path
        self.models: Dict[float, Any] = {}  # quantile -> fitted model
        self.is_fitted = False
        self.n_samples = 0
        self._training_data: List[TrainingRow] = []
        
        if model_path and Path(model_path).exists():
            self.load(model_path)
    
    def add_training_row(self, row: TrainingRow) -> None:
        """Add a training row."""
        self._training_data.append(row)
    
    def _row_to_features(self, row: TrainingRow) -> List[float]:
        """Convert TrainingRow to feature vector."""
        return [
            row.dte,
            row.moneyness,
            row.iv,
            row.spread_pct,
            1.0 if row.option_type == 'call' else 0.0,
            row.btc_change_pct,
            row.iv_change,
            float(row.days_held),
        ]
    
    def _make_features(
        self,
        dte: float,
        moneyness: float,
        option_type: str,
        btc_change_pct: float,
        iv: float = 0.50,
        spread_pct: float = 0.03,
        iv_change: float = 0.0,
        days_held: int = 1,
    ) -> np.ndarray:
        """Create feature array for prediction."""
        return np.array([[
            dte,
            moneyness,
            iv,
            spread_pct,
            1.0 if option_type == 'call' else 0.0,
            btc_change_pct,
            iv_change,
            float(days_held),
        ]])
    
    def fit(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        subsample: float = 0.8,
    ) -> Dict[str, float]:
        """
        Fit GBM models for each quantile.
        
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum depth of trees
            learning_rate: Learning rate shrinks contribution of each tree
            min_samples_leaf: Minimum samples required at leaf node
            subsample: Fraction of samples for fitting each tree
        
        Returns:
            Dict with training metrics
        """
        from sklearn.ensemble import GradientBoostingRegressor
        
        if not self._training_data:
            logger.warning("No training data - cannot fit GBM predictor")
            return {}
        
        # Build feature matrix and target
        X = np.array([self._row_to_features(row) for row in self._training_data])
        y = np.array([row.option_return_pct for row in self._training_data])
        
        self.n_samples = len(y)
        logger.info(f"Fitting GBM predictor on {self.n_samples} samples")
        
        metrics = {}
        
        for q in self.QUANTILES:
            logger.info(f"  Training quantile {q:.2f}...")
            
            model = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_samples_leaf=min_samples_leaf,
                subsample=subsample,
                random_state=42,
            )
            
            model.fit(X, y)
            self.models[q] = model
            
            # Compute training pinball loss
            y_pred = model.predict(X)
            pinball = self._pinball_loss(y, y_pred, q)
            metrics[f'pinball_p{int(q*100)}'] = pinball
        
        # Compute coverage (% of actuals within p10-p90 band)
        p10_pred = self.models[0.10].predict(X)
        p90_pred = self.models[0.90].predict(X)
        coverage = np.mean((y >= p10_pred) & (y <= p90_pred))
        metrics['coverage_10_90'] = coverage
        
        # MAE of median prediction
        p50_pred = self.models[0.50].predict(X)
        metrics['mae_p50'] = float(np.mean(np.abs(y - p50_pred)))
        
        self.is_fitted = True
        logger.info(f"GBM predictor fitted: MAE={metrics['mae_p50']:.4f}, Coverage={coverage:.2%}")
        
        return metrics
    
    def _pinball_loss(self, y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
        """Compute pinball (quantile) loss."""
        err = y_true - y_pred
        return float(np.mean(np.maximum(q * err, (q - 1) * err)))
    
    def predict(
        self,
        dte: float,
        moneyness: float,
        option_type: str,
        btc_change_pct: float,
        iv: float = 0.50,
        spread_pct: float = 0.03,
        iv_change: float = 0.0,
        days_held: int = 1,
    ) -> LeveragePrediction:
        """
        Predict option return quantiles.
        
        Args:
            dte: Days to expiry at entry
            moneyness: (strike - spot) / spot
            option_type: 'call' or 'put'
            btc_change_pct: Underlying price change
            iv: Implied volatility
            spread_pct: Bid-ask spread %
            iv_change: IV change from entry
            days_held: Days position held
        
        Returns:
            LeveragePrediction with quantile estimates
        """
        if not self.is_fitted:
            # Fall back to synthetic estimate
            return self._synthetic_estimate(dte, moneyness, option_type, btc_change_pct, iv)
        
        X = self._make_features(
            dte, moneyness, option_type, btc_change_pct,
            iv, spread_pct, iv_change, days_held
        )
        
        predictions = {q: float(self.models[q].predict(X)[0]) for q in self.QUANTILES}
        
        # Determine regime label from IV
        if iv < 0.35:
            regime = "very_low"
        elif iv < 0.45:
            regime = "low"
        elif iv < 0.60:
            regime = "normal"
        elif iv < 0.80:
            regime = "elevated"
        else:
            regime = "high"
        
        # Determine DTE bucket
        if dte <= 3:
            dte_bucket = "0_3"
        elif dte <= 7:
            dte_bucket = "4_7"
        elif dte <= 10:
            dte_bucket = "7_10"
        elif dte <= 14:
            dte_bucket = "11_14"
        elif dte <= 21:
            dte_bucket = "15_21"
        elif dte <= 30:
            dte_bucket = "22_30"
        else:
            dte_bucket = "30_plus"
        
        # Determine moneyness bucket
        if moneyness < -0.10:
            moneyness_bucket = "deep_itm"
        elif moneyness < -0.03:
            moneyness_bucket = "itm"
        elif moneyness < 0.03:
            moneyness_bucket = "atm"
        elif moneyness < 0.10:
            moneyness_bucket = "otm"
        else:
            moneyness_bucket = "deep_otm"
        
        return LeveragePrediction(
            p10=predictions[0.10],
            p25=predictions[0.25],
            p50=predictions[0.50],
            p75=predictions[0.75],
            p90=predictions[0.90],
            dte_bucket=dte_bucket,
            moneyness_bucket=moneyness_bucket,
            regime=regime,
            n_samples=self.n_samples,
        )
    
    def _synthetic_estimate(
        self,
        dte: float,
        moneyness: float,
        option_type: str,
        btc_change_pct: float,
        iv: float,
    ) -> LeveragePrediction:
        """Generate synthetic estimate when model not fitted."""
        # Simple delta-based approximation
        if abs(moneyness) < 0.03:
            delta = 0.50
        elif moneyness < -0.03:
            delta = 0.60 + min(abs(moneyness) * 2, 0.35)
        else:
            delta = 0.40 - min(moneyness * 2, 0.30)
        
        if option_type == 'put':
            delta = -delta
        
        leverage = abs(delta) * 10
        
        if option_type == 'call':
            base_return = btc_change_pct * leverage
        else:
            base_return = -btc_change_pct * leverage
        
        theta_decay = 0.02 * (1 if dte <= 7 else 0.5)
        base_return -= theta_decay
        
        std = abs(base_return) * 0.5 + 0.10
        
        return LeveragePrediction(
            p10=base_return - 1.28 * std,
            p25=base_return - 0.67 * std,
            p50=base_return,
            p75=base_return + 0.67 * std,
            p90=base_return + 1.28 * std,
            dte_bucket="unknown",
            moneyness_bucket="unknown",
            regime="synthetic",
            n_samples=0,
        )
    
    def save(self, path: Path) -> None:
        """Save predictor to disk using joblib."""
        import joblib
        
        data = {
            'models': self.models,
            'is_fitted': self.is_fitted,
            'n_samples': self.n_samples,
            'feature_names': self.FEATURE_NAMES,
            'quantiles': self.QUANTILES,
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, path)
        logger.info(f"Saved GBM predictor to {path}")
    
    def load(self, path: Path) -> None:
        """Load predictor from disk."""
        import joblib
        
        path = Path(path)
        if not path.exists():
            logger.warning(f"GBM predictor file not found: {path}")
            return
        
        data = joblib.load(path)
        self.models = data.get('models', {})
        self.is_fitted = data.get('is_fitted', False)
        self.n_samples = data.get('n_samples', 0)
        logger.info(f"Loaded GBM predictor from {path} with {len(self.models)} quantile models, {self.n_samples} training samples")
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for each quantile model."""
        if not self.is_fitted:
            return {}
        
        importance = {}
        for q, model in self.models.items():
            importance[f'p{int(q*100)}'] = dict(zip(
                self.FEATURE_NAMES,
                model.feature_importances_.tolist()
            ))
        return importance


class BlackScholes:
    """Black-Scholes option pricing utilities."""
    
    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    
    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * math.sqrt(T)
    
    @staticmethod
    def call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate call option price."""
        if T <= 0:
            return max(S - K, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    
    @staticmethod
    def put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate put option price."""
        if T <= 0:
            return max(K - S, 0)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option delta."""
        if T <= 0:
            if option_type == 'call':
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate option theta (per day)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
        
        if option_type == 'call':
            term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
        
        return (term1 + term2) / 365  # Per day
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega (per 1% IV change)."""
        if T <= 0:
            return 0.0
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * math.sqrt(T) / 100


class OptionPricer:
    """
    Multi-mode option pricer for backtesting.
    
    Modes:
    - LINEAR: Legacy fixed leverage model
    - PAYOFF_ONLY: Bounded payoff geometry (no IV modeling)
    - SYNTHETIC_IV: Full Black-Scholes with regime-mapped IV
    """
    
    def __init__(
        self,
        mode: PricingMode = PricingMode.PAYOFF_ONLY,
        config: Optional[PricingConfig] = None,
        risk_free_rate: float = 0.05,
    ):
        self.mode = mode
        self.config = config or PricingConfig()
        self.r = risk_free_rate
    
    def estimate_synthetic_iv(
        self,
        realized_vol: float,
        dte: float,
        moneyness: float,
        regime: str = "neutral",
    ) -> float:
        """
        Estimate synthetic IV from realized volatility and market conditions.
        
        IV_syn = RV_20d * regime_mult * moneyness_adj * term_structure_adj
        
        Args:
            realized_vol: 20-day realized volatility (annualized)
            dte: Days to expiry
            moneyness: (strike - spot) / spot
            regime: Market regime (trending, choppy, neutral)
        
        Returns:
            Estimated implied volatility
        """
        # Regime multiplier: IV tends to be higher in choppy/uncertain regimes
        regime_mult = {
            "trending": 0.85,
            "neutral": 1.0,
            "choppy": 1.2,
            "stressed": 1.4,
        }.get(regime, 1.0)
        
        # Moneyness adjustment: OTM options have higher IV (smile)
        abs_moneyness = abs(moneyness)
        if abs_moneyness < 0.02:  # ATM
            moneyness_adj = 1.0
        elif abs_moneyness < 0.05:  # Slightly OTM/ITM
            moneyness_adj = 1.05
        elif abs_moneyness < 0.10:  # Moderate OTM/ITM
            moneyness_adj = 1.10
        else:  # Deep OTM/ITM
            moneyness_adj = 1.15 + (abs_moneyness - 0.10) * 0.5
        
        # Term structure adjustment: Short-dated options often have higher IV
        if dte <= 7:
            term_adj = 1.15
        elif dte <= 14:
            term_adj = 1.05
        elif dte <= 21:
            term_adj = 1.0
        else:
            term_adj = 0.95
        
        iv = realized_vol * regime_mult * moneyness_adj * term_adj
        
        # Clamp to reasonable range
        return max(0.20, min(2.0, iv))
    
    def estimate_spread(
        self,
        dte: float,
        moneyness: float,
        mid_price: float,
    ) -> Tuple[float, float]:
        """
        Estimate bid-ask spread based on DTE and moneyness.
        
        Returns:
            (bid, ask) prices
        """
        # Get DTE bucket
        if dte <= 10:
            dte_bucket = "7_10"
        elif dte <= 14:
            dte_bucket = "11_14"
        else:
            dte_bucket = "15_21"
        
        # Get moneyness bucket
        abs_m = abs(moneyness)
        if abs_m < 0.02:
            m_bucket = "atm"
        elif abs_m < 0.05:
            m_bucket = "slightly_itm" if moneyness < 0 else "slightly_otm"
        else:
            m_bucket = "deep_itm" if moneyness < 0 else "deep_otm"
        
        # Calculate spread
        spread_pct = (
            self.config.base_spread_pct
            * self.config.spread_dte_mult.get(dte_bucket, 1.0)
            * self.config.spread_moneyness_mult.get(m_bucket, 1.0)
        )
        
        half_spread = mid_price * spread_pct / 2
        return mid_price - half_spread, mid_price + half_spread
    
    def price_at_entry(
        self,
        spot: float,
        strike: float,
        dte: float,
        option_type: str,
        iv_estimate: Optional[float] = None,
        realized_vol: Optional[float] = None,
        regime: str = "neutral",
    ) -> Dict[str, Any]:
        """
        Price option at entry.
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            dte: Days to expiry
            option_type: 'call' or 'put'
            iv_estimate: Explicit IV (if known)
            realized_vol: Realized volatility for synthetic IV
            regime: Market regime for synthetic IV
        
        Returns:
            Dict with mid, bid, ask, greeks
        """
        T = dte / 365
        moneyness = (strike - spot) / spot
        
        if self.mode == PricingMode.LINEAR:
            # Legacy mode: return placeholder values
            return {
                "mid": None,
                "bid": None,
                "ask": None,
                "delta": None,
                "gamma": None,
                "theta": None,
                "vega": None,
                "iv": None,
                "moneyness": moneyness,
                "dte": dte,
            }
        
        # Determine IV
        if iv_estimate is not None:
            iv = iv_estimate
        elif realized_vol is not None:
            iv = self.estimate_synthetic_iv(realized_vol, dte, moneyness, regime)
        else:
            # Default fallback: 60% annualized
            iv = 0.60
        
        # Calculate price
        if option_type == 'call':
            mid = BlackScholes.call_price(spot, strike, T, self.r, iv)
        else:
            mid = BlackScholes.put_price(spot, strike, T, self.r, iv)
        
        # Calculate greeks
        delta = BlackScholes.delta(spot, strike, T, self.r, iv, option_type)
        gamma = BlackScholes.gamma(spot, strike, T, self.r, iv)
        theta = BlackScholes.theta(spot, strike, T, self.r, iv, option_type)
        vega = BlackScholes.vega(spot, strike, T, self.r, iv)
        
        # Estimate spread
        bid, ask = self.estimate_spread(dte, moneyness, mid)
        
        return {
            "mid": mid,
            "bid": bid,
            "ask": ask,
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "iv": iv,
            "moneyness": moneyness,
            "dte": dte,
        }
    
    def price_at_exit(
        self,
        spot: float,
        strike: float,
        dte: float,
        option_type: str,
        entry_price: float,
        iv_estimate: Optional[float] = None,
        realized_vol: Optional[float] = None,
        regime: str = "neutral",
        iv_change: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Price option at exit and compute P&L.
        
        Args:
            spot: Current underlying price
            strike: Option strike price
            dte: Days to expiry remaining
            option_type: 'call' or 'put'
            entry_price: Price paid at entry
            iv_estimate: Explicit IV (if known)
            realized_vol: Realized volatility for synthetic IV
            regime: Market regime for synthetic IV
            iv_change: IV shift from entry (for scenario analysis)
        
        Returns:
            Dict with mid, pnl, pnl_pct, greeks
        """
        result = self.price_at_entry(
            spot, strike, dte, option_type,
            iv_estimate=iv_estimate,
            realized_vol=realized_vol,
            regime=regime,
        )
        
        # Apply IV change for scenario analysis
        if iv_change != 0 and result["iv"] is not None:
            adjusted_iv = result["iv"] + iv_change
            T = dte / 365
            if option_type == 'call':
                result["mid"] = BlackScholes.call_price(spot, strike, T, self.r, adjusted_iv)
            else:
                result["mid"] = BlackScholes.put_price(spot, strike, T, self.r, adjusted_iv)
            result["iv"] = adjusted_iv
        
        # Compute P&L
        if result["mid"] is not None and entry_price is not None:
            result["pnl"] = result["mid"] - entry_price
            result["pnl_pct"] = result["pnl"] / entry_price if entry_price > 0 else 0
        else:
            result["pnl"] = None
            result["pnl_pct"] = None
        
        result["entry_price"] = entry_price
        return result
    
    def payoff_at_expiry(
        self,
        spot: float,
        strike: float,
        option_type: str,
        entry_price: float,
        direction: str = "long",
    ) -> Dict[str, float]:
        """
        Calculate payoff at expiry (intrinsic value only).
        
        Args:
            spot: Underlying price at expiry
            strike: Option strike
            option_type: 'call' or 'put'
            entry_price: Price paid at entry
            direction: 'long' or 'short'
        
        Returns:
            Dict with intrinsic, pnl, pnl_pct
        """
        if option_type == 'call':
            intrinsic = max(spot - strike, 0)
        else:
            intrinsic = max(strike - spot, 0)
        
        if direction == "long":
            pnl = intrinsic - entry_price
        else:
            pnl = entry_price - intrinsic
        
        pnl_pct = pnl / entry_price if entry_price > 0 else 0
        
        return {
            "intrinsic": intrinsic,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
        }
    
    def spread_payoff(
        self,
        spot: float,
        long_strike: float,
        short_strike: float,
        option_type: str,
        net_debit: float,
        direction: str = "long",
    ) -> Dict[str, float]:
        """
        Calculate debit spread payoff at expiry.
        
        For a bull call spread: long lower strike, short higher strike
        For a bear put spread: long higher strike, short lower strike
        
        Args:
            spot: Underlying price at expiry
            long_strike: Strike of long leg
            short_strike: Strike of short leg
            option_type: 'call' or 'put'
            net_debit: Net premium paid
            direction: 'long' (debit) or 'short' (credit)
        
        Returns:
            Dict with max_profit, max_loss, pnl, pnl_pct
        """
        if option_type == 'call':
            long_intrinsic = max(spot - long_strike, 0)
            short_intrinsic = max(spot - short_strike, 0)
        else:
            long_intrinsic = max(long_strike - spot, 0)
            short_intrinsic = max(short_strike - spot, 0)
        
        spread_value = long_intrinsic - short_intrinsic
        
        if direction == "long":
            pnl = spread_value - net_debit
            max_profit = abs(short_strike - long_strike) - net_debit
            max_loss = -net_debit
        else:
            pnl = net_debit - spread_value
            max_profit = net_debit
            max_loss = -(abs(short_strike - long_strike) - net_debit)
        
        pnl_pct = pnl / abs(net_debit) if net_debit != 0 else 0
        
        return {
            "spread_value": spread_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "max_profit": max_profit,
            "max_loss": max_loss,
        }


class ScenarioRunner:
    """
    Run pricing scenarios for interval-based backtesting.
    
    Generates base/conservative/stressed outcomes.
    """
    
    def __init__(self, pricer: OptionPricer):
        self.pricer = pricer
    
    def run_scenarios(
        self,
        spot_entry: float,
        spot_exit: float,
        strike: float,
        dte_entry: float,
        dte_exit: float,
        option_type: str,
        entry_iv: float,
        realized_vol: float,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run base/conservative/stressed scenarios.
        
        Args:
            spot_entry: Spot at entry
            spot_exit: Spot at exit
            strike: Option strike
            dte_entry: DTE at entry
            dte_exit: DTE at exit
            option_type: 'call' or 'put'
            entry_iv: IV at entry
            realized_vol: Realized volatility
        
        Returns:
            Dict with 'base', 'conservative', 'stressed' scenario results
        """
        # Entry pricing (same for all scenarios)
        entry = self.pricer.price_at_entry(
            spot=spot_entry,
            strike=strike,
            dte=dte_entry,
            option_type=option_type,
            iv_estimate=entry_iv,
        )
        
        # Use ask for entry (conservative fill)
        entry_price = entry["ask"] if entry["ask"] else entry["mid"]
        
        scenarios = {}
        
        # Base case: mid fills, stable IV
        exit_base = self.pricer.price_at_exit(
            spot=spot_exit,
            strike=strike,
            dte=dte_exit,
            option_type=option_type,
            entry_price=entry_price,
            iv_estimate=entry_iv,  # IV unchanged
        )
        scenarios["base"] = {
            "entry_price": entry_price,
            "exit_price": exit_base["mid"],
            "pnl": exit_base["pnl"],
            "pnl_pct": exit_base["pnl_pct"],
            "iv_assumption": "stable",
        }
        
        # Conservative: worst spread, mild IV crush on winners
        spot_move = (spot_exit - spot_entry) / spot_entry
        is_favorable = (spot_move > 0 and option_type == 'call') or (spot_move < 0 and option_type == 'put')
        
        iv_change = -0.05 if is_favorable else 0.03  # IV crush on winners, expansion on losers
        
        exit_conservative = self.pricer.price_at_exit(
            spot=spot_exit,
            strike=strike,
            dte=dte_exit,
            option_type=option_type,
            entry_price=entry_price,
            iv_estimate=entry_iv,
            iv_change=iv_change,
        )
        # Use bid for exit (conservative fill)
        conservative_exit_price = exit_conservative["bid"] if exit_conservative["bid"] else exit_conservative["mid"]
        conservative_pnl = conservative_exit_price - entry_price if conservative_exit_price else None
        
        scenarios["conservative"] = {
            "entry_price": entry_price,
            "exit_price": conservative_exit_price,
            "pnl": conservative_pnl,
            "pnl_pct": conservative_pnl / entry_price if conservative_pnl and entry_price else None,
            "iv_assumption": f"{'crush' if is_favorable else 'expansion'} {iv_change:+.0%}",
        }
        
        # Stressed: adverse IV shift, worst fills
        iv_change_stressed = -0.10 if is_favorable else 0.08
        
        exit_stressed = self.pricer.price_at_exit(
            spot=spot_exit,
            strike=strike,
            dte=dte_exit,
            option_type=option_type,
            entry_price=entry_price,
            iv_estimate=entry_iv,
            iv_change=iv_change_stressed,
        )
        # Additional slippage on stressed exit
        stressed_exit_price = exit_stressed["bid"] * 0.97 if exit_stressed["bid"] else None
        stressed_pnl = stressed_exit_price - entry_price if stressed_exit_price else None
        
        scenarios["stressed"] = {
            "entry_price": entry_price,
            "exit_price": stressed_exit_price,
            "pnl": stressed_pnl,
            "pnl_pct": stressed_pnl / entry_price if stressed_pnl and entry_price else None,
            "iv_assumption": f"{'crush' if is_favorable else 'expansion'} {iv_change_stressed:+.0%} + slippage",
        }
        
        return scenarios


class SameBarPolicy(Enum):
    """Policy for handling same-bar TP/SL ambiguity."""
    STOP_FIRST = "stop_first"      # Assume stop hit first (conservative)
    TP_FIRST = "tp_first"          # Assume TP hit first (optimistic)
    PROBABILISTIC = "probabilistic"  # Weight by distance to each level
    CLOSE_ONLY = "close_only"      # Only use close price, ignore intrabar


def resolve_same_bar_ambiguity(
    high: float,
    low: float,
    close: float,
    entry_price: float,
    tp_level: float,
    sl_level: float,
    is_long: bool,
    policy: SameBarPolicy = SameBarPolicy.STOP_FIRST,
) -> Tuple[str, float]:
    """
    Resolve ambiguity when both TP and SL are touched in same bar.
    
    Args:
        high: Bar high
        low: Bar low
        close: Bar close
        entry_price: Entry price
        tp_level: Take profit price level
        sl_level: Stop loss price level
        is_long: True if long position
        policy: Resolution policy
    
    Returns:
        (exit_type, exit_price) tuple
    """
    if is_long:
        tp_touched = high >= tp_level
        sl_touched = low <= sl_level
    else:
        tp_touched = low <= tp_level
        sl_touched = high >= sl_level
    
    if not tp_touched and not sl_touched:
        return ("hold", close)
    
    if tp_touched and not sl_touched:
        return ("tp", tp_level)
    
    if sl_touched and not tp_touched:
        return ("stop", sl_level)
    
    # Both touched - apply policy
    if policy == SameBarPolicy.STOP_FIRST:
        return ("stop", sl_level)
    
    elif policy == SameBarPolicy.TP_FIRST:
        return ("tp", tp_level)
    
    elif policy == SameBarPolicy.CLOSE_ONLY:
        return ("hold", close)
    
    elif policy == SameBarPolicy.PROBABILISTIC:
        # Weight by distance from open (proxy for which was hit first)
        # Closer to entry = more likely hit first
        if is_long:
            tp_dist = abs(tp_level - entry_price)
            sl_dist = abs(entry_price - sl_level)
        else:
            tp_dist = abs(entry_price - tp_level)
            sl_dist = abs(sl_level - entry_price)
        
        total_dist = tp_dist + sl_dist
        if total_dist == 0:
            return ("stop", sl_level)  # Fallback
        
        # Probability of stop first is proportional to how close SL is
        stop_prob = tp_dist / total_dist  # Closer TP = higher stop prob (counterintuitive but correct)
        
        # Use deterministic midpoint for reproducibility
        if stop_prob >= 0.5:
            return ("stop", sl_level)
        else:
            return ("tp", tp_level)
    
    return ("stop", sl_level)  # Default fallback
