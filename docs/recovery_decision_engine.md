# Recovery Decision Engine

## Overview

The Recovery Decision Engine is a service that provides recovery decision logic for adverse trades based on policy parameters. It integrates with the existing trade execution pipeline to determine when to flip, cut, or hold trades that are performing poorly.

Based on historical analysis showing **+32% net edge** for flipping vs holding across 75 adverse trades.

## Key Components

### 1. RecoveryDecisionEngine Class

The main service class that provides recovery decision logic:

```python
from execution.services.recovery import RecoveryDecisionEngine
from execution.services.policy import get_policy

engine = RecoveryDecisionEngine(get_policy())

# Check if trade should be flipped
should_flip = engine.should_flip("CALL", entry_price, current_price, days_held)

# Check if trade should be cut
should_cut = engine.should_cut("CALL", entry_price, current_price, days_held)

# Get comprehensive recovery decision
decision = engine.get_recovery_decision("CALL", entry_price, current_price, days_held)
```

### 2. RecoveryDecision Dataclass

Contains the decision result with supporting metrics:

```python
@dataclass
class RecoveryDecision:
    action: str              # "FLIP", "CUT", or "HOLD"
    confidence: float        # 0.0 to 1.0 confidence in the decision
    checkpoint_return: float # Return at checkpoint (negative = adverse)
    recovery_potential: float # Estimated recovery potential (0.0 to 1.0)
    rationale: str          # Human-readable explanation
```

### 3. Policy Integration

The engine uses policy-derived parameters for:

- **Checkpoint timing**: TTH p75 values from `policy.get_path_profile()`
- **Adverse thresholds**: MAE p75 / 2 from policy path profiles
- **Recovery thresholds**: Flip/cut/target values from `RecoveryConfig`

## Decision Logic

### Recovery Candidate Identification

A trade qualifies as a recovery candidate when:

1. **At or past checkpoint day**: Based on TTH p75 values per signal type
2. **Adverse movement**: Price moved against original direction beyond threshold

### Recovery Decision Making

The engine makes decisions based on estimated recovery potential:

- **FLIP**: Recovery potential > flip threshold (default 5%)
- **CUT**: Recovery potential < cut threshold (default 2%)
- **HOLD**: Recovery potential between thresholds or trade not adverse

### Signal Type Specific Parameters

| Signal Type | Checkpoint Day | Adverse Threshold | Edge | Recommendation |
|-------------|----------------|-------------------|------|----------------|
| CALL        | 7              | 2.35%            | +27% | Flip |
| PUT         | 6              | 2.21%            | +46% | Flip |
| OPTION_CALL | 5              | 4.24%            | +100% | Always flip |
| OPTION_PUT  | 3              | 3.41%            | +17% | Flip |
| TACTICAL_PUT| 6              | 1.54%            | +44% | Flip |
| BULL_PROBE  | 5              | 1.92%            | +22% | Flip |
| BEAR_PROBE  | 9              | 3.26%            | +33% | Flip |
| MVRV_SHORT  | 10             | 3.60%            | +14% | Flip |

## Usage Examples

### Basic Usage

```python
from execution.services.recovery import should_flip_trade, should_cut_trade

# Simple checks
if should_flip_trade("CALL", 100000, 97000, 7):
    print("Recommend flipping to SHORT")

if should_cut_trade("CALL", 100000, 95000, 7):
    print("Recommend cutting losses")
```

### Comprehensive Analysis

```python
from execution.services.recovery import get_recovery_decision

decision = get_recovery_decision("OPTION_CALL", 100000, 92000, 5)

print(f"Action: {decision.action}")
print(f"Confidence: {decision.confidence:.1%}")
print(f"Rationale: {decision.rationale}")
```

### Recovery Analysis Command

```bash
# Full recovery analysis with policy integration
python manage.py analyze_loser_recovery

# Filter by signal type
python manage.py analyze_loser_recovery --type MVRV_SHORT

# Run policy simulation
python manage.py analyze_loser_recovery --simulate-policy

# Sensitivity analysis
python manage.py analyze_loser_recovery --sensitivity-analysis

# Custom thresholds
python manage.py analyze_loser_recovery --recovery-flip-threshold 0.06 --recovery-cut-threshold 0.015
```

## Integration Points

### 1. Position Monitoring

The service can be integrated with position monitoring to:

- Continuously check positions for recovery opportunities
- Generate alerts when recovery actions are recommended
- Track recovery decision performance

### 2. Execution Pipeline

Integration with the existing execution pipeline:

- Monitor `ExecutionIntent` objects for adverse performance
- Execute recovery actions (flip, cut, hold)
- Log recovery decisions for audit and analysis

### 3. Risk Management

Integration with risk management:

- Respect position size limits for recovery trades
- Consider account-level risk when making recovery decisions
- Implement safeguards against excessive recovery actions

## Configuration

### Policy Parameters

Recovery parameters are configured in `policy.py`:

```python
recovery_configs={
    "CALL": RecoveryConfig(
        recovery_flip_threshold=0.05,  # 5% MFE threshold for flip
        recovery_cut_threshold=0.02,   # 2% MFE threshold for cut
        recovery_target=0.03,          # 3% profit target for recovery
    ),
    # ... other signal types
}
```

No additional environment variables required — uses existing policy configuration.

## Testing

```bash
# Recovery service unit tests (28 tests)
python manage.py test execution.tests_recovery

# Recovery policy validation tests (17 tests)
python manage.py test signals.tests_recovery_policy_validation

# Performance regression tests (11 tests)
python manage.py test signals.tests_recovery_performance_regression

# All recovery-related tests
python manage.py test execution.tests_recovery signals.tests_recovery_candidate signals.tests_recovery_mfe_analysis signals.tests_recovery_policy_validation signals.tests_recovery_performance_regression
```

## Performance Considerations

### Caching

The engine caches checkpoint days and adverse thresholds to avoid repeated policy lookups.

### Estimation

Recovery potential estimation uses lightweight heuristics. For production use, consider:

- Historical recovery analysis data
- Machine learning models for recovery prediction
- Real-time market condition adjustments

### Scalability

The service is designed to be stateless and can handle multiple concurrent requests.

## Future Enhancements

### 1. Machine Learning Integration

- Train ML models on historical recovery data
- Predict recovery probability based on market conditions
- Dynamic threshold adjustment based on market regime

### 2. Advanced Recovery Strategies

- Partial position recovery (scale in/out)
- Time-based recovery adjustments
- Volatility-adjusted recovery thresholds

### 3. Real-time Market Integration

- Incorporate real-time volatility measures
- Adjust recovery potential based on market conditions
- Consider correlation with other positions

## Dependencies

- `execution.services.policy`: Policy configuration
- `execution.models`: Database models for logging
- `django`: Framework dependencies
- Standard Python libraries (dataclasses, logging, etc.)

## API Reference

See the docstrings in `execution/services/recovery.py` for detailed API documentation.
