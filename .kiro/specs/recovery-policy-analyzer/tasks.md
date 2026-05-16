# Implementation Plan: Recovery Policy Analyzer Enhancement

## Overview

**Status: EXISTING COMMAND FOUND** - The recovery analyzer already exists as `analyze_loser_recovery.py` with comprehensive functionality. Based on analysis findings, this implementation plan focuses on enhancing the existing command with policy integration and advanced recovery recommendations.

## Analysis Findings Summary

The existing command revealed:
- **+32% net edge** for flipping vs holding across 75 adverse trades
- **All signal types** benefit from flipping (edge ranges from +14.3% to +100%)
- **Recovery success rate**: 61.3% vs 29.3% late recovery rate
- **Key insight**: Cut losses strategy needed for trades with <2% recovery potential

## Enhanced Tasks

- [x] 1. Add policy integration to existing command
  - [x] 1.1 Enhance PolicyConfigAdapter integration
    - Modify existing `analyze_loser_recovery.py` to use `execution.services.policy.get_policy()`
    - Replace hardcoded TTH_P75_BY_TYPE with dynamic policy.get_path_profile() calls
    - Replace hardcoded MAE_W_P75_BY_TYPE with dynamic policy.path_profiles["mae_p75"] / 2
    - Add fallback to existing hardcoded values if policy data unavailable
    - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4_

  - [x] 1.2 Add recovery recommendation logic
    - Enhance RecoveryCandidate dataclass with `recommendation` property
    - Implement logic: recovery_mfe > 5% → "FLIP", < 2% → "CUT", else → "HOLD"
    - Add recommendation column to detailed trade list output
    - _Requirements: 5.4, 5.5, 5.6_

- [x] 2. Enhance decision matrix with cut losses strategy
  - [x] 2.1 Add three-way recommendation system
    - Modify decision matrix to show FLIP vs HOLD vs CUT recommendations
    - Calculate optimal thresholds based on recovery MFE distribution
    - Add "Cut Losses" category for trades with <2% recovery potential
    - Display recommendation percentages for each strategy
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 2.2 Add recovery MFE analysis
    - Add recovery MFE distribution analysis (quartiles, percentiles)
    - Identify optimal flip/cut thresholds based on historical data
    - Display threshold recommendations in output
    - _Requirements: 6.1, 6.2_

- [x] 3. Add policy parameter recommendations
  - [x] 3.1 Generate policy enhancement suggestions
    - Analyze which signal types have highest flip edge
    - Suggest recovery_flip_threshold and recovery_cut_threshold parameters
    - Recommend recovery_target values per signal type
    - Output policy enhancement recommendations
    - _Requirements: 6.1, 6.2_

  - [x] 3.2 Add recovery policy configuration
    - Add --recovery-flip-threshold and --recovery-cut-threshold CLI arguments
    - Allow testing different threshold values
    - Show impact of different thresholds on recommendation distribution
    - _Requirements: 8.1_

- [x] 4. Enhance output with actionable insights
  - [x] 4.1 Add executive summary section
    - Add top-level summary with key metrics and recommendations
    - Include "Policy Enhancement Suggestions" section
    - Show expected improvement from implementing recovery policy
    - _Requirements: 6.1, 6.2_

  - [x] 4.2 Add recovery strategy comparison
    - Compare current "hold until expiry" vs proposed recovery policy
    - Calculate expected P&L improvement from recovery policy
    - Show win rate improvements by signal type
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 5. Add recovery policy simulation
  - [x] 5.1 Implement policy backtesting
    - Add --simulate-policy flag to test recovery policy on historical data
    - Calculate P&L impact of FLIP/HOLD/CUT decisions
    - Compare simulated results vs actual hold-to-expiry results
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 5.2 Add sensitivity analysis
    - Test different recovery thresholds (1%, 2%, 3%, 5%, 10%)
    - Show optimal threshold selection based on risk/reward
    - Display threshold sensitivity table
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 6. Integration with execution policy
  - [x] 6.1 Add recovery parameters to policy.py
    - Add RecoveryConfig dataclass to policy.py
    - Include recovery_flip_threshold, recovery_cut_threshold, recovery_target per signal
    - Integrate with existing ExitConfig structure
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 6.2 Create recovery execution service
    - Create `execution/services/recovery.py` module
    - Implement RecoveryDecisionEngine class
    - Add methods: should_flip(), should_cut(), get_recovery_target()
    - Integrate with existing trade execution pipeline
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 7. Enhanced testing and validation
  - [x] 7.1 Add recovery policy validation tests
    - Test policy integration with various signal types
    - Validate recovery threshold calculations
    - Test edge case handling (missing policy data, invalid thresholds)
    - _Requirements: 8.1, 8.2, 8.3_

  - [x] 7.2 Add performance regression tests
    - Ensure enhanced command maintains existing functionality
    - Test all existing CLI arguments still work
    - Validate output format compatibility
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

## Implementation Priority

**Phase 1 (High Priority):**
- Tasks 1.1, 1.2: Policy integration and recommendation logic
- Task 2.1: Enhanced decision matrix with cut losses
- Task 4.1: Executive summary with actionable insights

**Phase 2 (Medium Priority):**
- Tasks 3.1, 3.2: Policy parameter recommendations
- Task 5.1: Recovery policy simulation
- Task 6.1: Integration with policy.py

**Phase 3 (Enhancement):**
- Task 6.2: Recovery execution service
- Tasks 7.1, 7.2: Comprehensive testing
- Tasks 2.2, 4.2, 5.2: Advanced analytics

## Notes

- **DISCOVERY**: The recovery analyzer already exists as `analyze_loser_recovery.py` with comprehensive functionality
- **FINDINGS**: Analysis shows +32% net edge for flipping vs holding across all signal types
- **STRATEGY**: Enhance existing command rather than create new one
- **PRIORITY**: Focus on policy integration and actionable recovery recommendations
- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- The implementation enhances existing trade analysis infrastructure
- Python is used as the implementation language (matching the existing codebase)

## Key Findings from Analysis

| Signal Type | Edge | Recommendation |
|-------------|------|----------------|
| OPTION_CALL | +100% | Always flip |
| PRIMARY_SHORT | +45.5% | Flip |
| TACTICAL_PUT | +44.4% | Flip |
| BEAR_PROBE | +33.3% | Flip |
| LONG | +27.3% | Flip |
| BULL_PROBE | +22.2% | Flip |
| OPTION_PUT | +16.7% | Flip |
| MVRV_SHORT | +14.3% | Flip |

**Overall**: 48% should flip, 16% should hold, 23% should cut losses, 13% either works

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "3.1"] },
    { "id": 2, "tasks": ["4.1", "5.1"] },
    { "id": 3, "tasks": ["6.1", "7.1"] },
    { "id": 4, "tasks": ["6.2", "7.2"] }
  ]
}
```
