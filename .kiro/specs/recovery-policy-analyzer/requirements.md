# Requirements Document

## Introduction

The Recovery Policy Analyzer extends the existing `analyze_loser_recovery.py` command to provide a comprehensive analysis of trades that fail to hit their target by the TTH p75 checkpoint day. The analyzer uses signal-specific TTH p75 values from `policy.py` for checkpoint timing and MAE(W) p75 / 2 as the adverse threshold to identify recovery candidates. It outputs a decision matrix showing whether to hold (late recovery), flip (recovery wins), both win, or both lose for each trade type.

## Glossary

- **Recovery_Policy_Analyzer**: The Django management command that analyzes loser recovery potential using policy-derived parameters
- **TTH_p75**: Time-to-hit 75th percentile - the number of days by which 75% of winning trades hit their target, sourced from `path_profiles` in `policy.py`
- **MAE_W_p75**: Maximum Adverse Excursion for winners at 75th percentile - the worst drawdown experienced by winning trades before hitting target
- **Adverse_Threshold**: The price movement threshold (MAE(W) p75 / 2) that triggers recovery analysis; direction-specific (down for longs, up for shorts)
- **Checkpoint_Day**: The day at which a trade is evaluated for recovery potential, derived from TTH p75 per signal type
- **Original_Target**: The profit target for the original trade direction (5%)
- **Recovery_Target**: The profit target for the flipped recovery trade (3%)
- **Decision_Matrix**: A 2x2 outcome classification showing hold vs flip outcomes for adverse trades
- **Trade_Builder**: The logic from `analyze_hit_rate.py` that constructs trades using fusion signals, overlays, and cooldowns

## Requirements

### Requirement 1: Policy-Based Checkpoint Configuration

**User Story:** As a trader, I want the analyzer to use TTH p75 values from policy.py so that checkpoint timing is consistent with the calibrated execution policy.

#### Acceptance Criteria

1. WHEN the Recovery_Policy_Analyzer initializes, THE Recovery_Policy_Analyzer SHALL read TTH p75 values from `path_profiles` in `policy.py` via `get_policy().get_path_profile(signal_type)`
2. THE Recovery_Policy_Analyzer SHALL map signal types to checkpoint days as follows:
   - CALL: Day 7 (TTH p75=7d)
   - PUT: Day 6 (TTH p75=6d)
   - OPTION_CALL: Day 5 (TTH p75=5d)
   - OPTION_PUT: Day 3 (TTH p75=3d)
   - TACTICAL_PUT: Day 6 (TTH p75=5.5d rounded up)
   - BULL_PROBE: Day 5 (TTH p75=5d)
   - BEAR_PROBE: Day 9 (TTH p75=8.5d rounded up)
   - MVRV_SHORT: Day 10 (TTH p75=10d)
   - IRON_CONDOR: Day 7 (7d horizon)
3. WHEN a signal type is not found in path_profiles, THE Recovery_Policy_Analyzer SHALL use a default checkpoint of 7 days

### Requirement 2: Policy-Based Adverse Threshold Configuration

**User Story:** As a trader, I want the analyzer to use MAE(W) p75 / 2 as the adverse threshold so that recovery candidates are identified using calibrated risk parameters.

#### Acceptance Criteria

1. WHEN the Recovery_Policy_Analyzer evaluates a trade for recovery potential, THE Recovery_Policy_Analyzer SHALL calculate the adverse threshold as MAE(W) p75 / 2 from `path_profiles` in `policy.py`
2. THE Recovery_Policy_Analyzer SHALL apply direction-specific adverse thresholds:
   - CALL: adverse if price down 2.35% (4.71% / 2)
   - PUT: adverse if price up 2.2% (4.41% / 2)
   - OPTION_CALL: adverse if price down 4.24% (8.48% / 2)
   - OPTION_PUT: adverse if price up 3.41% (6.82% / 2)
   - TACTICAL_PUT: adverse if price up 1.54% (3.08% / 2)
   - BULL_PROBE: adverse if price down 1.92% (3.84% / 2)
   - BEAR_PROBE: adverse if price up 3.27% (6.53% / 2)
   - MVRV_SHORT: adverse if price up 3.6% (7.19% / 2)
3. WHEN a signal type is not found in path_profiles, THE Recovery_Policy_Analyzer SHALL use a default adverse threshold of 2.0%
4. WHEN the user provides an `--adverse-threshold` command-line argument, THE Recovery_Policy_Analyzer SHALL use the user-provided value instead of the policy-derived value

### Requirement 3: Target Configuration

**User Story:** As a trader, I want to configure original and recovery targets so that I can analyze recovery potential with appropriate profit expectations.

#### Acceptance Criteria

1. THE Recovery_Policy_Analyzer SHALL use 5% as the default original target for determining if the original trade hit
2. THE Recovery_Policy_Analyzer SHALL use 3% as the default recovery target for determining if the flipped trade would hit
3. WHEN the user provides a `--target` command-line argument, THE Recovery_Policy_Analyzer SHALL use the user-provided value as the original target
4. WHEN the user provides a `--recovery-target` command-line argument, THE Recovery_Policy_Analyzer SHALL use the user-provided value as the recovery target

### Requirement 4: Trade Building Logic Reuse

**User Story:** As a developer, I want the analyzer to reuse trade building logic from analyze_hit_rate.py so that trade identification is consistent across analysis tools.

#### Acceptance Criteria

1. THE Recovery_Policy_Analyzer SHALL identify trades using the same fusion signal logic as `analyze_hit_rate.py`
2. THE Recovery_Policy_Analyzer SHALL apply overlay filtering unless the `--no-overlay` flag is provided
3. THE Recovery_Policy_Analyzer SHALL apply cooldown logic unless the `--no-cooldown` flag is provided
4. THE Recovery_Policy_Analyzer SHALL support filtering by trade type via the `--type` argument
5. THE Recovery_Policy_Analyzer SHALL support filtering by year via the `--year` argument
6. THE Recovery_Policy_Analyzer SHALL exclude IRON_CONDOR trades from recovery analysis since they are neutral direction

### Requirement 5: Decision Matrix Output

**User Story:** As a trader, I want to see a decision matrix showing hold vs flip outcomes so that I can make informed decisions about adverse trades.

#### Acceptance Criteria

1. THE Recovery_Policy_Analyzer SHALL output a decision matrix with four outcome categories:
   - "Both win": original trade eventually hits AND recovery trade would hit
   - "Hold (late recovery)": original trade eventually hits AND recovery trade would miss
   - "Flip (recovery wins)": original trade misses AND recovery trade would hit
   - "Both lose": original trade misses AND recovery trade would miss
2. THE Recovery_Policy_Analyzer SHALL display the count and percentage for each outcome category
3. THE Recovery_Policy_Analyzer SHALL calculate and display the net edge as (flip_wins_pct - hold_wins_pct)
4. WHEN the net edge exceeds +5%, THE Recovery_Policy_Analyzer SHALL recommend flipping
5. WHEN the net edge is below -5%, THE Recovery_Policy_Analyzer SHALL recommend holding
6. WHEN the net edge is between -5% and +5%, THE Recovery_Policy_Analyzer SHALL indicate neutral recommendation

### Requirement 6: Console Output Format

**User Story:** As a trader, I want clear console output with summary statistics so that I can quickly understand recovery potential by trade type.

#### Acceptance Criteria

1. THE Recovery_Policy_Analyzer SHALL output a summary table by trade type showing:
   - Number of recovery candidates
   - Original hit percentage (trades that eventually hit despite being adverse at checkpoint)
   - Recovery hit percentage (trades where flipping would have hit)
   - Average checkpoint return
   - Average recovery MFE (maximum favorable excursion)
   - Edge (recovery hit % - original hit %)
2. THE Recovery_Policy_Analyzer SHALL output overall statistics including:
   - Total trades adverse at checkpoint
   - Original trades that eventually hit
   - Recovery trades that would hit
   - Average and median recovery MFE
3. THE Recovery_Policy_Analyzer SHALL output a detailed trade list sorted by recovery potential showing:
   - Date
   - Trade type
   - Direction (with emoji indicator)
   - Checkpoint day
   - Checkpoint return
   - Recovery MFE
   - Original hit status (with emoji indicator)
   - Recovery hit status (with emoji indicator)

### Requirement 7: Recovery Candidate Identification

**User Story:** As a trader, I want the analyzer to correctly identify trades that are adverse at checkpoint so that only failing trades are considered for recovery.

#### Acceptance Criteria

1. WHEN a LONG trade has price below entry by more than the adverse threshold at checkpoint, THE Recovery_Policy_Analyzer SHALL classify it as a recovery candidate
2. WHEN a SHORT trade has price above entry by more than the adverse threshold at checkpoint, THE Recovery_Policy_Analyzer SHALL classify it as a recovery candidate
3. WHEN a trade is not adverse at checkpoint, THE Recovery_Policy_Analyzer SHALL exclude it from recovery analysis
4. THE Recovery_Policy_Analyzer SHALL calculate recovery return as the maximum favorable excursion in the flipped direction from checkpoint to horizon end

### Requirement 8: Command-Line Interface

**User Story:** As a user, I want a complete command-line interface so that I can run the analyzer with various configurations.

#### Acceptance Criteria

1. THE Recovery_Policy_Analyzer SHALL accept the following command-line arguments:
   - `--csv`: Input features CSV file (default: features_14d_5pct.csv)
   - `--year`: Filter to specific year
   - `--no-overlay`: Disable overlay filtering
   - `--no-cooldown`: Disable cooldown logic
   - `--long-model`: Path to long model (default: models/long_model.joblib)
   - `--short-model`: Path to short model (default: models/short_model.joblib)
   - `--type`: Filter by trade type
   - `--horizon`: Forward path horizon in days (default: 14)
   - `--target`: Original target threshold (default: 0.05)
   - `--adverse-threshold`: Override adverse threshold
   - `--recovery-target`: Recovery target threshold (default: 0.03)
2. WHEN the CSV file does not exist, THE Recovery_Policy_Analyzer SHALL display an error message and exit
3. WHEN no trades match the specified filters, THE Recovery_Policy_Analyzer SHALL display an appropriate message
