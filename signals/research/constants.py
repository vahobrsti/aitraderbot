"""
Research contract: canonical targets, horizons, and required columns.
"""

# ── Canonical horizon & target ──────────────────────────────────────
CANONICAL_HORIZON = 14   # days
CANONICAL_TARGET = 0.05  # 5 %

# ── Primary targets (the pipeline optimises against these) ──────────
PRIMARY_TARGETS = [
    "label_good_move_long",
    "label_good_move_short",
    "ret_14d",
]

# ── Secondary diagnostics (reported, not optimised) ─────────────────
SECONDARY_DIAGNOSTICS = [
    "ret_7d",
    "ret_21d",
    "mfe_14d",
    "mae_14d",
]

# ── Return horizons to compute ──────────────────────────────────────
RETURN_HORIZONS = [1, 7, 14, 21]

# ── Minimum sample size for group stats ─────────────────────────────
MIN_SAMPLE_THRESHOLD = 10

# ── Required regime columns (must exist in the feature CSV) ─────────
REQUIRED_REGIME_COLUMNS = [
    # MDIA
    "mdia_regime_strong_inflow",
    "mdia_regime_inflow",
    "mdia_regime_aging",
    # Whales
    "whale_regime_broad_accum",
    "whale_regime_strategic_accum",
    "whale_regime_distribution_strong",
    "whale_regime_distribution",
    "whale_regime_mixed",
    # MVRV-LS
    "mvrv_ls_regime_call_confirm_recovery",
    "mvrv_ls_regime_call_confirm_trend",
    "mvrv_ls_regime_call_confirm",
    "mvrv_ls_regime_distribution_warning",
    "mvrv_ls_regime_put_confirm",
    # Conflicts (used by fusion scoring, good to validate)
    "mvrv_ls_conflict",
    "whale_mega_conflict",
]

# ── Canonical bucket names per metric ───────────────────────────────
MDIA_BUCKETS = ["strong_inflow", "inflow", "aging", "neutral"]
WHALE_BUCKETS = [
    "broad_accum",
    "strategic_accum",
    "distribution_strong",
    "distribution",
    "mixed",
    "neutral",
]
MVRV_LS_BUCKETS = [
    "early_recovery",
    "trend_confirm",
    "call_confirm",
    "distribution_warning",
    "put_confirm",
    "neutral",
]

# ── Bucket column names in the research table ───────────────────────
BUCKET_COLS = ["mdia_bucket", "whale_bucket", "mvrv_ls_bucket"]

COMBO_2_COLS = [
    "combo_2_mdia_whale",
    "combo_2_mdia_mvrv",
    "combo_2_whale_mvrv",
]
COMBO_3_COL = "combo_3_all"
