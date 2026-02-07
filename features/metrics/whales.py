import pandas as pd

def calculate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Whale Accumulation (Layered Bucket System)
    """
    feats = {}
    
    whale_groups = {
        'mega': 'btc_holders_100_10k',   # 100-10k BTC (primary signal)
        'small': 'btc_holders_1_100',    # 1-100 BTC (secondary/catalyst)
    }
    
    # Needs checking if columns exist, for now assume they do (from raw)
    
    # ========== A) PER-GROUP DIRECTIONAL BUCKETS (Z-Score Based) ==========
    whale_horizons = [1, 2, 4, 7]
    whale_thresholds = {1: 0.8, 2: 0.8, 4: 1.0, 7: 1.0}
    
    group_buckets = {}  # {group: {horizon: bucket_series}}
    
    for group, col in whale_groups.items():
        group_buckets[group] = {}
        
        for h in whale_horizons:
            # Raw balance change
            delta_h = df[col] - df[col].shift(h)
            feats[f'whale_{group}_delta_{h}d'] = delta_h
            
            # Z-score of delta
            delta_mean = delta_h.rolling(90, min_periods=30).mean().fillna(0)
            delta_std = delta_h.rolling(90, min_periods=30).std().fillna(1e-9)
            delta_z = (delta_h - delta_mean) / (delta_std + 1e-9)
            feats[f'whale_{group}_delta_z_{h}d'] = delta_z
            
            # Directional bucket
            thresh = whale_thresholds[h]
            bucket = pd.Series(0, index=df.index)
            
            is_strong_z = delta_z > thresh
            
            # Enhanced Logic: If MVRV-60d < 1.0 (Undervalued)
            # This depends on 'mvrv_usd_60d' being present in df
            if 'mvrv_usd_60d' in df.columns:
                 mvrv_60d_val = df['mvrv_usd_60d']
            else:
                 # Fallback if not physically present yet (though it should be raw)
                 mvrv_60d_val = pd.Series(1.0, index=df.index) # Neutral fallback

            if group == 'mega':
                # For Mega whales, be more sensitive at bottoms
                is_cheap_accum = (mvrv_60d_val < 1.0) & (delta_z > 0.2)
                bucket.loc[is_strong_z | is_cheap_accum] = 1
                
                # Check for PROFIT TAKING
                is_strong_distrib = delta_z < -thresh
                is_profit_distrib = (mvrv_60d_val > 1.1) & (delta_z < -0.65)
                bucket.loc[is_strong_distrib | is_profit_distrib] = -1
            else:
                bucket.loc[is_strong_z] = 1
                bucket.loc[delta_z < -thresh] = -1
            
            feats[f'whale_{group}_bucket_{h}d'] = bucket.astype(int)
            group_buckets[group][h] = bucket
    
    # ========== A.2) 14D CAMPAIGN CONFIRMATION (Mega Whales Only) ==========
    mega_col = whale_groups['mega']
    delta_14d = df[mega_col] - df[mega_col].shift(14)
    feats['whale_mega_delta_14d'] = delta_14d
    
    delta_14d_mean = delta_14d.rolling(90, min_periods=30).mean().fillna(0)
    delta_14d_std = delta_14d.rolling(90, min_periods=30).std().fillna(1e-9)
    delta_14d_z = (delta_14d - delta_14d_mean) / (delta_14d_std + 1e-9)
    feats['whale_mega_delta_z_14d'] = delta_14d_z
    
    mega_14d_bucket = pd.Series(0, index=df.index)
    mega_14d_bucket.loc[delta_14d_z > 1.2] = 1   # Campaign accumulation
    mega_14d_bucket.loc[delta_14d_z < -1.2] = -1  # Campaign distribution
    feats['whale_mega_bucket_14d'] = mega_14d_bucket.astype(int)
    group_buckets['mega'][14] = mega_14d_bucket
    
    # ========== B) PER-GROUP MULTI-HORIZON CONFIRMATION ==========
    group_regimes = {}
    
    for group in whale_groups.keys():
        buckets = group_buckets[group]
        
        accum_count = (
            (buckets[1] == 1).astype(int) +
            (buckets[2] == 1).astype(int) +
            (buckets[4] == 1).astype(int) +
            (buckets[7] == 1).astype(int)
        )
        distrib_count = (
            (buckets[1] == -1).astype(int) +
            (buckets[2] == -1).astype(int) +
            (buckets[4] == -1).astype(int) +
            (buckets[7] == -1).astype(int)
        )
        
        feats[f'whale_{group}_accum_count'] = accum_count
        feats[f'whale_{group}_distrib_count'] = distrib_count
        
        # Per-group regimes
        strong_accum = (accum_count >= 3) & (distrib_count == 0)
        moderate_accum = (accum_count >= 2) & (distrib_count == 0) & ~strong_accum
        any_accum = strong_accum | moderate_accum
        
        strong_distrib = (distrib_count >= 3) & (accum_count == 0)
        moderate_distrib = (distrib_count >= 2) & (accum_count == 0) & ~strong_distrib
        any_distrib = strong_distrib | moderate_distrib
        
        mixed = ~any_accum & ~any_distrib
        conflict = (accum_count > 0) & (distrib_count > 0)
        
        feats[f'whale_{group}_strong_accum'] = strong_accum.astype(int)
        feats[f'whale_{group}_moderate_accum'] = moderate_accum.astype(int)
        feats[f'whale_{group}_any_accum'] = any_accum.astype(int)
        feats[f'whale_{group}_strong_distrib'] = strong_distrib.astype(int)
        feats[f'whale_{group}_moderate_distrib'] = moderate_distrib.astype(int)
        feats[f'whale_{group}_any_distrib'] = any_distrib.astype(int)
        feats[f'whale_{group}_mixed'] = mixed.astype(int)
        feats[f'whale_{group}_conflict'] = conflict.astype(int)
        
        group_regimes[group] = {
            'strong_accum': strong_accum,
            'moderate_accum': moderate_accum,
            'any_accum': any_accum,
            'strong_distrib': strong_distrib,
            'any_distrib': any_distrib,
            'mixed': mixed,
        }
    
    # ========== B.2) MEGA WHALE CAMPAIGN CONFIRMATION ==========
    mega_14d_accum = group_buckets['mega'][14] == 1
    mega_14d_distrib = group_buckets['mega'][14] == -1
    mega_any_accum = group_regimes['mega']['any_accum']
    mega_any_distrib = group_regimes['mega']['any_distrib']
    
    feats['whale_mega_campaign_accum'] = (mega_14d_accum & mega_any_accum).astype(int)
    feats['whale_mega_campaign_distrib'] = (mega_14d_distrib & mega_any_distrib).astype(int)
    
    # ========== C) CROSS-GROUP INTERACTION REGIMES ==========
    mega = group_regimes['mega']
    small = group_regimes['small']
    
    # Total balance for divergence resolution
    # df['whale_total_balance'] might overwrite if we set it, but good to have
    # Assuming columns exist
    df_temp_total = df['btc_holders_1_100'] + df['btc_holders_100_10k']
    feats['whale_total_balance'] = df_temp_total
    
    total_change_7d = df_temp_total - df_temp_total.shift(7)
    feats['whale_total_change_7d'] = total_change_7d
    
    total_flat_or_rising = total_change_7d >= 0
    
    # REGIME 1: Broad Whale Accumulation
    feats['whale_regime_broad_accum'] = (
        mega['any_accum'] & small['any_accum']
    ).astype(int)
    
    # REGIME 2: Strategic Accumulation
    strategic_clean = mega['any_accum'] & small['mixed']
    strategic_divergent = mega['any_accum'] & small['any_distrib']
    
    feats['whale_regime_strategic_accum'] = (
        mega['any_accum'] & ~small['any_accum']
    ).astype(int)
    feats['whale_regime_strategic_accum_clean'] = strategic_clean.astype(int)
    feats['whale_regime_strategic_accum_divergent'] = strategic_divergent.astype(int)
    
    # REGIME 3: Retail-Driven Rally
    retail_rally_base = small['any_accum'] & (mega['any_distrib'] | mega['mixed'])
    feats['whale_regime_retail_rally'] = retail_rally_base.astype(int)
    feats['whale_regime_retail_rally_fragile'] = (
        retail_rally_base & total_flat_or_rising
    ).astype(int)
    feats['whale_regime_retail_rally_trap'] = (
        retail_rally_base & ~total_flat_or_rising
    ).astype(int)
    
    # REGIME 4: Whale Distribution
    feats['whale_regime_distribution'] = mega['any_distrib'].astype(int)
    feats['whale_regime_distribution_strong'] = mega['strong_distrib'].astype(int)
    
    # REGIME 5: Mixed / No Signal
    feats['whale_regime_mixed'] = (
        mega['mixed'] & small['mixed']
    ).astype(int)

    return pd.DataFrame(feats, index=df.index)
