# signals/management/commands/analyze_factor_weights.py
"""
Measure how much each analyze_engine factor actually moves forward price, and
derive a per-factor weight for a weighted bullishness score.

The engine (signals/engine_metrics.compute_engine_score) currently treats all 7
inputs equally: each votes -1/0/+1 and the votes are summed (range -7..+7). This
command reconstructs those exact 7 votes for every historical row, joins forward
BTC returns, and reports:

  1. RANKING  - each factor's association with forward price action (which output
     has the highest weight on price), via correlation / AUC / directional lift.
  2. WEIGHTS  - a suggested weight per factor from a joint regression of the 7
     votes on ret_14d, oriented so "higher and more positive = more bullish".

Read-only: it does not modify the engine. It just prints the numbers so a
weighted score can be designed on evidence.

Usage:
    python manage.py analyze_factor_weights
    python manage.py analyze_factor_weights --csv features_14d_5pct.csv --horizon 14
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand

from signals.engine_metrics import (
    DIRECTION_SIGNS,
    _NORMALIZED_SOURCES,
    compute_engine_score,
    compute_fusion_components,
    ensure_normalized_columns,
    _num,
)

# Display order: normalized metrics first, then fusion components.
FACTORS = ["mvrv_60d", "sentiment", "exchange_flow", "mvrv_composite",
           "mdia", "whale", "mvrv_ls"]
NORMALIZED = set(DIRECTION_SIGNS.keys())


class Command(BaseCommand):
    help = "Rank analyze_engine factors by their weight on forward price and suggest weights."

    def add_arguments(self, parser):
        parser.add_argument("--csv", type=str, default="features_14d_5pct.csv")
        parser.add_argument("--horizon", type=int, default=14,
                            help="Forward-return horizon in days for the primary target")
        parser.add_argument("--stride", type=int, default=None,
                            help="Non-overlapping subsample stride (default = horizon) "
                                 "for an overlap-robust correlation check")

    # ── data assembly ────────────────────────────────────────────────
    def _build_factor_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-row oriented signal (continuous, bullish+) and vote (-1/0/+1)
        for each of the 7 engine factors, exactly as compute_engine_score sees
        them."""
        rows = []
        for idx, row in df.iterrows():
            normalized = {}
            for name, (_val_col, z_col) in _NORMALIZED_SOURCES.items():
                normalized[name] = {"z_90": _num(row, z_col)}
            fc = compute_fusion_components(row)
            votes = compute_engine_score(normalized, fc)["votes"]

            rec = {"date": idx}
            # oriented continuous signals
            for name in NORMALIZED:
                z = normalized[name]["z_90"]
                rec[f"sig_{name}"] = (DIRECTION_SIGNS[name] * z
                                      if z is not None and not pd.isna(z) else np.nan)
            for name in ("mdia", "whale", "mvrv_ls"):
                n = fc[name].get("n", 0)
                rec[f"sig_{name}"] = (fc[name]["sum"] / n) if n > 0 else np.nan
            # discrete votes
            for name in FACTORS:
                rec[f"vote_{name}"] = votes[name]
            rows.append(rec)

        out = pd.DataFrame(rows).set_index("date")
        out.index = pd.to_datetime(out.index)
        return out

    def _forward_returns(self, feat_index: pd.DatetimeIndex, horizons) -> pd.DataFrame:
        """ret_Nd on the full daily price series, reindexed to feature dates."""
        from signals.research.fusion_table import _load_prices_from_db
        px = _load_prices_from_db()
        if px.empty:
            raise RuntimeError("RawDailyData empty: cannot compute forward returns.")
        close = px["btc_close"].astype(float)
        rets = pd.DataFrame(index=close.index)
        for h in horizons:
            rets[f"ret_{h}d"] = close.shift(-h) / close - 1.0
        return rets.reindex(feat_index.normalize())

    # ── stats helpers ────────────────────────────────────────────────
    @staticmethod
    def _pearson(x: pd.Series, y: pd.Series) -> float:
        m = x.notna() & y.notna()
        if m.sum() < 30 or x[m].std() == 0 or y[m].std() == 0:
            return np.nan
        return float(np.corrcoef(x[m], y[m])[0, 1])

    @staticmethod
    def _spearman(x: pd.Series, y: pd.Series) -> float:
        m = x.notna() & y.notna()
        if m.sum() < 30:
            return np.nan
        return float(np.corrcoef(x[m].rank(), y[m].rank())[0, 1])

    @staticmethod
    def _auc(sig: pd.Series, label: pd.Series) -> float:
        m = sig.notna() & label.notna()
        if m.sum() < 30 or label[m].nunique() < 2:
            return np.nan
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(label[m].astype(int), sig[m]))
        except Exception:
            return np.nan

    # ── main ─────────────────────────────────────────────────────────
    def handle(self, *args, **opts):
        from pathlib import Path
        csv_path = Path(opts["csv"])
        if not csv_path.exists():
            self.stderr.write(f"CSV not found: {csv_path}")
            return
        h = opts["horizon"]
        stride = opts["stride"] or h

        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df = ensure_normalized_columns(df)

        ff = self._build_factor_frame(df)
        horizons = sorted({7, h, 21})
        rets = self._forward_returns(ff.index, horizons)
        data = ff.join(rets)

        target = f"ret_{h}d"
        y = data[target]
        y_long = df["label_good_move_long"].reindex(data.index) if "label_good_move_long" in df else pd.Series(index=data.index, dtype=float)

        # ── per-factor association ───────────────────────────────────
        table = []
        for name in FACTORS:
            sig = data[f"sig_{name}"]
            vote = data[f"vote_{name}"].astype(float)
            r = self._pearson(sig, y)
            r_sub = self._pearson(sig.iloc[::stride], y.iloc[::stride])
            rho = self._spearman(sig, y)
            auc = self._auc(sig, y_long)
            # directional lift: mean fwd return when the vote is bullish vs bearish
            up = y[vote > 0].mean()
            dn = y[vote < 0].mean()
            lift = (up - dn) if (pd.notna(up) and pd.notna(dn)) else np.nan
            active = int((vote != 0).sum())
            table.append({
                "factor": name,
                "family": "normalized" if name in NORMALIZED else "fusion",
                "pearson_r": r,
                "pearson_r_nonoverlap": r_sub,
                "spearman": rho,
                "auc_long": auc,
                "lift_ret": lift,
                "active_rows": active,
            })
        tdf = pd.DataFrame(table).set_index("factor")
        tdf["abs_r"] = tdf["pearson_r"].abs()
        ranked = tdf.sort_values("abs_r", ascending=False)

        # ── joint weights: regress the 7 votes on the target ─────────
        vote_cols = [f"vote_{n}" for n in FACTORS]
        reg = data[vote_cols + [target]].dropna()
        X = reg[vote_cols].to_numpy(dtype=float)
        yv = reg[target].to_numpy(dtype=float)
        Xd = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
        joint_beta = dict(zip(FACTORS, beta[1:]))  # return units per +1 vote

        # standalone beta (univariate) for comparison
        standalone = {}
        for n in FACTORS:
            v = reg[f"vote_{n}"].to_numpy(dtype=float)
            if v.std() == 0:
                standalone[n] = np.nan
                continue
            b1, b0 = np.polyfit(v, yv, 1)
            standalone[n] = b1

        # Weights normalized so mean(|w|) = 1.0 (equal-weight baseline = 1.0 each).
        jb = np.array([joint_beta[n] for n in FACTORS])
        scale = np.mean(np.abs(jb)) or 1.0
        weights = {n: joint_beta[n] / scale for n in FACTORS}

        self._report(target, h, stride, ranked, joint_beta, standalone, weights, len(reg))

    # ── printing ─────────────────────────────────────────────────────
    def _report(self, target, h, stride, ranked, joint_beta, standalone, weights, n_reg):
        w = self.stdout.write
        w("")
        w("=" * 78)
        w(f"ANALYZE_ENGINE FACTOR WEIGHTS  |  target={target}  (forward {h}d return)")
        w("=" * 78)
        w("Orientation: every signal/vote is oriented bullish+ (higher = more bullish).")
        w("A NEGATIVE correlation means the factor's current bullish orientation in")
        w("engine_metrics is empirically INVERTED vs forward price.")
        w("")

        # (1) ranking
        w("(1) WHICH FACTOR HAS THE MOST WEIGHT ON PRICE  (sorted by |Pearson r|)")
        w("")
        w(f"    {'factor':<15}{'family':<11}{'r':>8}{'r_nonov':>9}{'spearman':>10}"
          f"{'auc_long':>10}{'lift':>9}{'active':>8}")
        w("    " + "-" * 74)
        for name, r in ranked.iterrows():
            def f(x, p=3):
                return "   N/A" if pd.isna(x) else f"{x:+.{p}f}"
            w(f"    {name:<15}{r['family']:<11}{f(r['pearson_r']):>8}"
              f"{f(r['pearson_r_nonoverlap']):>9}{f(r['spearman']):>10}"
              f"{('N/A' if pd.isna(r['auc_long']) else format(r['auc_long'],'.3f')):>10}"
              f"{f(r['lift_ret']):>9}{int(r['active_rows']):>8}")
        top = ranked.index[0]
        w("")
        w(f"    -> Highest weight on price: {top}  "
          f"(|r|={ranked.iloc[0]['abs_r']:.3f}, r={ranked.iloc[0]['pearson_r']:+.3f})")
        w("    lift = mean forward return on bullish-vote days minus bearish-vote days.")
        w("")

        # (2) weights
        w(f"(2) SUGGESTED PER-FACTOR WEIGHTS  (joint OLS of the 7 votes on {target}, "
          f"n={n_reg})")
        w("")
        w(f"    {'factor':<15}{'joint_beta':>12}{'standalone_beta':>18}{'weight':>12}")
        w("    " + "-" * 57)
        for name in FACTORS:
            w(f"    {name:<15}{joint_beta[name]:>+12.5f}"
              f"{standalone[name]:>+18.5f}{weights[name]:>+12.3f}")
        w("")
        w("    joint_beta = extra forward return per +1 vote, holding other factors")
        w("      fixed (accounts for overlap/redundancy between factors).")
        w("    weight = joint_beta rescaled so mean|weight| = 1.0 (equal-weight = 1.0).")
        w("    A weighted score would be:  sum( weight_i * vote_i ).")
        w("")
        w("    Suggested weights dict (paste-ready):")
        w("    FACTOR_WEIGHTS = {")
        for name in FACTORS:
            w(f"        {name!r}: {round(weights[name], 3)},")
        w("    }")
        w("")
        w("    NOTE: overlapping forward windows inflate significance; compare 'r' with")
        w("    'r_nonov' (stride=%d) for stability. Weak/negative-weight factors are" % stride)
        w("    candidates to down-weight or re-orient, not necessarily to keep at 1.0.")
        w("")
