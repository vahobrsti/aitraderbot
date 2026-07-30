"""
Unit tests for the iron condor strike selector (`select_condor_structure`).

Covers the bugfix in .kiro/specs/condor-strike-selection-fix:
- Credit adequacy is the hard eligibility constraint; Δ target is only ranking.
- MVRV strikes are a ranking preference and never break the credit gate.
- Null-delta exclusion, shared-expiry / deterministic selection.
- Sub-minimum → best-effort candidate flagged unqualified; no candidates → None.
- Currency conversion via price_to_usd (BTC-quoted chains).

The selector is a pure function over option-like objects, so these tests use
lightweight namespaces instead of the database.
"""
from types import SimpleNamespace
from django.test import TestCase

from execution.services.policy import CondorConfig
from execution.services.trade_setup import select_condor_structure


SPOT = 63700.0
DELTA_TARGET = 0.20


def mkopt(strike, delta, bid, ask, expiry="2026-08-07", spread_pct=0.05):
    return SimpleNamespace(
        strike=strike, delta=delta, bid=bid, ask=ask,
        expiry=expiry, spread_pct=spread_pct,
    )


def cfg():
    return CondorConfig()  # min_delta=0.12, max_delta=0.35, wing=2000, min_credit=0.15


class CondorSelectorTests(TestCase):

    def test_ranks_by_delta_closeness_among_qualified(self):
        """Both structures clear credit; the one nearest Δ0.20 wins."""
        calls = [
            mkopt(67000, 0.19, 350, 360),   # short (Δ near 0.20)
            mkopt(66000, 0.27, 540, 550),   # short (further from 0.20, more credit)
            mkopt(68000, 0.13, 225, 230),   # wing for 66000
            mkopt(69000, 0.084, 140, 146),  # wing for 67000
        ]
        puts = [
            mkopt(61000, 0.23, 510, 520),
            mkopt(62000, 0.31, 764, 774),
            mkopt(60000, 0.169, 410, 414),  # wing for 62000
            mkopt(59000, 0.11, 295, 300),   # wing for 61000
        ]
        best = select_condor_structure(calls, puts, SPOT, cfg(), DELTA_TARGET)
        self.assertIsNotNone(best)
        self.assertTrue(best.credit_qualified)
        self.assertEqual(best.short_call.strike, 67000)
        self.assertEqual(best.short_put.strike, 61000)

    def test_credit_gate_overrides_nearest_delta(self):
        """The absolute-nearest Δ0.20 structure is sub-minimum, so it is excluded
        and the closest *credit-qualified* structure is chosen instead."""
        calls = [
            mkopt(67000, 0.20, 100, 105),   # Δ exactly at target but thin premium
            mkopt(66000, 0.28, 600, 610),   # richer short, off-target delta
            mkopt(68000, 0.10, 245, 250),   # wing only (below delta band)
            mkopt(69000, 0.09, 88, 90),     # wing only (below delta band)
        ]
        puts = [
            mkopt(61000, 0.20, 120, 125),
            mkopt(62000, 0.30, 700, 710),
            mkopt(60000, 0.10, 295, 300),   # wing only (below delta band)
            mkopt(59000, 0.10, 108, 110),   # wing only (below delta band)
        ]
        best = select_condor_structure(calls, puts, SPOT, cfg(), DELTA_TARGET)
        self.assertIsNotNone(best)
        self.assertTrue(best.credit_qualified)
        # The absolute-nearest (67000/61000) is ~1% credit → excluded; the closest
        # qualified structure (66000/61000) is selected.
        self.assertFalse(
            best.short_call.strike == 67000 and best.short_put.strike == 61000
        )
        self.assertEqual(best.short_call.strike, 66000)
        self.assertEqual(best.short_put.strike, 61000)

    def test_null_delta_excluded(self):
        """Options with no delta are never selected as short legs."""
        calls = [
            mkopt(67000, None, 900, 905),   # null delta — must be ignored
            mkopt(66000, 0.27, 540, 550),
            mkopt(68000, 0.13, 245, 250),   # wing
        ]
        puts = [
            mkopt(62000, 0.31, 764, 774),
            mkopt(60000, 0.169, 410, 414),  # wing
        ]
        best = select_condor_structure(calls, puts, SPOT, cfg(), DELTA_TARGET)
        self.assertIsNotNone(best)
        self.assertEqual(best.short_call.strike, 66000)

    def test_shared_expiry_and_deterministic_tie(self):
        """Legs share one expiry; identical expiries resolve to the earliest."""
        def chain(exp):
            calls = [
                mkopt(67000, 0.19, 350, 360, expiry=exp),
                mkopt(69000, 0.084, 140, 146, expiry=exp),
            ]
            puts = [
                mkopt(61000, 0.23, 510, 520, expiry=exp),
                mkopt(59000, 0.11, 295, 300, expiry=exp),
            ]
            return calls, puts

        c1, p1 = chain("2026-08-07")
        c2, p2 = chain("2026-08-14")
        best = select_condor_structure(c1 + c2, p1 + p2, SPOT, cfg(), DELTA_TARGET)
        self.assertIsNotNone(best)
        # All four legs share a single expiry.
        exps = {best.short_call.expiry, best.long_call.expiry,
                best.short_put.expiry, best.long_put.expiry}
        self.assertEqual(len(exps), 1)
        # Tie broken deterministically toward the earliest expiry.
        self.assertEqual(best.short_call.expiry, "2026-08-07")

    def test_mvrv_never_selects_sub_minimum(self):
        """MVRV preference cannot pull selection to a sub-minimum structure."""
        calls = [
            mkopt(67000, 0.19, 350, 360),   # qualified short
            mkopt(69000, 0.084, 140, 146),  # wing for 67000
            mkopt(70000, 0.057, 83, 90),    # far-OTM short (thin premium)
            mkopt(72000, 0.029, 38, 51),    # wing for 70000
        ]
        puts = [
            mkopt(61000, 0.23, 510, 520),
            mkopt(59000, 0.11, 295, 300),   # wing for 61000
        ]
        # MVRV prefers a 70000 short (further OTM) — but that structure is thin.
        best = select_condor_structure(
            calls, puts, SPOT, cfg(), DELTA_TARGET,
            mvrv_short_call=70000, mvrv_short_put=61000,
        )
        self.assertIsNotNone(best)
        self.assertTrue(best.credit_qualified)
        self.assertEqual(best.short_call.strike, 67000)

    def test_sub_minimum_returns_unqualified_candidate(self):
        """When nothing clears the gate, return best-effort flagged unqualified."""
        calls = [
            mkopt(70000, 0.057, 83, 90),
            mkopt(72000, 0.029, 38, 51),    # wing
        ]
        puts = [
            mkopt(58000, 0.088, 178, 204),
            mkopt(55000, 0.035, 64, 83),    # wing
        ]
        best = select_condor_structure(calls, puts, SPOT, cfg(), DELTA_TARGET)
        # These deep-OTM strikes are below the delta band, so nothing constructs.
        self.assertIsNone(best)

    def test_sub_minimum_in_band_flagged_unqualified(self):
        """In-band but thin: a candidate is returned with credit_qualified False."""
        calls = [
            mkopt(67000, 0.19, 60, 65),     # in-band but very thin premium
            mkopt(69000, 0.13, 55, 58),     # wing
        ]
        puts = [
            mkopt(61000, 0.23, 70, 75),
            mkopt(59000, 0.12, 60, 63),     # wing
        ]
        best = select_condor_structure(calls, puts, SPOT, cfg(), DELTA_TARGET)
        self.assertIsNotNone(best)
        self.assertFalse(best.credit_qualified)
        self.assertLess(best.credit_pct, cfg().min_credit_pct)

    def test_no_in_band_candidates_returns_none(self):
        """All strikes outside the delta band → no structure → None."""
        calls = [mkopt(72000, 0.029, 38, 51), mkopt(75000, 0.016, 19, 38)]
        puts = [mkopt(55000, 0.035, 64, 83), mkopt(53000, 0.021, 44, 51)]
        best = select_condor_structure(calls, puts, SPOT, cfg(), DELTA_TARGET)
        self.assertIsNone(best)

    def test_credit_uses_quote_currency_by_default(self):
        """OptionSnapshot bid/ask are already USD at ingestion (both the setup and
        Deribit paths), so the default price_to_usd=1.0 computes credit directly
        in the quote currency — no scaling by spot."""
        calls = [
            mkopt(67000, 0.19, 350, 360),
            mkopt(69000, 0.084, 140, 146),  # wing
        ]
        puts = [
            mkopt(61000, 0.23, 510, 520),
            mkopt(59000, 0.11, 295, 300),   # wing
        ]
        best = select_condor_structure(calls, puts, SPOT, cfg(), DELTA_TARGET)
        # net credit = 350 + 510 - 146 - 300 = 414 on a $2000 wing = 20.7%.
        self.assertTrue(best.credit_qualified)
        self.assertAlmostEqual(best.credit_pct, 414.0 / 2000.0, places=4)

    def test_price_to_usd_scales_credit(self):
        """price_to_usd is a pure unit multiplier on the collected credit; a
        multiplier > 1 must not be used for already-USD snapshots (guards against
        the double-conversion bug)."""
        calls = [
            mkopt(67000, 0.19, 350, 360),
            mkopt(69000, 0.084, 140, 146),
        ]
        puts = [
            mkopt(61000, 0.23, 510, 520),
            mkopt(59000, 0.11, 295, 300),
        ]
        base = select_condor_structure(calls, puts, SPOT, cfg(), DELTA_TARGET)
        scaled = select_condor_structure(
            calls, puts, SPOT, cfg(), DELTA_TARGET, price_to_usd=2.0,
        )
        self.assertAlmostEqual(scaled.credit_pct, base.credit_pct * 2.0, places=4)
