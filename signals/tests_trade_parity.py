"""
Test that analyze_hit_rate and analyze_path_stats enumerate identical trade sets.

Both commands replay the fusion engine over historical data to build a list of
trades.  Any divergence means one command's diagnostics are computed on a
different universe than the other, making cross-comparisons invalid.
"""

import ast
import inspect
import textwrap

from django.test import SimpleTestCase


class TestTradeEnumerationParity(SimpleTestCase):
    """
    Structural parity check: both commands must share the same trade-type
    universe and the same gating logic for option / MVRV signals.
    """

    @staticmethod
    def _get_source(module_path: str) -> str:
        """Import a command module and return its source code."""
        import importlib
        mod = importlib.import_module(module_path)
        return inspect.getsource(mod.Command)

    def _extract_trade_types(self, source: str) -> set[str]:
        """Pull every string literal assigned to a 'type' key in trade dicts."""
        tree = ast.parse(source)
        types = set()
        for node in ast.walk(tree):
            # Match: "type": "SOME_TYPE"
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "type"
                        and isinstance(value, ast.Constant)
                        and isinstance(value.value, str)
                    ):
                        types.add(value.value)
        return types

    def test_both_commands_enumerate_same_trade_types(self):
        """Both commands must produce the exact same set of trade types."""
        hr_src = self._get_source(
            "signals.management.commands.analyze_hit_rate"
        )
        ps_src = self._get_source(
            "signals.management.commands.analyze_path_stats"
        )

        hr_types = self._extract_trade_types(hr_src)
        ps_types = self._extract_trade_types(ps_src)

        missing_in_path_stats = hr_types - ps_types
        missing_in_hit_rate = ps_types - hr_types

        self.assertEqual(
            missing_in_path_stats,
            set(),
            f"Trade types in analyze_hit_rate but missing from analyze_path_stats: "
            f"{missing_in_path_stats}",
        )
        self.assertEqual(
            missing_in_hit_rate,
            set(),
            f"Trade types in analyze_path_stats but missing from analyze_hit_rate: "
            f"{missing_in_hit_rate}",
        )

    def test_option_signals_gated_by_fusion_traded(self):
        """Option signals must only fire when fusion didn't trade (fusion_traded guard)."""
        ps_src = self._get_source(
            "signals.management.commands.analyze_path_stats"
        )
        self.assertIn(
            "fusion_traded",
            ps_src,
            "analyze_path_stats must gate option signals behind a 'fusion_traded' check, "
            "matching analyze_hit_rate",
        )

    def test_option_call_excludes_option_put(self):
        """OPTION_PUT must not fire on the same day as OPTION_CALL."""
        ps_src = self._get_source(
            "signals.management.commands.analyze_path_stats"
        )
        self.assertIn(
            "option_call_fired",
            ps_src,
            "analyze_path_stats must check option_call_fired before firing OPTION_PUT, "
            "matching analyze_hit_rate",
        )

    def test_mvrv_short_signal_present(self):
        """MVRV_SHORT logic must exist in analyze_path_stats."""
        ps_src = self._get_source(
            "signals.management.commands.analyze_path_stats"
        )
        self.assertIn(
            "MVRV_SHORT",
            ps_src,
            "analyze_path_stats must include MVRV_SHORT trade enumeration",
        )
        self.assertIn(
            "check_mvrv_short_signal",
            ps_src,
            "analyze_path_stats must call check_mvrv_short_signal",
        )

    def test_mvrv_short_cooldown_imported(self):
        """MVRV_SHORT_COOLDOWN_DAYS must be imported in analyze_path_stats."""
        ps_src = self._get_source(
            "signals.management.commands.analyze_path_stats"
        )
        self.assertIn(
            "MVRV_SHORT_COOLDOWN_DAYS",
            ps_src,
            "analyze_path_stats must import and use MVRV_SHORT_COOLDOWN_DAYS",
        )

    def test_cooldown_constants_match(self):
        """Both commands must import the same set of cooldown constants."""
        hr_src = self._get_source(
            "signals.management.commands.analyze_hit_rate"
        )
        ps_src = self._get_source(
            "signals.management.commands.analyze_path_stats"
        )

        cooldown_names = [
            "CORE_SIGNAL_COOLDOWN_DAYS",
            "PROBE_COOLDOWN_DAYS",
            "TACTICAL_PUT_COOLDOWN_DAYS",
            "OPTION_SIGNAL_COOLDOWN_DAYS",
            "MVRV_SHORT_COOLDOWN_DAYS",
        ]
        for name in cooldown_names:
            self.assertIn(name, hr_src, f"{name} missing from analyze_hit_rate")
            self.assertIn(name, ps_src, f"{name} missing from analyze_path_stats")
