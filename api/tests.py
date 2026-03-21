from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path


class FusionExplainTests(APITestCase):
    def setUp(self):
        # Create user and token for authentication
        self.user = User.objects.create_user(username='testuser', password='password')
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key)
        
        # Create mock feature data that fuse_signals can process
        # Row 1: Normal mode (cycle_day=100), should trigger STRONG_BULLISH
        # Row 2: Bear mode (cycle_day=600), should trigger bear trace
        dates = pd.date_range('2024-01-01', periods=2, freq='D')
        self.mock_df = pd.DataFrame({
            # MDIA regime columns
            'mdia_regime_strong_inflow': [1, 0],
            'mdia_regime_inflow': [0, 1],
            'mdia_regime_aging': [0, 0],
            # Whale regime columns
            'whale_regime_broad_accum': [1, 0],
            'whale_regime_mixed': [0, 0],
            'whale_regime_distrib': [0, 0],
            'whale_regime_distrib_strong': [0, 0],
            # MVRV regime columns
            'mvrv_ls_regime_call_confirm': [1, 0],
            'mvrv_ls_regime_call_confirm_recovery': [0, 0],
            'mvrv_ls_regime_put_confirm': [0, 0],
            'mvrv_ls_regime_bear_continuation': [0, 1],
            'mvrv_ls_regime_neutral': [0, 0],
            # Cycle info
            'cycle_days_since_halving': [100, 600],
            # MVRV 60d for bear mode
            'mvrv_60d': [1.50, 0.70],
        }, index=dates)

    @patch('pandas.read_csv')
    @patch.object(Path, 'exists', return_value=True)
    def test_fusion_explain_success(self, mock_exists, mock_read_csv):
        # Setup mock to return the mock dataframe
        mock_read_csv.return_value = self.mock_df.copy()
        
        url = reverse('api:fusion-explain')
        response = self.client.get(url, {'date': '2024-01-01'})
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['meta']['date'], '2024-01-01')
        
        result = response.data['result']
        self.assertIn('state', result)
        self.assertIn('confidence', result)
        self.assertIn('score', result)
        self.assertIn('components', result)
        self.assertIn('trace', result)
        
        # Verify trace semantics - normal mode has 7 states
        trace = result['trace']
        self.assertEqual(len(trace), 7)
        
        # Verify trace order matches classifier order
        expected_order = [
            "strong_bullish", "early_recovery", "bear_continuation", 
            "bear_probe", "distribution_risk", "momentum", "bull_probe"
        ]
        actual_order = [t['state'] for t in trace]
        self.assertEqual(actual_order, expected_order)
        
        # Verify the matched state correctly aligns with result.state
        matched_states = [t['state'] for t in trace if t['matched']]
        if matched_states:
            if result['state'] != 'no_trade':
                self.assertEqual(matched_states[0], result['state'])
        else:
            self.assertEqual(result['state'], 'no_trade')
            
        # Verify STRONG_BULLISH is matched (mdia_strong + whale_sponsored + macro_bullish)
        strong_bullish_rule = next(t for t in trace if t['state'] == 'strong_bullish')
        self.assertTrue(strong_bullish_rule['matched'])
        self.assertIn("mdia_strong=True", strong_bullish_rule['details'])
        self.assertIn("whale_sponsored=True", strong_bullish_rule['details'])
        self.assertIn("macro_bullish=True", strong_bullish_rule['details'])

    @patch('pandas.read_csv')
    @patch.object(Path, 'exists', return_value=True)
    def test_fusion_explain_bear_mode(self, mock_exists, mock_read_csv):
        mock_read_csv.return_value = self.mock_df.copy()

        url = reverse('api:fusion-explain')
        response = self.client.get(url, {'date': '2024-01-02'})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        result = response.data['result']
        trace = result['trace']
        
        # Bear trace has 5 states
        self.assertEqual(len(trace), 5)
        
        expected_order = [
            "bear_exhaustion_long", "bear_rally_long", "bear_continuation_short", 
            "late_distribution_short", "transition_chop"
        ]
        actual_order = [t['state'] for t in trace]
        self.assertEqual(actual_order, expected_order)

    @patch('pandas.read_csv')
    @patch.object(Path, 'exists', return_value=True)
    def test_fusion_explain_missing_date(self, mock_exists, mock_read_csv):
        mock_read_csv.return_value = self.mock_df.copy()
        
        url = reverse('api:fusion-explain')
        response = self.client.get(url)  # Missing date param
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch('pandas.read_csv')
    @patch.object(Path, 'exists', return_value=True)
    def test_fusion_explain_date_not_found(self, mock_exists, mock_read_csv):
        mock_read_csv.return_value = self.mock_df.copy()
        
        url = reverse('api:fusion-explain')
        response = self.client.get(url, {'date': '2025-01-01'})
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
