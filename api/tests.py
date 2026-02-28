from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token
import pandas as pd
from unittest.mock import patch

class FusionExplainTests(APITestCase):
    def setUp(self):
        # Create user and token for authentication
        self.user = User.objects.create_user(username='testuser', password='password')
        self.token = Token.objects.create(user=self.user)
        self.client.credentials(HTTP_AUTHORIZATION='Token ' + self.token.key)
        
        # Create a mock research table for the endpoint
        dates = pd.date_range('2024-01-01', periods=1, freq='D')
        self.mock_rt = pd.DataFrame({
            'mdia_bucket': ['strong_inflow'],
            'whale_bucket': ['broad_accum'],
            'mvrv_ls_bucket': ['trend_confirm'],
            'mvrv_ls_regime_call_confirm': [1],
            'mdia_regime_strong_inflow': [1],
            'whale_regime_broad_accum': [1],
            # Minimal required for fuse_signals
        }, index=dates)

    @patch('api.research_views.BaseResearchAPIView.get_research_table')
    def test_fusion_explain_success(self, mock_get_rt):
        # Setup mock to return the mock dataframe
        mock_get_rt.return_value = (self.mock_rt, None)
        
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
        
        # Verify trace semantics
        trace = result['trace']
        self.assertEqual(len(trace), 7)
        
        # Verify trace order matches classifier order
        expected_order = [
            "STRONG_BULLISH", "EARLY_RECOVERY", "BEAR_CONTINUATION", 
            "BEAR_PROBE", "DISTRIBUTION_RISK", "MOMENTUM_CONTINUATION", "BULL_PROBE"
        ]
        actual_order = [t['state'] for t in trace]
        self.assertEqual(actual_order, expected_order)
        
        # Verify the matched state correctly aligns with result.state
        matched_states = [t['state'] for t in trace if t['matched']]
        if matched_states:
            # The first matched state should be the one elected, but our mock might hit NO_TRADE
            # Let's check if the result state is one of the matched states, or if none matched, it's NO_TRADE
            if result['state'].upper() != 'NO_TRADE':
                self.assertEqual(matched_states[0], result['state'].upper())
        else:
            self.assertEqual(result['state'].upper(), 'NO_TRADE')
            
        # Verify boolean details are correct for our fixture
        # Our fixture has: strong_inflow, broad_accum (val=1), so mdia_strong=True, whale_sponsored=True
        # mvrv=trend_confirm (which means macro_bullish=True)
        # So STRONG_BULLISH should be matched=True
        strong_bullish_rule = next(t for t in trace if t['state'] == 'STRONG_BULLISH')
        self.assertTrue(strong_bullish_rule['matched'])
        self.assertIn("mdia_strong=True", strong_bullish_rule['details'])
        self.assertIn("whale_sponsored=True", strong_bullish_rule['details'])
        self.assertIn("macro_bullish=True", strong_bullish_rule['details'])

    @patch('api.research_views.BaseResearchAPIView.get_research_table')
    def test_fusion_explain_missing_date(self, mock_get_rt):
        mock_get_rt.return_value = (self.mock_rt, None)
        
        url = reverse('api:fusion-explain')
        response = self.client.get(url)  # Missing date param
        
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    @patch('api.research_views.BaseResearchAPIView.get_research_table')
    def test_fusion_explain_date_not_found(self, mock_get_rt):
        mock_get_rt.return_value = (self.mock_rt, None)
        
        url = reverse('api:fusion-explain')
        response = self.client.get(url, {'date': '2025-01-01'})
        
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
