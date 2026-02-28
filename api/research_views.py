"""
API views for the research pipeline.
"""
from pathlib import Path
from django.conf import settings
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
# Research modules are imported locally within methods to avoid 
# import errors at Django startup during route resolution.

class BaseResearchAPIView(APIView):
    """Base class for research API views providing data loading."""
    
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]

    def get_research_table(self, request):
        """Build the research table from a default or requested CSV."""
        # For now, default to the known CSV in the project root
        csv_path = Path(settings.BASE_DIR) / 'features_14d_5pct.csv'
        if not csv_path.exists():
            return None, Response(
                {"error": f"Feature CSV not found at {csv_path}"},
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            import pandas as pd
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # Apply year filter if requested
            year = request.query_params.get('year')
            if year and year.isdigit():
                df = df.loc[f'{year}-01-01':f'{year}-12-31']
            from signals.research.fusion_table import build_research_table
            rt = build_research_table(df)
            return rt, None
        except Exception as e:
            return None, Response(
                {"error": f"Failed to build research table: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def format_response(self, rt, results_df, filters=None):
        """Format the standardized JSON response."""
        import pandas as pd
        # Convert DataFrame to list of dicts, replacing NaN with None
        results = results_df.replace({pd.NA: None}).where(pd.notnull(results_df), None).to_dict('records')
        
        return Response({
            "meta": {
                "horizon": 14,
                "target": 0.05
            },
            "filters": filters or {},
            "summary": {
                "total_rows": len(rt),
                "date_range": [
                    rt.index.min().strftime('%Y-%m-%d') if not rt.empty else None,
                    rt.index.max().strftime('%Y-%m-%d') if not rt.empty else None
                ]
            },
            "results": results
        })


class MetricStatsView(BaseResearchAPIView):
    """GET /api/v1/fusion/analysis/metric-stats/"""
    
    def get(self, request):
        rt, error_response = self.get_research_table(request)
        if error_response: return error_response

        metric = request.query_params.get('metric')
        valid_metrics = ['mdia_bucket', 'whale_bucket', 'mvrv_ls_bucket']
        if not metric or metric not in valid_metrics:
            return Response(
                {"error": f"'metric' query param is required and must be one of {valid_metrics}"},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        from signals.research.stats import compute_bucket_stats    
        min_count = int(request.query_params.get('min_count', 10))
        stats = compute_bucket_stats(rt, metric, min_count=min_count)
        
        return self.format_response(
            rt, stats, 
            filters={"year": request.query_params.get('year'), "metric": metric, "min_count": min_count}
        )


class ComboStatsView(BaseResearchAPIView):
    """GET /api/v1/fusion/analysis/combo-stats/"""
    
    def get(self, request):
        rt, error_response = self.get_research_table(request)
        if error_response: return error_response

        combo = request.query_params.get('group_by')  # e.g., "mdia_bucket,whale_bucket"
        if not combo:
            return Response(
                {"error": "'group_by' query param is required, e.g., 'mdia_bucket,whale_bucket'"},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        from signals.research.stats import compute_combo_stats    
        group_cols = combo.split(',')
        min_count = int(request.query_params.get('min_count', 10))
        stats = compute_combo_stats(rt, group_cols, min_count=min_count)
        
        return self.format_response(
            rt, stats, 
            filters={"year": request.query_params.get('year'), "group_by": combo, "min_count": min_count}
        )


class StateStatsView(BaseResearchAPIView):
    """GET /api/v1/fusion/analysis/state-stats/"""
    
    def get(self, request):
        rt, error_response = self.get_research_table(request)
        if error_response: return error_response

        from signals.research.stats import compute_state_stats
        min_count = int(request.query_params.get('min_count', 10))
        stats = compute_state_stats(rt, min_count=min_count)
        
        return self.format_response(
            rt, stats, 
            filters={"year": request.query_params.get('year'), "min_count": min_count}
        )


class FusionExplainView(BaseResearchAPIView):
    """GET /api/v1/fusion/explain/"""
    
    def get(self, request):
        rt, error_response = self.get_research_table(request)
        if error_response: return error_response

        date_str = request.query_params.get('date')
        if not date_str:
            return Response(
                {"error": "'date' query param is required, e.g., '2024-11-20'"},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        import pandas as pd
        from signals.fusion import fuse_signals
        
        try:
            target_date = pd.to_datetime(date_str)
            # Find the row closest to or matching the date
            if target_date not in rt.index:
                return Response(
                    {"error": f"Date {date_str} not found in feature dataset"},
                    status=status.HTTP_404_NOT_FOUND
                )
                
            row = rt.loc[target_date]
            fusion_result = fuse_signals(row)
            
            c = fusion_result.components
            mdia_strong = c.get('mdia_strong', 0) == 1
            mdia_inflow = c.get('mdia_inflow', 0) == 1
            mdia_non_inflow = not mdia_inflow
            
            whale_sponsored = c.get('whale_sponsored', 0) == 1
            whale_mixed = c.get('whale_mixed', 0) == 1
            whale_distrib = c.get('whale_distrib', 0) == 1
            whale_distrib_strong = c.get('whale_distrib_strong', 0) == 1
            
            mvrv_macro_bullish = c.get('mvrv_macro_bullish', 0) == 1
            mvrv_recovery = row.get('mvrv_ls_regime_call_confirm_recovery', 0) == 1
            mvrv_macro_neutral = c.get('mvrv_macro_neutral', 0) == 1
            mvrv_put_or_bear = row.get('mvrv_ls_regime_put_confirm', 0) == 1 or row.get('mvrv_ls_regime_bear_continuation', 0) == 1
            
            trace = [
                {"state": "STRONG_BULLISH", "matched": bool(mdia_strong and whale_sponsored and mvrv_macro_bullish), "details": f"mdia_strong={mdia_strong}, whale_sponsored={whale_sponsored}, macro_bullish={mvrv_macro_bullish}"},
                {"state": "EARLY_RECOVERY", "matched": bool(mdia_inflow and whale_sponsored and mvrv_recovery), "details": f"mdia_inflow={mdia_inflow}, whale_sponsored={whale_sponsored}, mvrv_recovery={mvrv_recovery}"},
                {"state": "BEAR_CONTINUATION", "matched": bool(mdia_non_inflow and whale_distrib and mvrv_put_or_bear), "details": f"not_mdia_inflow={mdia_non_inflow}, whale_distrib={whale_distrib}, mvrv_put/bear={mvrv_put_or_bear}"},
                {"state": "BEAR_PROBE", "matched": bool(mdia_non_inflow and whale_distrib_strong and mvrv_macro_neutral), "details": f"not_mdia_inflow={mdia_non_inflow}, whale_distrib_strong={whale_distrib_strong}, macro_neutral={mvrv_macro_neutral}"},
                {"state": "DISTRIBUTION_RISK", "matched": bool(mdia_non_inflow and whale_distrib and not mvrv_macro_bullish), "details": f"not_mdia_inflow={mdia_non_inflow}, whale_distrib={whale_distrib}, not_macro_bullish={not mvrv_macro_bullish}"},
                {"state": "MOMENTUM_CONTINUATION", "matched": bool(mdia_inflow and (whale_sponsored or whale_mixed) and mvrv_macro_bullish), "details": f"mdia_inflow={mdia_inflow}, whale_sponsored/mixed={(whale_sponsored or whale_mixed)}, macro_bullish={mvrv_macro_bullish}"},
                {"state": "BULL_PROBE", "matched": bool(mdia_inflow and whale_sponsored and mvrv_macro_neutral), "details": f"mdia_inflow={mdia_inflow}, whale_sponsored={whale_sponsored}, macro_neutral={mvrv_macro_neutral}"},
            ]

            return Response({
                "meta": {
                    "date": date_str,
                    "model_version": "v1-static"
                },
                "result": {
                    "state": fusion_result.state.value,
                    "confidence": fusion_result.confidence.value,
                    "score": fusion_result.score,
                    "components": fusion_result.components,
                    "trace": trace
                }
            })
        except Exception as e:
            return Response(
                {"error": f"Failed to process explanation: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ScoreValidationView(BaseResearchAPIView):
    """GET /api/v1/fusion/analysis/score-validation/"""
    
    def get(self, request):
        rt, error_response = self.get_research_table(request)
        if error_response: return error_response

        from signals.research.reporting import validate_monotonicity
        min_count = int(request.query_params.get('min_count', 10))
        validate_type = request.query_params.get('type', 'monotonicity') # could be 'stability'
        
        mono = validate_monotonicity(rt, min_count=min_count)
        
        return self.format_response(
            rt, mono, 
            filters={"year": request.query_params.get('year'), "min_count": min_count, "type": validate_type}
        )
