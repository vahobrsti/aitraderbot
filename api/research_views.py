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
        # Load the raw features CSV (needed for fuse_signals which requires cycle_days_since_halving)
        csv_path = Path(settings.BASE_DIR) / 'features_14d_5pct.csv'
        if not csv_path.exists():
            return Response(
                {"error": f"Feature CSV not found at {csv_path}"},
                status=status.HTTP_404_NOT_FOUND
            )

        date_str = request.query_params.get('date')
        if not date_str:
            return Response(
                {"error": "'date' query param is required, e.g., '2024-11-20'"},
                status=status.HTTP_400_BAD_REQUEST
            )
            
        import pandas as pd
        from signals.fusion import fuse_signals
        
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            target_date = pd.to_datetime(date_str)
            
            # Normalize index for comparison
            df.index = pd.to_datetime(df.index).normalize()
            target_date = target_date.normalize()
            
            if target_date not in df.index:
                return Response(
                    {"error": f"Date {date_str} not found in feature dataset"},
                    status=status.HTTP_404_NOT_FOUND
                )
                
            # Use the raw features row (has cycle_days_since_halving for bear mode detection)
            row = df.loc[target_date]
            fusion_result = fuse_signals(row)
            
            # Fetch price data from RawDailyData
            from datafeed.models import RawDailyData
            try:
                raw_data = RawDailyData.objects.get(date=target_date.date())
                price_context = {
                    "btc_close": raw_data.btc_close,
                    "mvrv_usd_7d": raw_data.mvrv_usd_7d,
                    "mvrv_usd_30d": raw_data.mvrv_usd_30d,
                }
            except RawDailyData.DoesNotExist:
                price_context = {
                    "btc_close": None,
                    "mvrv_usd_7d": None,
                    "mvrv_usd_30d": None,
                }
            
            from signals.fusion import build_explain_trace
            trace = build_explain_trace(row, fusion_result)

            return Response({
                "meta": {
                    "date": date_str,
                    "model_version": "v1-static"
                },
                "price_context": price_context,
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
