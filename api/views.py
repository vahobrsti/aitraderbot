"""
REST API views.
All API endpoints are defined here.
"""
from datetime import datetime

from rest_framework import generics, status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from signals.models import DailySignal
from .serializers import DailySignalSerializer, DailySignalSummarySerializer


class DailySignalListView(generics.ListAPIView):
    """
    GET /api/v1/signals/
    
    List all daily signals, paginated.
    Returns summary view for efficiency.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = DailySignal.objects.all().order_by('-date', '-updated_at')
    serializer_class = DailySignalSummarySerializer


class DailySignalLatestView(APIView):
    """
    GET /api/v1/signals/latest/
    
    Get the most recent signal.
    Returns full detail view.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        try:
            # Get latest date with tradeable signals, pick by priority
            latest_tradeable = DailySignal.objects.exclude(
                trade_decision="NO_TRADE"
            ).order_by('-date').first()
            if latest_tradeable:
                candidates = DailySignal.objects.filter(
                    date=latest_tradeable.date
                ).exclude(trade_decision="NO_TRADE")
                signal = DailySignal.pick_highest_priority(candidates)
            else:
                signal = DailySignal.objects.order_by('-date').first()
            if signal is None:
                raise DailySignal.DoesNotExist
            serializer = DailySignalSerializer(signal)
            return Response(serializer.data)
        except DailySignal.DoesNotExist:
            return Response(
                {"error": "No signals found"},
                status=status.HTTP_404_NOT_FOUND
            )


class DailySignalDetailView(generics.ListAPIView):
    """
    GET /api/v1/signals/<date>/
    
    Get all signals for a specific date (may be multiple trade types).
    Date format: YYYY-MM-DD
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    serializer_class = DailySignalSerializer

    def get_queryset(self):
        date_str = self.kwargs['date']
        return DailySignal.objects.filter(date=date_str).order_by('-updated_at')


class TradeSetupView(APIView):
    """
    GET /api/v1/signals/<date>/setup/
    GET /api/v1/signals/<date>/setup/?type=IRON_CONDOR
    
    Get complete trade setup for a signal date.
    Includes option legs, metrics, position sizing, exit rules, and validation.
    
    Query params:
        type: Override signal type (e.g., IRON_CONDOR, MVRV_SHORT)
              If not provided, uses the stored trade_decision from the signal.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request, date):
        from execution.services.trade_setup import TradeSetupBuilder
        
        try:
            signal_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return Response(
                {"error": "Invalid date format. Use YYYY-MM-DD"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get signal type from query param first
        signal_type = request.query_params.get('type', None)
        builder = TradeSetupBuilder()
        if signal_type:
            signal_type = signal_type.upper()
            # Validate against supported signal types
            direction, _ = builder._get_direction_and_type(signal_type)
            if direction is None:
                return Response(
                    {"error": f"Unsupported signal type: {signal_type}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        # Load signal for that date (filtered by type if provided)
        try:
            if signal_type:
                signal = DailySignal.objects.get(date=signal_date, trade_decision=signal_type)
            else:
                candidates = DailySignal.objects.filter(date=signal_date).exclude(
                    trade_decision="NO_TRADE"
                )
                signal = DailySignal.pick_highest_priority(candidates)
                if signal is None:
                    raise DailySignal.DoesNotExist
        except DailySignal.DoesNotExist:
            return Response(
                {"error": f"No signal found for {date}"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Use stored decision if no type override
        if not signal_type:
            signal_type = signal.trade_decision
        
        # Skip NO_TRADE
        if signal_type == "NO_TRADE":
            return Response(
                {"error": "No trade setup for NO_TRADE signal", "signal_type": "NO_TRADE"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Build setup with optional type override
        setup = builder.build_setup(signal_date, signal_type=signal_type)
        
        if setup is None:
            return Response(
                {"error": f"Could not build trade setup for {signal_type}. No option data available or signal type not supported."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        return Response(setup.to_dict())


class TradeSetupLatestView(APIView):
    """
    GET /api/v1/signals/latest/setup/
    
    Get trade setup for the most recent tradeable signal.
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        from execution.services.trade_setup import TradeSetupBuilder
        
        # Find latest tradeable signal by priority
        latest_tradeable = DailySignal.objects.exclude(
            trade_decision="NO_TRADE"
        ).order_by('-date').first()
        
        if not latest_tradeable:
            return Response(
                {"error": "No tradeable signals found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        candidates = DailySignal.objects.filter(
            date=latest_tradeable.date
        ).exclude(trade_decision="NO_TRADE")
        signal = DailySignal.pick_highest_priority(candidates)
        
        if not signal:
            return Response(
                {"error": "No tradeable signals found"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        builder = TradeSetupBuilder()
        setup = builder.build_setup(signal.date)
        
        if setup is None:
            return Response(
                {"error": "Could not build trade setup. No option data available."},
                status=status.HTTP_404_NOT_FOUND
            )
        
        return Response(setup.to_dict())


class HealthCheckView(APIView):
    """
    GET /api/v1/health/
    
    Health check endpoint (no auth required).
    """
    authentication_classes = []
    permission_classes = []
    
    def get(self, request):
        return Response({"status": "ok"})


class OptionPricePredict(APIView):
    """
    POST /api/v1/options/predict/
    
    Predict option price under different BTC scenarios.
    Uses the trained LeveragePredictor model to estimate option returns.
    
    Request body:
    {
        "current_spot": 80000,           # Current BTC price
        "strike": 79000,                 # Option strike price
        "option_type": "put",            # "call" or "put"
        "dte": 8,                        # Days to expiry
        "entry_premium": 2000,           # Premium paid at entry (per BTC)
        "current_premium": 720,          # Current option value (optional, for existing position)
        "scenarios": [77000, 75000, 73000],  # BTC prices to simulate
        "iv": 0.45                       # Current IV estimate (optional, default 0.50)
    }
    
    Response:
    {
        "current": { ... },
        "scenarios": [
            {
                "btc_price": 77000,
                "btc_change_pct": -3.75,
                "moneyness": "itm",
                "predicted_return": { "p10": ..., "p50": ..., "p90": ... },
                "estimated_premium": { "conservative": ..., "base": ..., "optimistic": ... },
                "intrinsic_value": 2000,
                "black_scholes": { "iv_40": ..., "iv_50": ..., "iv_60": ... }
            },
            ...
        ]
    }
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        from pathlib import Path
        from execution.services.option_pricer import LeveragePredictor, GBMPredictor, BlackScholes
        
        # Parse request
        data = request.data
        
        # Required fields
        current_spot = data.get('current_spot')
        strike = data.get('strike')
        option_type = data.get('option_type', 'put').lower()
        dte = data.get('dte')
        
        if not all([current_spot, strike, dte]):
            return Response(
                {"error": "Missing required fields: current_spot, strike, dte"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Optional fields
        entry_premium = data.get('entry_premium')
        current_premium = data.get('current_premium', entry_premium)
        scenarios = data.get('scenarios', [])
        iv = data.get('iv', 0.50)
        
        if not scenarios:
            # Default: simulate ±3%, ±5%, ±8% moves
            scenarios = [
                current_spot * 0.92,
                current_spot * 0.95,
                current_spot * 0.97,
                current_spot * 1.03,
                current_spot * 1.05,
                current_spot * 1.08,
            ]
        
        # Load predictor - prefer GBM if available, fall back to bucket model
        gbm_path = Path('models/option_response_gbm.joblib')
        bucket_path = Path('models/option_response_predictor_v2.json')
        
        if gbm_path.exists():
            predictor = GBMPredictor(model_path=gbm_path)
            model_type = "gbm"
        elif bucket_path.exists():
            predictor = LeveragePredictor(model_path=bucket_path)
            model_type = "bucket"
        else:
            predictor = LeveragePredictor()  # Will use synthetic fallback
            model_type = "synthetic"
        
        # Current position info
        current_moneyness = (strike - current_spot) / current_spot
        current_moneyness_label = self._moneyness_label(current_moneyness, option_type)
        
        result = {
            "current": {
                "spot": current_spot,
                "strike": strike,
                "option_type": option_type,
                "dte": dte,
                "moneyness_pct": round(current_moneyness * 100, 2),
                "moneyness_label": current_moneyness_label,
                "iv": iv,
                "entry_premium": entry_premium,
                "current_premium": current_premium,
                "intrinsic": max(strike - current_spot, 0) if option_type == 'put' else max(current_spot - strike, 0),
            },
            "scenarios": [],
            "model_info": {
                "model_type": model_type,
                "model_loaded": predictor.is_fitted,
                "n_samples": predictor.n_samples if hasattr(predictor, 'n_samples') else len(getattr(predictor, 'bucket_stats', {})),
            }
        }
        
        # Process each scenario
        for target_spot in sorted(scenarios):
            scenario = self._predict_scenario(
                predictor=predictor,
                current_spot=current_spot,
                target_spot=target_spot,
                strike=strike,
                option_type=option_type,
                dte=dte,
                iv=iv,
                current_premium=current_premium,
            )
            result["scenarios"].append(scenario)
        
        return Response(result)
    
    def _predict_scenario(
        self,
        predictor,
        current_spot: float,
        target_spot: float,
        strike: float,
        option_type: str,
        dte: int,
        iv: float,
        current_premium: float,
    ) -> dict:
        from execution.services.option_pricer import BlackScholes
        
        btc_change_pct = (target_spot - current_spot) / current_spot
        target_moneyness = (strike - target_spot) / target_spot
        
        # Get model prediction - use TARGET moneyness for the scenario
        prediction = predictor.predict(
            dte=dte,
            moneyness=target_moneyness,  # Use target moneyness, not entry
            option_type=option_type,
            btc_change_pct=btc_change_pct,
            iv=iv,
            days_held=1,  # Assume 1-day horizon
        )
        
        # Calculate estimated premiums from prediction
        # Use 6 decimal places for BTC-denominated premiums (Deribit standard)
        estimated_premiums = {}
        if current_premium:
            estimated_premiums = {
                "conservative": round(current_premium * (1 + prediction.p10), 6),
                "base": round(current_premium * (1 + prediction.p50), 6),
                "optimistic": round(current_premium * (1 + prediction.p90), 6),
            }
        
        # Black-Scholes reference prices (in BTC)
        T = max(dte - 1, 1) / 365  # Assume 1 day passes
        bs_prices = {}
        for test_iv in [0.40, 0.50, 0.60, 0.70]:
            if option_type == 'put':
                bs_price_usd = BlackScholes.put_price(target_spot, strike, T, 0.05, test_iv)
            else:
                bs_price_usd = BlackScholes.call_price(target_spot, strike, T, 0.05, test_iv)
            # Convert to BTC
            bs_price_btc = bs_price_usd / target_spot if target_spot > 0 else 0
            bs_prices[f"iv_{int(test_iv*100)}"] = round(bs_price_btc, 6)
        
        # Intrinsic value in BTC (Deribit denomination)
        if option_type == 'put':
            intrinsic_usd = max(strike - target_spot, 0)
        else:
            intrinsic_usd = max(target_spot - strike, 0)
        intrinsic_btc = intrinsic_usd / target_spot if target_spot > 0 else 0
        
        return {
            "btc_price": round(target_spot, 2),
            "btc_change_pct": round(btc_change_pct * 100, 2),
            "moneyness_pct": round(target_moneyness * 100, 2),
            "moneyness_label": self._moneyness_label(target_moneyness, option_type),
            "predicted_return": {
                "p10": round(prediction.p10 * 100, 1),
                "p25": round(prediction.p25 * 100, 1),
                "p50": round(prediction.p50 * 100, 1),
                "p75": round(prediction.p75 * 100, 1),
                "p90": round(prediction.p90 * 100, 1),
            },
            "prediction_meta": {
                "dte_bucket": prediction.dte_bucket,
                "moneyness_bucket": prediction.moneyness_bucket,
                "regime": prediction.regime,
                "n_samples": prediction.n_samples,
            },
            "estimated_premium": estimated_premiums,
            "intrinsic_value_usd": round(intrinsic_usd, 2),
            "intrinsic_value_btc": round(intrinsic_btc, 6),
            "black_scholes": bs_prices,
        }
    
    def _moneyness_label(self, moneyness: float, option_type: str) -> str:
        """Convert moneyness to human-readable label."""
        # For puts: positive moneyness = OTM, negative = ITM
        # For calls: positive moneyness = OTM, negative = ITM
        abs_m = abs(moneyness)
        
        if option_type == 'put':
            is_itm = moneyness > 0  # strike > spot
        else:
            is_itm = moneyness < 0  # strike < spot
        
        if abs_m < 0.02:
            return "ATM"
        elif abs_m < 0.05:
            return "slightly ITM" if is_itm else "slightly OTM"
        elif abs_m < 0.10:
            return "ITM" if is_itm else "OTM"
        else:
            return "deep ITM" if is_itm else "deep OTM"
