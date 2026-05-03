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
    queryset = DailySignal.objects.all().order_by('-date')
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
            signal = DailySignal.objects.latest('date')
            serializer = DailySignalSerializer(signal)
            return Response(serializer.data)
        except DailySignal.DoesNotExist:
            return Response(
                {"error": "No signals found"},
                status=status.HTTP_404_NOT_FOUND
            )


class DailySignalDetailView(generics.RetrieveAPIView):
    """
    GET /api/v1/signals/<date>/
    
    Get signal for a specific date.
    Date format: YYYY-MM-DD
    """
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]
    queryset = DailySignal.objects.all()
    serializer_class = DailySignalSerializer
    lookup_field = 'date'


class TradeSetupView(APIView):
    """
    GET /api/v1/signals/<date>/setup/
    
    Get complete trade setup for a signal date.
    Includes option legs, metrics, position sizing, exit rules, and validation.
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
        
        # Check signal exists
        try:
            signal = DailySignal.objects.get(date=signal_date)
        except DailySignal.DoesNotExist:
            return Response(
                {"error": f"No signal found for {date}"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Skip NO_TRADE
        if signal.trade_decision == "NO_TRADE":
            return Response(
                {"error": "No trade setup for NO_TRADE signal", "signal_type": "NO_TRADE"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Build setup
        builder = TradeSetupBuilder()
        setup = builder.build_setup(signal_date)
        
        if setup is None:
            return Response(
                {"error": "Could not build trade setup. No option data available."},
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
        
        # Find latest tradeable signal
        signal = DailySignal.objects.exclude(
            trade_decision="NO_TRADE"
        ).order_by('-date').first()
        
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
