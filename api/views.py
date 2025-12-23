"""
REST API views.
All API endpoints are defined here.
"""
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


class HealthCheckView(APIView):
    """
    GET /api/v1/health/
    
    Health check endpoint (no auth required).
    """
    authentication_classes = []
    permission_classes = []
    
    def get(self, request):
        return Response({"status": "ok"})
