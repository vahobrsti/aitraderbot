"""
API URL configuration.
All API routes are versioned under /api/v1/
"""
from django.urls import path

from . import views

app_name = 'api'

urlpatterns = [
    # Health check (no auth)
    path('v1/health/', views.HealthCheckView.as_view(), name='health'),
    
    # Signal endpoints (auth required)
    path('v1/signals/', views.DailySignalListView.as_view(), name='signal-list'),
    path('v1/signals/latest/', views.DailySignalLatestView.as_view(), name='signal-latest'),
    path('v1/signals/<str:date>/', views.DailySignalDetailView.as_view(), name='signal-detail'),
]
