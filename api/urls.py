"""
API URL configuration.
All API routes are versioned under /api/v1/
"""
from django.urls import path

from . import views, research_views

app_name = 'api'

urlpatterns = [
    # Health check (no auth)
    path('v1/health/', views.HealthCheckView.as_view(), name='health'),
    
    # Signal endpoints (auth required)
    path('v1/signals/', views.DailySignalListView.as_view(), name='signal-list'),
    path('v1/signals/latest/', views.DailySignalLatestView.as_view(), name='signal-latest'),
    path('v1/signals/<str:date>/', views.DailySignalDetailView.as_view(), name='signal-detail'),
    
    # Research endpoints (auth required)
    path('v1/fusion/explain/', research_views.FusionExplainView.as_view(), name='fusion-explain'),
    path('v1/fusion/analysis/metric-stats/', research_views.MetricStatsView.as_view(), name='fusion-metric-stats'),
    path('v1/fusion/analysis/combo-stats/', research_views.ComboStatsView.as_view(), name='fusion-combo-stats'),
    path('v1/fusion/analysis/state-stats/', research_views.StateStatsView.as_view(), name='fusion-state-stats'),
    path('v1/fusion/analysis/score-validation/', research_views.ScoreValidationView.as_view(), name='fusion-score-validation'),
]
