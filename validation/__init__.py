"""
NGVT Validation Infrastructure
Production-Grade Validation and Testing System

This module provides comprehensive validation tools for NGVT including:
- Real-world ARC dataset validation
- Competitive performance analysis
- Error analysis and diagnostics
- Temporal stability testing
- Production stress testing
"""

from .real_world_arc_validator import RealWorldARCValidator
from .competitive_performance_analyzer import CompetitivePerformanceAnalyzer
from .error_analysis_system import ErrorAnalysisSystem
from .temporal_stability_validator import TemporalStabilityValidator
from .deployment_stress_tester import DeploymentStressTester

__all__ = [
    'RealWorldARCValidator',
    'CompetitivePerformanceAnalyzer',
    'ErrorAnalysisSystem',
    'TemporalStabilityValidator',
    'DeploymentStressTester',
]

__version__ = '1.0.0'
