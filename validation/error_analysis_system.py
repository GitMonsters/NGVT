"""
Error Analysis System

Provides comprehensive error analysis, diagnostics, and debugging capabilities
for AI system validation and testing.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """Represents a pattern of errors"""
    pattern_type: str
    frequency: int
    examples: List[str] = field(default_factory=list)
    severity: str = "MEDIUM"
    suggested_fix: str = ""


@dataclass
class ErrorAnalysisResult:
    """Results from error analysis"""
    total_errors: int
    error_rate: float
    error_categories: Dict[str, int] = field(default_factory=dict)
    error_patterns: List[ErrorPattern] = field(default_factory=list)
    severity_breakdown: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'total_errors': self.total_errors,
            'error_rate': self.error_rate,
            'error_categories': self.error_categories,
            'error_patterns': [
                {
                    'pattern_type': p.pattern_type,
                    'frequency': p.frequency,
                    'severity': p.severity,
                    'suggested_fix': p.suggested_fix
                }
                for p in self.error_patterns
            ],
            'severity_breakdown': self.severity_breakdown,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp,
        }


class ErrorAnalysisSystem:
    """
    Comprehensive error analysis and diagnostics system
    
    Analyzes errors from validation runs to identify:
    - Common error patterns
    - Root causes
    - Severity levels
    - Recommendations for fixes
    """
    
    ERROR_CATEGORIES = {
        'shape_mismatch': 'Output shape does not match expected',
        'value_error': 'Output values are incorrect',
        'timeout': 'Execution exceeded time limit',
        'memory_error': 'Out of memory during execution',
        'type_error': 'Incorrect data type',
        'runtime_error': 'Runtime exception during execution',
        'logic_error': 'Logical reasoning error',
        'pattern_recognition': 'Failed to recognize pattern',
        'generalization': 'Failed to generalize from examples',
    }
    
    def __init__(self):
        """Initialize error analysis system"""
        self.error_history = []
        self.pattern_cache = {}
    
    def analyze_errors(self, 
                      errors: Dict[str, str],
                      accuracies: Dict[str, float],
                      execution_times: Dict[str, float]) -> ErrorAnalysisResult:
        """
        Analyze errors from a validation run
        
        Args:
            errors: Dictionary of problem_id -> error_message
            accuracies: Dictionary of problem_id -> accuracy_score
            execution_times: Dictionary of problem_id -> execution_time
            
        Returns:
            ErrorAnalysisResult with comprehensive error analysis
        """
        logger.info(f"Analyzing {len(errors)} errors...")
        
        total_problems = len(accuracies)
        failed_problems = sum(1 for acc in accuracies.values() if acc < 1.0)
        error_rate = failed_problems / total_problems if total_problems > 0 else 0.0
        
        # Categorize errors
        error_categories = self._categorize_errors(errors)
        
        # Identify error patterns
        error_patterns = self._identify_patterns(errors, accuracies)
        
        # Analyze severity
        severity_breakdown = self._analyze_severity(errors, execution_times)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            error_categories, error_patterns, severity_breakdown
        )
        
        result = ErrorAnalysisResult(
            total_errors=len(errors),
            error_rate=error_rate,
            error_categories=error_categories,
            error_patterns=error_patterns,
            severity_breakdown=severity_breakdown,
            recommendations=recommendations
        )
        
        logger.info(f"Error analysis complete: {len(errors)} errors analyzed")
        
        return result
    
    def _categorize_errors(self, errors: Dict[str, str]) -> Dict[str, int]:
        """
        Categorize errors into predefined categories
        
        Args:
            errors: Dictionary of error messages
            
        Returns:
            Dictionary of category -> count
        """
        categories = defaultdict(int)
        
        for error_msg in errors.values():
            error_lower = error_msg.lower()
            
            # Check against known categories
            if 'shape' in error_lower or 'dimension' in error_lower:
                categories['shape_mismatch'] += 1
            elif 'timeout' in error_lower or 'time limit' in error_lower:
                categories['timeout'] += 1
            elif 'memory' in error_lower or 'memoryerror' in error_lower:
                categories['memory_error'] += 1
            elif 'type' in error_lower or 'typeerror' in error_lower:
                categories['type_error'] += 1
            elif 'value' in error_lower or 'valueerror' in error_lower:
                categories['value_error'] += 1
            elif 'pattern' in error_lower:
                categories['pattern_recognition'] += 1
            elif 'generali' in error_lower:
                categories['generalization'] += 1
            else:
                categories['runtime_error'] += 1
        
        return dict(categories)
    
    def _identify_patterns(self, 
                          errors: Dict[str, str],
                          accuracies: Dict[str, float]) -> List[ErrorPattern]:
        """
        Identify common error patterns
        
        Args:
            errors: Error messages
            accuracies: Accuracy scores
            
        Returns:
            List of identified error patterns
        """
        patterns = []
        
        # Pattern 1: Consistent failures (0% accuracy)
        zero_accuracy_problems = [
            pid for pid, acc in accuracies.items() if acc == 0.0
        ]
        if len(zero_accuracy_problems) > 5:
            patterns.append(ErrorPattern(
                pattern_type="Consistent Failures",
                frequency=len(zero_accuracy_problems),
                examples=zero_accuracy_problems[:3],
                severity="HIGH",
                suggested_fix="Review model architecture and training data"
            ))
        
        # Pattern 2: Timeout issues
        timeout_errors = [
            pid for pid, msg in errors.items() 
            if 'timeout' in msg.lower()
        ]
        if len(timeout_errors) > 3:
            patterns.append(ErrorPattern(
                pattern_type="Timeout Issues",
                frequency=len(timeout_errors),
                examples=timeout_errors[:3],
                severity="MEDIUM",
                suggested_fix="Optimize inference speed or increase timeout limits"
            ))
        
        # Pattern 3: Memory issues
        memory_errors = [
            pid for pid, msg in errors.items()
            if 'memory' in msg.lower()
        ]
        if len(memory_errors) > 0:
            patterns.append(ErrorPattern(
                pattern_type="Memory Issues",
                frequency=len(memory_errors),
                examples=memory_errors[:3],
                severity="HIGH",
                suggested_fix="Reduce model size or batch processing"
            ))
        
        # Pattern 4: Type mismatches
        type_errors = [
            pid for pid, msg in errors.items()
            if 'type' in msg.lower()
        ]
        if len(type_errors) > 2:
            patterns.append(ErrorPattern(
                pattern_type="Type Mismatches",
                frequency=len(type_errors),
                examples=type_errors[:3],
                severity="MEDIUM",
                suggested_fix="Add input/output type validation"
            ))
        
        return patterns
    
    def _analyze_severity(self,
                         errors: Dict[str, str],
                         execution_times: Dict[str, float]) -> Dict[str, int]:
        """
        Analyze error severity levels
        
        Args:
            errors: Error messages
            execution_times: Execution times
            
        Returns:
            Severity breakdown dictionary
        """
        severity = defaultdict(int)
        
        for pid, error_msg in errors.items():
            error_lower = error_msg.lower()
            
            # Critical: Memory errors, crashes
            if 'memory' in error_lower or 'crash' in error_lower:
                severity['CRITICAL'] += 1
            # High: Timeouts, exceptions
            elif 'timeout' in error_lower or 'exception' in error_lower:
                severity['HIGH'] += 1
            # Medium: Logic errors, value errors
            elif 'value' in error_lower or 'logic' in error_lower:
                severity['MEDIUM'] += 1
            # Low: Minor issues
            else:
                severity['LOW'] += 1
        
        return dict(severity)
    
    def _generate_recommendations(self,
                                 categories: Dict[str, int],
                                 patterns: List[ErrorPattern],
                                 severity: Dict[str, int]) -> List[str]:
        """
        Generate recommendations for fixing errors
        
        Args:
            categories: Error categories
            patterns: Error patterns
            severity: Severity breakdown
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # High-level recommendations based on error categories
        if categories.get('shape_mismatch', 0) > 5:
            recommendations.append(
                "⚠️  Multiple shape mismatch errors detected. "
                "Review output dimensions and ensure proper reshaping."
            )
        
        if categories.get('timeout', 0) > 3:
            recommendations.append(
                "⏱️  Timeout issues detected. "
                "Consider optimizing model inference or increasing time limits."
            )
        
        if categories.get('memory_error', 0) > 0:
            recommendations.append(
                "💾 Memory errors detected. "
                "Reduce batch size or model complexity."
            )
        
        # Pattern-based recommendations
        for pattern in patterns:
            if pattern.severity in ['HIGH', 'CRITICAL'] and pattern.suggested_fix:
                recommendations.append(f"🔧 {pattern.suggested_fix}")
        
        # Severity-based recommendations
        if severity.get('CRITICAL', 0) > 0:
            recommendations.append(
                "🚨 CRITICAL errors found. Immediate attention required!"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "✅ No major error patterns detected. "
                "Focus on improving accuracy through model refinement."
            )
        
        return recommendations
    
    def generate_error_report(self, result: ErrorAnalysisResult) -> str:
        """
        Generate comprehensive error analysis report
        
        Args:
            result: Error analysis results
            
        Returns:
            Formatted report string
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║            ERROR ANALYSIS REPORT                             ║
╚══════════════════════════════════════════════════════════════╝

Summary:
  • Total Errors: {result.total_errors}
  • Error Rate: {result.error_rate:.2%}

Error Categories:
"""
        
        for category, count in sorted(result.error_categories.items(), 
                                     key=lambda x: x[1], 
                                     reverse=True):
            report += f"  • {category}: {count}\n"
        
        report += f"\nSeverity Breakdown:\n"
        for severity, count in sorted(result.severity_breakdown.items()):
            report += f"  • {severity}: {count}\n"
        
        if result.error_patterns:
            report += f"\nIdentified Error Patterns:\n"
            for pattern in result.error_patterns:
                report += f"  • {pattern.pattern_type} ({pattern.frequency} occurrences)\n"
                report += f"    Severity: {pattern.severity}\n"
                if pattern.suggested_fix:
                    report += f"    Fix: {pattern.suggested_fix}\n"
        
        if result.recommendations:
            report += f"\nRecommendations:\n"
            for rec in result.recommendations:
                report += f"  {rec}\n"
        
        report += f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result.timestamp))}\n"
        
        return report
    
    def track_error_trends(self, 
                          current_result: ErrorAnalysisResult,
                          historical_results: List[ErrorAnalysisResult]) -> Dict[str, Any]:
        """
        Track error trends over time
        
        Args:
            current_result: Current error analysis
            historical_results: Previous error analyses
            
        Returns:
            Trend analysis
        """
        if not historical_results:
            return {'status': 'No historical data available'}
        
        # Calculate trends
        current_rate = current_result.error_rate
        prev_rate = historical_results[-1].error_rate
        
        rate_change = current_rate - prev_rate
        trend = "IMPROVING" if rate_change < 0 else "WORSENING" if rate_change > 0 else "STABLE"
        
        return {
            'trend': trend,
            'rate_change': rate_change,
            'current_rate': current_rate,
            'previous_rate': prev_rate,
            'percentage_change': (rate_change / prev_rate * 100) if prev_rate > 0 else 0,
        }
