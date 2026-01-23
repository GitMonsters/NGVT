"""
NGVT Validation Infrastructure Example

This example demonstrates how to use the complete NGVT validation infrastructure
to assess and validate an AI model for production readiness.

Usage:
    python examples/validation_example.py
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation import (
    RealWorldARCValidator,
    CompetitivePerformanceAnalyzer,
    ErrorAnalysisSystem,
    TemporalStabilityValidator,
    DeploymentStressTester
)


class ExampleNGVTModel:
    """
    Example NGVT model for demonstration
    
    Replace this with your actual NGVT model implementation
    """
    
    def __init__(self, accuracy_level: float = 0.35):
        """
        Initialize example model
        
        Args:
            accuracy_level: Simulated accuracy (0.0 to 1.0)
        """
        self.accuracy_level = accuracy_level
        self.call_count = 0
        print(f"Initialized ExampleNGVTModel with {accuracy_level:.0%} accuracy")
    
    def solve_arc_problem(self, problem):
        """
        Solve an ARC problem
        
        Args:
            problem: ARC problem dictionary with 'train' and 'test'
            
        Returns:
            Solution grid (2D list)
        """
        self.call_count += 1
        
        # Simulate solving with configured accuracy
        if np.random.random() < self.accuracy_level:
            # Return correct answer
            if 'test' in problem and len(problem['test']) > 0:
                return problem['test'][0].get('output', [[1, 1]])
        
        # Return wrong answer
        return [[0, 0]]
    
    def predict(self, input_data):
        """
        Generic prediction interface
        
        Args:
            input_data: Input tensor/array
            
        Returns:
            Model output
        """
        self.call_count += 1
        
        # Simple transformation
        if isinstance(input_data, np.ndarray):
            return input_data * 0.5 + np.random.randn(*input_data.shape) * 0.1
        
        return input_data


def print_section_header(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_complete_validation():
    """
    Run complete validation pipeline
    
    This demonstrates all five validation components working together
    to provide comprehensive system assessment.
    """
    print_section_header("NGVT VALIDATION INFRASTRUCTURE - COMPLETE EXAMPLE")
    
    # Initialize model
    print("\n🤖 Initializing NGVT Model...")
    model = ExampleNGVTModel(accuracy_level=0.35)
    
    # ========================================================================
    # 1. ARC VALIDATION
    # ========================================================================
    print_section_header("1. REAL-WORLD ARC VALIDATION")
    
    print("\n📊 Running ARC dataset validation...")
    print("   This tests abstract reasoning and pattern recognition capabilities.")
    
    arc_validator = RealWorldARCValidator()
    arc_result = arc_validator.validate_system(model, max_problems=10)
    
    print(f"\n✅ Validation Complete!")
    print(f"   Accuracy: {arc_result.overall_accuracy:.2%}")
    print(f"   Problems Solved: {arc_result.solved_problems}/{arc_result.total_problems}")
    print(f"   Performance Tier: {arc_validator.get_performance_tier(arc_result.overall_accuracy)}")
    
    # Generate detailed report
    print("\n📄 Detailed Report:")
    print(arc_validator.generate_report(arc_result))
    
    # ========================================================================
    # 2. COMPETITIVE ANALYSIS
    # ========================================================================
    print_section_header("2. COMPETITIVE PERFORMANCE ANALYSIS")
    
    print("\n📈 Analyzing competitive standing...")
    print("   Comparing against state-of-the-art systems (GPT-4, Claude-3, etc.)")
    
    analyzer = CompetitivePerformanceAnalyzer()
    comp_result = analyzer.analyze_performance(
        accuracy=arc_result.overall_accuracy,
        system_name="NGVT-Example"
    )
    
    print(f"\n✅ Analysis Complete!")
    print(f"   Ranking: {comp_result.our_rank} out of {comp_result.total_systems} systems")
    print(f"   Percentile: {comp_result.percentile:.1f}th")
    print(f"   Competitive Tier: {comp_result.competitive_tier}")
    print(f"   Beating: {comp_result.beating_systems} competing systems")
    
    # Show improvement targets
    targets = analyzer.get_improvement_targets(arc_result.overall_accuracy)
    if targets:
        print(f"\n🎯 Near-term Improvement Targets:")
        for name, acc, gap in targets[:3]:
            print(f"   • {name}: {acc:.2%} (gap: {gap:.2%})")
    
    # Generate detailed report
    print("\n📄 Detailed Report:")
    print(analyzer.generate_competitive_report(comp_result, "NGVT-Example"))
    
    # ========================================================================
    # 3. ERROR ANALYSIS
    # ========================================================================
    print_section_header("3. COMPREHENSIVE ERROR ANALYSIS")
    
    print("\n🔍 Analyzing errors and failure patterns...")
    print("   Identifying root causes and providing recommendations.")
    
    error_system = ErrorAnalysisSystem()
    error_result = error_system.analyze_errors(
        errors=arc_result.errors,
        accuracies=arc_result.problem_accuracies,
        execution_times=arc_result.execution_times
    )
    
    print(f"\n✅ Analysis Complete!")
    print(f"   Total Errors: {error_result.total_errors}")
    print(f"   Error Rate: {error_result.error_rate:.2%}")
    
    if error_result.error_categories:
        print(f"\n📊 Error Categories:")
        for category, count in sorted(error_result.error_categories.items(), 
                                      key=lambda x: x[1], 
                                      reverse=True):
            print(f"   • {category}: {count}")
    
    if error_result.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in error_result.recommendations:
            print(f"   {rec}")
    
    # Generate detailed report
    print("\n📄 Detailed Report:")
    print(error_system.generate_error_report(error_result))
    
    # ========================================================================
    # 4. TEMPORAL STABILITY TESTING
    # ========================================================================
    print_section_header("4. TEMPORAL STABILITY VALIDATION")
    
    print("\n⏱️  Testing temporal consistency and stability...")
    print("   Ensuring reproducible and stable behavior over time.")
    
    stability_validator = TemporalStabilityValidator()
    test_input = np.random.randn(10, 10)
    
    stability_result = stability_validator.validate_stability(
        model=model,
        test_input=test_input,
        num_runs=5,
        time_interval=0.5
    )
    
    print(f"\n✅ Stability Test Complete!")
    print(f"   Stability Score: {stability_result.stability_score:.2%}")
    print(f"   Consistency Score: {stability_result.consistency_score:.2%}")
    print(f"   Reproducibility: {stability_result.reproducibility_score:.2%}")
    print(f"   Stability Tier: {stability_result.stability_tier}")
    print(f"   Drift Detected: {'Yes ⚠️' if stability_result.drift_detected else 'No ✓'}")
    
    # Generate detailed report
    print("\n📄 Detailed Report:")
    print(stability_validator.generate_stability_report(stability_result))
    
    # ========================================================================
    # 5. DEPLOYMENT STRESS TESTING
    # ========================================================================
    print_section_header("5. DEPLOYMENT STRESS TESTING")
    
    print("\n🚀 Running production stress tests...")
    print("   Testing system behavior under load conditions.")
    
    stress_tester = DeploymentStressTester(
        max_latency_ms=5000,
        max_memory_mb=8192
    )
    
    stress_result = stress_tester.run_comprehensive_stress_test(
        model=model,
        test_input=test_input,
        duration_seconds=10,
        concurrent_requests=5,
        request_rate=3
    )
    
    print(f"\n✅ Stress Test Complete!")
    print(f"   Deployment Readiness: {stress_result.readiness_classification}")
    print(f"   Stability Score: {stress_result.overall_stability_score:.2%}")
    print(f"   Total Requests: {stress_result.total_requests}")
    print(f"   Success Rate: {(1 - stress_result.error_rate):.2%}")
    print(f"   Throughput: {stress_result.throughput:.1f} req/s")
    print(f"   Avg Latency: {stress_result.average_latency:.2f} ms")
    
    # Generate detailed report
    print("\n📄 Detailed Report:")
    print(stress_tester.generate_stress_report(stress_result))
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_section_header("VALIDATION SUMMARY")
    
    print("\n📊 Overall Assessment:")
    print(f"\n   ARC Performance:")
    print(f"      • Accuracy: {arc_result.overall_accuracy:.2%}")
    print(f"      • Tier: {arc_validator.get_performance_tier(arc_result.overall_accuracy)}")
    
    print(f"\n   Competitive Position:")
    print(f"      • Rank: {comp_result.our_rank}/{comp_result.total_systems}")
    print(f"      • Tier: {comp_result.competitive_tier}")
    
    print(f"\n   Quality Metrics:")
    print(f"      • Error Rate: {error_result.error_rate:.2%}")
    print(f"      • Stability: {stability_result.stability_tier}")
    print(f"      • Deployment: {stress_result.readiness_classification}")
    
    # Overall readiness assessment
    print("\n🎯 Production Readiness Assessment:")
    
    readiness_score = 0
    max_score = 5
    
    if arc_result.overall_accuracy >= 0.15:
        readiness_score += 1
        print("   ✅ Meets minimum accuracy threshold (≥15%)")
    else:
        print("   ❌ Below minimum accuracy threshold")
    
    if comp_result.competitive_tier in ['WORLD-CLASS', 'EXCELLENT', 'STRONG', 'COMPETITIVE']:
        readiness_score += 1
        print("   ✅ Competitive performance level")
    else:
        print("   ⚠️  Below competitive baseline")
    
    if error_result.error_rate < 0.20:
        readiness_score += 1
        print("   ✅ Acceptable error rate")
    else:
        print("   ⚠️  High error rate")
    
    if stability_result.stability_score >= 0.70:
        readiness_score += 1
        print("   ✅ Adequate stability")
    else:
        print("   ⚠️  Stability issues detected")
    
    if stress_result.readiness_classification in ['PRODUCTION-READY', 'BETA-READY', 'ALPHA-READY']:
        readiness_score += 1
        print("   ✅ Deployment ready")
    else:
        print("   ❌ Not ready for deployment")
    
    print(f"\n📈 Overall Readiness Score: {readiness_score}/{max_score}")
    
    if readiness_score >= 4:
        print("\n🎉 SYSTEM READY FOR DEPLOYMENT!")
    elif readiness_score >= 3:
        print("\n✓  System shows promise - continue development")
    else:
        print("\n⚠️  Significant improvements needed")
    
    print("\n" + "=" * 70)
    print("  Validation Complete - Review reports above for details")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║      NGVT VALIDATION INFRASTRUCTURE EXAMPLE                  ║
    ║      Production-Grade Testing & Assessment                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        run_complete_validation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
