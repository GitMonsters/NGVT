"""
Performance Testing Suite

Comprehensive performance tests for NGVT validation infrastructure.
Tests all validation components for correctness and performance.
"""

import unittest
import sys
import time
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


class MockModel:
    """Mock model for testing purposes"""
    
    def __init__(self, accuracy: float = 0.5, fail_rate: float = 0.0):
        """
        Initialize mock model
        
        Args:
            accuracy: Simulated accuracy (0.0 to 1.0)
            fail_rate: Rate of failures (0.0 to 1.0)
        """
        self.accuracy = accuracy
        self.fail_rate = fail_rate
        self.call_count = 0
    
    def predict(self, input_data):
        """Mock prediction"""
        self.call_count += 1
        
        # Simulate failure
        if np.random.random() < self.fail_rate:
            raise RuntimeError("Simulated failure")
        
        # Return mock output
        if isinstance(input_data, dict):
            # ARC-style input
            if 'test' in input_data and len(input_data['test']) > 0:
                return input_data['test'][0].get('output', [[1, 1]])
        
        return np.random.randn(5, 5)
    
    def solve_arc_problem(self, problem):
        """Mock ARC problem solver"""
        self.call_count += 1
        
        # Simulate accuracy by randomly succeeding/failing
        if np.random.random() < self.accuracy:
            # Return correct answer
            if 'test' in problem and len(problem['test']) > 0:
                return problem['test'][0].get('output', [[1, 1]])
        
        # Return wrong answer
        return [[0, 0]]


class TestRealWorldARCValidator(unittest.TestCase):
    """Test suite for RealWorldARCValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = RealWorldARCValidator()
        self.model = MockModel(accuracy=0.7)
    
    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        self.assertIsNotNone(self.validator)
        self.assertIsNotNone(self.validator.data_dir)
    
    def test_create_sample_problems(self):
        """Test sample problem creation"""
        problems = self.validator._create_sample_problems()
        self.assertIsInstance(problems, dict)
        self.assertGreater(len(problems), 0)
        
        # Check problem structure
        for problem_id, problem in problems.items():
            self.assertIn('train', problem)
            self.assertIn('test', problem)
    
    def test_validate_system(self):
        """Test system validation"""
        problems = self.validator._create_sample_problems()
        result = self.validator.validate_system(self.model, problems, max_problems=2)
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.overall_accuracy, 0.0)
        self.assertLessEqual(result.overall_accuracy, 1.0)
        self.assertEqual(result.total_problems, 2)
        self.assertLessEqual(result.solved_problems, result.total_problems)
    
    def test_performance_tier(self):
        """Test performance tier classification"""
        tiers = [
            (0.90, "A+ (EXCEPTIONAL)"),
            (0.75, "A (EXCELLENT)"),
            (0.55, "B (GOOD)"),
            (0.35, "C (FAIR)"),
            (0.20, "D (BASIC)"),
            (0.05, "F (NEEDS IMPROVEMENT)")
        ]
        
        for accuracy, expected_tier in tiers:
            tier = self.validator.get_performance_tier(accuracy)
            self.assertEqual(tier, expected_tier)
    
    def test_generate_report(self):
        """Test report generation"""
        problems = self.validator._create_sample_problems()
        result = self.validator.validate_system(self.model, problems, max_problems=2)
        report = self.validator.generate_report(result)
        
        self.assertIsInstance(report, str)
        self.assertIn("ARC VALIDATION REPORT", report)
        self.assertIn("Performance Tier", report)
        self.assertIn("Accuracy", report)


class TestCompetitivePerformanceAnalyzer(unittest.TestCase):
    """Test suite for CompetitivePerformanceAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = CompetitivePerformanceAnalyzer()
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly"""
        self.assertIsNotNone(self.analyzer)
        self.assertGreater(len(self.analyzer.benchmarks), 0)
    
    def test_analyze_performance(self):
        """Test performance analysis"""
        result = self.analyzer.analyze_performance(0.40, "TestSystem")
        
        self.assertIsNotNone(result)
        self.assertEqual(result.our_accuracy, 0.40)
        self.assertGreater(result.our_rank, 0)
        self.assertGreater(result.total_systems, 0)
        self.assertGreaterEqual(result.percentile, 0.0)
        self.assertLessEqual(result.percentile, 100.0)
    
    def test_competitive_tier(self):
        """Test competitive tier classification"""
        tiers = [
            (0.90, "WORLD-CLASS"),
            (0.65, "EXCELLENT"),
            (0.45, "STRONG"),
            (0.30, "COMPETITIVE"),
            (0.20, "DEVELOPING"),
            (0.05, "BASELINE")
        ]
        
        for accuracy, expected_tier in tiers:
            tier = self.analyzer._get_competitive_tier(accuracy, 50.0)
            self.assertEqual(tier, expected_tier)
    
    def test_improvement_targets(self):
        """Test improvement target identification"""
        targets = self.analyzer.get_improvement_targets(0.30)
        
        self.assertIsInstance(targets, list)
        for name, acc, gap in targets:
            self.assertGreater(acc, 0.30)
            self.assertGreater(gap, 0.0)
    
    def test_generate_competitive_report(self):
        """Test competitive report generation"""
        result = self.analyzer.analyze_performance(0.35, "TestSystem")
        report = self.analyzer.generate_competitive_report(result, "TestSystem")
        
        self.assertIsInstance(report, str)
        self.assertIn("COMPETITIVE PERFORMANCE ANALYSIS", report)
        self.assertIn("Rank", report)


class TestErrorAnalysisSystem(unittest.TestCase):
    """Test suite for ErrorAnalysisSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.error_system = ErrorAnalysisSystem()
    
    def test_system_initialization(self):
        """Test error system initializes correctly"""
        self.assertIsNotNone(self.error_system)
        self.assertGreater(len(self.error_system.ERROR_CATEGORIES), 0)
    
    def test_categorize_errors(self):
        """Test error categorization"""
        errors = {
            'prob1': 'Shape mismatch error',
            'prob2': 'Timeout exceeded',
            'prob3': 'Memory error occurred',
            'prob4': 'Type error in input'
        }
        
        categories = self.error_system._categorize_errors(errors)
        
        self.assertGreater(len(categories), 0)
        self.assertIn('shape_mismatch', categories)
        self.assertIn('timeout', categories)
        self.assertIn('memory_error', categories)
        self.assertIn('type_error', categories)
    
    def test_analyze_errors(self):
        """Test comprehensive error analysis"""
        errors = {
            'prob1': 'Shape mismatch',
            'prob2': 'Timeout error',
        }
        accuracies = {
            'prob1': 0.0,
            'prob2': 0.0,
            'prob3': 1.0,
        }
        times = {
            'prob1': 0.5,
            'prob2': 10.0,
            'prob3': 0.3,
        }
        
        result = self.error_system.analyze_errors(errors, accuracies, times)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.total_errors, 2)
        self.assertGreater(result.error_rate, 0.0)
        self.assertGreater(len(result.error_categories), 0)
    
    def test_generate_error_report(self):
        """Test error report generation"""
        errors = {'prob1': 'Test error'}
        accuracies = {'prob1': 0.0, 'prob2': 1.0}
        times = {'prob1': 0.5, 'prob2': 0.3}
        
        result = self.error_system.analyze_errors(errors, accuracies, times)
        report = self.error_system.generate_error_report(result)
        
        self.assertIsInstance(report, str)
        self.assertIn("ERROR ANALYSIS REPORT", report)


class TestTemporalStabilityValidator(unittest.TestCase):
    """Test suite for TemporalStabilityValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = TemporalStabilityValidator()
        self.model = MockModel(accuracy=0.8)
    
    def test_validator_initialization(self):
        """Test validator initializes correctly"""
        self.assertIsNotNone(self.validator)
        self.assertGreater(self.validator.max_variance, 0.0)
        self.assertGreater(self.validator.min_runs, 0)
    
    def test_validate_stability(self):
        """Test stability validation"""
        test_input = np.random.randn(5, 5)
        result = self.validator.validate_stability(
            self.model, test_input, num_runs=3, time_interval=0.1
        )
        
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.stability_score, 0.0)
        self.assertLessEqual(result.stability_score, 1.0)
        self.assertGreaterEqual(result.test_runs, 0)
    
    def test_stability_tier(self):
        """Test stability tier classification"""
        tiers = [
            (0.96, "PRODUCTION-READY"),
            (0.88, "STABLE"),
            (0.75, "MODERATELY-STABLE"),
            (0.55, "UNSTABLE"),
            (0.30, "HIGHLY-UNSTABLE")
        ]
        
        for score, expected_tier in tiers:
            tier = self.validator._get_stability_tier(score)
            self.assertEqual(tier, expected_tier)
    
    def test_generate_stability_report(self):
        """Test stability report generation"""
        test_input = np.random.randn(5, 5)
        result = self.validator.validate_stability(
            self.model, test_input, num_runs=3, time_interval=0.1
        )
        report = self.validator.generate_stability_report(result)
        
        self.assertIsInstance(report, str)
        self.assertIn("TEMPORAL STABILITY REPORT", report)


class TestDeploymentStressTester(unittest.TestCase):
    """Test suite for DeploymentStressTester"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tester = DeploymentStressTester()
        self.model = MockModel(accuracy=0.8)
    
    def test_tester_initialization(self):
        """Test tester initializes correctly"""
        self.assertIsNotNone(self.tester)
        self.assertGreater(self.tester.max_latency_ms, 0)
        self.assertGreater(self.tester.max_memory_mb, 0)
    
    def test_comprehensive_stress_test(self):
        """Test comprehensive stress testing"""
        test_input = np.random.randn(5, 5)
        result = self.tester.run_comprehensive_stress_test(
            self.model, test_input,
            duration_seconds=5,
            concurrent_requests=3,
            request_rate=2
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(result.total_requests, 0)
        self.assertGreaterEqual(result.overall_stability_score, 0.0)
        self.assertLessEqual(result.overall_stability_score, 1.0)
    
    def test_readiness_classification(self):
        """Test readiness classification"""
        classifications = [
            (0.95, 0.005, 500, "PRODUCTION-READY"),
            (0.80, 0.04, 2000, "BETA-READY"),
            (0.65, 0.08, 3000, "ALPHA-READY"),
        ]
        
        for stability, error_rate, latency, expected in classifications:
            readiness = self.tester._classify_readiness(stability, error_rate, latency)
            self.assertEqual(readiness, expected)
    
    def test_generate_stress_report(self):
        """Test stress report generation"""
        test_input = np.random.randn(5, 5)
        result = self.tester.run_comprehensive_stress_test(
            self.model, test_input,
            duration_seconds=3,
            concurrent_requests=2,
            request_rate=1
        )
        report = self.tester.generate_stress_report(result)
        
        self.assertIsInstance(report, str)
        self.assertIn("DEPLOYMENT STRESS TEST REPORT", report)


class TestIntegration(unittest.TestCase):
    """Integration tests for full validation pipeline"""
    
    def test_full_validation_pipeline(self):
        """Test complete validation workflow"""
        model = MockModel(accuracy=0.6)
        
        # 1. ARC Validation
        arc_validator = RealWorldARCValidator()
        problems = arc_validator._create_sample_problems()
        arc_result = arc_validator.validate_system(model, problems, max_problems=2)
        self.assertIsNotNone(arc_result)
        
        # 2. Competitive Analysis
        analyzer = CompetitivePerformanceAnalyzer()
        comp_result = analyzer.analyze_performance(arc_result.overall_accuracy)
        self.assertIsNotNone(comp_result)
        
        # 3. Error Analysis
        error_system = ErrorAnalysisSystem()
        error_result = error_system.analyze_errors(
            arc_result.errors,
            arc_result.problem_accuracies,
            arc_result.execution_times
        )
        self.assertIsNotNone(error_result)
        
        # 4. Temporal Stability
        stability_validator = TemporalStabilityValidator()
        test_input = np.random.randn(5, 5)
        stability_result = stability_validator.validate_stability(
            model, test_input, num_runs=3, time_interval=0.1
        )
        self.assertIsNotNone(stability_result)
        
        # 5. Stress Testing
        stress_tester = DeploymentStressTester()
        stress_result = stress_tester.run_comprehensive_stress_test(
            model, test_input,
            duration_seconds=3,
            concurrent_requests=2,
            request_rate=1
        )
        self.assertIsNotNone(stress_result)
        
        # Verify all results are valid
        self.assertGreaterEqual(arc_result.overall_accuracy, 0.0)
        self.assertGreater(comp_result.total_systems, 0)
        self.assertGreaterEqual(error_result.error_rate, 0.0)
        self.assertGreaterEqual(stability_result.stability_score, 0.0)
        self.assertGreater(stress_result.total_requests, 0)


def run_performance_tests():
    """Run all performance tests"""
    print("=" * 70)
    print("NGVT VALIDATION INFRASTRUCTURE - PERFORMANCE TEST SUITE")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRealWorldARCValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestCompetitivePerformanceAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorAnalysisSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalStabilityValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestDeploymentStressTester))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_performance_tests()
    sys.exit(exit_code)
