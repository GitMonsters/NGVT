# NGVT Validation Infrastructure Guide

## Overview

The NGVT Validation Infrastructure provides production-grade testing and validation capabilities for AI systems. It includes comprehensive tools for performance assessment, competitive benchmarking, error analysis, stability testing, and deployment readiness evaluation.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Validation Components](#validation-components)
3. [Usage Examples](#usage-examples)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)
6. [API Reference](#api-reference)

## Quick Start

### Installation

Ensure you have all dependencies installed:

```bash
cd torusScode
pip install -r requirements_web.txt
```

### Basic Usage

```python
from validation import RealWorldARCValidator

# Create validator
validator = RealWorldARCValidator()

# Validate your model
result = validator.validate_system(your_model)

# View results
print(f"Accuracy: {result.overall_accuracy:.2%}")
print(f"Solved: {result.solved_problems}/{result.total_problems}")

# Generate report
report = validator.generate_report(result)
print(report)
```

## Validation Components

### 1. RealWorldARCValidator

Validates AI systems against the official ARC (Abstraction and Reasoning Corpus) dataset.

**Purpose**: Measure abstract reasoning and pattern recognition capabilities.

**Key Metrics**:
- Overall accuracy (0-100%)
- Problems solved
- Execution time per problem
- Error tracking

**Performance Tiers**:
- A+ (EXCEPTIONAL): ≥85% accuracy
- A (EXCELLENT): ≥70% accuracy
- B (GOOD): ≥50% accuracy
- C (FAIR): ≥30% accuracy
- D (BASIC): ≥15% accuracy
- F (NEEDS IMPROVEMENT): <15% accuracy

**Usage**:
```python
from validation import RealWorldARCValidator

validator = RealWorldARCValidator()

# Load custom problems
problems = validator.load_arc_problems('training')

# Validate system
result = validator.validate_system(
    model=your_model,
    problems=problems,
    max_problems=100  # Limit number of problems
)

# Check performance tier
tier = validator.get_performance_tier(result.overall_accuracy)
print(f"Performance Tier: {tier}")
```

### 2. CompetitivePerformanceAnalyzer

Compares system performance against state-of-the-art benchmarks.

**Purpose**: Understand competitive position in the AI landscape.

**Benchmark Systems**:
- Leading LLMs (GPT-4, Claude-3, Gemini)
- Human performance baselines
- Traditional AI approaches
- Research state-of-the-art

**Usage**:
```python
from validation import CompetitivePerformanceAnalyzer

analyzer = CompetitivePerformanceAnalyzer()

# Analyze performance
result = analyzer.analyze_performance(
    accuracy=0.35,
    system_name="MyAI"
)

print(f"Rank: {result.our_rank}/{result.total_systems}")
print(f"Percentile: {result.percentile:.1f}")
print(f"Tier: {result.competitive_tier}")

# Generate report
report = analyzer.generate_competitive_report(result)
print(report)

# Get improvement targets
targets = analyzer.get_improvement_targets(0.35)
for name, acc, gap in targets:
    print(f"Target: {name} - {acc:.2%} (gap: {gap:.2%})")
```

### 3. ErrorAnalysisSystem

Provides comprehensive error analysis and diagnostics.

**Purpose**: Identify error patterns and root causes.

**Error Categories**:
- Shape mismatch
- Value errors
- Timeouts
- Memory errors
- Type errors
- Runtime errors
- Logic errors
- Pattern recognition failures
- Generalization failures

**Usage**:
```python
from validation import ErrorAnalysisSystem

error_system = ErrorAnalysisSystem()

# Analyze errors from validation
result = error_system.analyze_errors(
    errors=validation_result.errors,
    accuracies=validation_result.problem_accuracies,
    execution_times=validation_result.execution_times
)

print(f"Error Rate: {result.error_rate:.2%}")
print(f"Total Errors: {result.total_errors}")

# View error categories
for category, count in result.error_categories.items():
    print(f"  {category}: {count}")

# Get recommendations
for rec in result.recommendations:
    print(rec)

# Generate report
report = error_system.generate_error_report(result)
print(report)
```

### 4. TemporalStabilityValidator

Tests temporal consistency and stability over time.

**Purpose**: Ensure reproducible and stable behavior.

**Metrics**:
- Stability score
- Consistency score
- Reproducibility score
- Variance
- Drift detection

**Stability Tiers**:
- PRODUCTION-READY: ≥95% stability
- STABLE: ≥85% stability
- MODERATELY-STABLE: ≥70% stability
- UNSTABLE: ≥50% stability
- HIGHLY-UNSTABLE: <50% stability

**Usage**:
```python
from validation import TemporalStabilityValidator
import numpy as np

validator = TemporalStabilityValidator()

# Prepare test input
test_input = np.random.randn(10, 10)

# Run stability test
result = validator.validate_stability(
    model=your_model,
    test_input=test_input,
    num_runs=5,
    time_interval=1.0  # seconds between runs
)

print(f"Stability: {result.stability_score:.2%}")
print(f"Tier: {result.stability_tier}")
print(f"Drift Detected: {result.drift_detected}")

# Long-term stability test
long_term = validator.validate_long_term_stability(
    model=your_model,
    test_cases=[test_input],
    duration_minutes=30
)
```

### 5. DeploymentStressTester

Production-grade stress testing for deployment readiness.

**Purpose**: Assess system behavior under load.

**Metrics**:
- Throughput (requests/second)
- Latency (average, P95, P99, max)
- Error rate
- Resource usage (CPU, memory)
- Stability score

**Readiness Classifications**:
- PRODUCTION-READY: ≥90% stability, <1% error rate
- BETA-READY: ≥75% stability, <5% error rate
- ALPHA-READY: ≥60% stability, <10% error rate
- DEVELOPMENT: ≥40% stability
- NOT-READY: <40% stability

**Usage**:
```python
from validation import DeploymentStressTester

tester = DeploymentStressTester(
    max_latency_ms=5000,
    max_memory_mb=8192
)

# Run stress test
result = tester.run_comprehensive_stress_test(
    model=your_model,
    test_input=test_data,
    duration_seconds=60,
    concurrent_requests=10,
    request_rate=5  # requests per second
)

print(f"Readiness: {result.readiness_classification}")
print(f"Throughput: {result.throughput:.1f} req/s")
print(f"Success Rate: {(1-result.error_rate):.2%}")

# Run spike test
spike_result = tester.run_spike_test(
    model=your_model,
    test_input=test_data,
    spike_duration=10
)
```

## Usage Examples

### Complete Validation Pipeline

```python
from validation import (
    RealWorldARCValidator,
    CompetitivePerformanceAnalyzer,
    ErrorAnalysisSystem,
    TemporalStabilityValidator,
    DeploymentStressTester
)

# Your model
from your_module import YourModel
model = YourModel()

# 1. ARC Validation
print("=== ARC Validation ===")
arc_validator = RealWorldARCValidator()
arc_result = arc_validator.validate_system(model)
print(validator.generate_report(arc_result))

# 2. Competitive Analysis
print("\n=== Competitive Analysis ===")
analyzer = CompetitivePerformanceAnalyzer()
comp_result = analyzer.analyze_performance(arc_result.overall_accuracy)
print(analyzer.generate_competitive_report(comp_result))

# 3. Error Analysis
print("\n=== Error Analysis ===")
error_system = ErrorAnalysisSystem()
error_result = error_system.analyze_errors(
    arc_result.errors,
    arc_result.problem_accuracies,
    arc_result.execution_times
)
print(error_system.generate_error_report(error_result))

# 4. Stability Testing
print("\n=== Stability Testing ===")
stability_validator = TemporalStabilityValidator()
stability_result = stability_validator.validate_stability(
    model, test_input, num_runs=5
)
print(stability_validator.generate_stability_report(stability_result))

# 5. Stress Testing
print("\n=== Stress Testing ===")
stress_tester = DeploymentStressTester()
stress_result = stress_tester.run_comprehensive_stress_test(
    model, test_input, duration_seconds=60
)
print(stress_tester.generate_stress_report(stress_result))
```

### Model Requirements

Your model should implement one of these interfaces:

**Option 1: ARC-specific interface**
```python
class YourModel:
    def solve_arc_problem(self, problem):
        """
        Solve an ARC problem
        
        Args:
            problem: Dict with 'train' and 'test' keys
            
        Returns:
            2D list representing the solution
        """
        # Your implementation
        return [[0, 1], [1, 0]]
```

**Option 2: Generic prediction interface**
```python
class YourModel:
    def predict(self, input_data):
        """
        Make a prediction
        
        Args:
            input_data: Any input format
            
        Returns:
            Model output
        """
        # Your implementation
        return output
```

**Option 3: Callable**
```python
class YourModel:
    def __call__(self, input_data):
        """Direct call interface"""
        return output
```

## Best Practices

### 1. Regular Validation

Run validation regularly during development:

```python
# Quick validation (5 problems)
quick_result = validator.validate_system(model, max_problems=5)

# Full validation (100+ problems)
full_result = validator.validate_system(model, max_problems=100)
```

### 2. Track Progress Over Time

```python
# Save results
import json
from datetime import datetime

result_dict = arc_result.to_dict()
result_dict['date'] = datetime.now().isoformat()

with open(f'results_{datetime.now():%Y%m%d}.json', 'w') as f:
    json.dump(result_dict, f, indent=2)
```

### 3. Use Error Analysis for Debugging

```python
# Identify problematic areas
for problem_id, error_msg in arc_result.errors.items():
    if 'timeout' in error_msg.lower():
        print(f"Timeout on {problem_id}")
        # Debug this specific problem
```

### 4. Incremental Stress Testing

Start with low load and gradually increase:

```python
# Start small
result_low = tester.run_comprehensive_stress_test(
    model, test_input, 
    duration_seconds=10,
    concurrent_requests=2,
    request_rate=1
)

# Increase load
result_med = tester.run_comprehensive_stress_test(
    model, test_input,
    duration_seconds=30,
    concurrent_requests=10,
    request_rate=5
)

# Full production load
result_high = tester.run_comprehensive_stress_test(
    model, test_input,
    duration_seconds=60,
    concurrent_requests=50,
    request_rate=10
)
```

## Troubleshooting

### Common Issues

**1. Import Errors**

```python
# Wrong (from project root)
from validation import RealWorldARCValidator

# If imports fail, add to path:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from validation import RealWorldARCValidator
```

**2. Model Interface Issues**

```
Error: Model has no attribute 'predict' or 'solve_arc_problem'
```

Solution: Implement at least one required interface (see Model Requirements above).

**3. Memory Errors During Stress Testing**

```
MemoryError: Unable to allocate array
```

Solution: Reduce concurrent requests or batch size:
```python
result = tester.run_comprehensive_stress_test(
    model, test_input,
    concurrent_requests=5,  # Reduced from 50
    request_rate=2          # Reduced from 10
)
```

**4. Timeout Errors**

```
TimeoutError: Problem validation exceeded time limit
```

Solution: Optimize model inference or increase timeout in validator code.

## API Reference

### RealWorldARCValidator

```python
RealWorldARCValidator(data_dir: Optional[Path] = None)
```

**Methods**:
- `load_arc_problems(split: str = 'training') -> Dict[str, Any]`
- `validate_system(model, problems: Optional[Dict] = None, max_problems: int = 100) -> ARCValidationResult`
- `get_performance_tier(accuracy: float) -> str`
- `generate_report(result: ARCValidationResult) -> str`

### CompetitivePerformanceAnalyzer

```python
CompetitivePerformanceAnalyzer(custom_benchmarks: Dict[str, float] = None)
```

**Methods**:
- `analyze_performance(accuracy: float, system_name: str = "NGVT") -> CompetitiveResult`
- `compare_to_category(accuracy: float, category: str) -> Dict[str, Any]`
- `get_improvement_targets(accuracy: float) -> List[Tuple[str, float, float]]`
- `generate_competitive_report(result: CompetitiveResult, system_name: str = "NGVT") -> str`

### ErrorAnalysisSystem

```python
ErrorAnalysisSystem()
```

**Methods**:
- `analyze_errors(errors: Dict[str, str], accuracies: Dict[str, float], execution_times: Dict[str, float]) -> ErrorAnalysisResult`
- `generate_error_report(result: ErrorAnalysisResult) -> str`
- `track_error_trends(current_result: ErrorAnalysisResult, historical_results: List[ErrorAnalysisResult]) -> Dict[str, Any]`

### TemporalStabilityValidator

```python
TemporalStabilityValidator(max_variance: float = 0.05, min_runs: int = 3)
```

**Methods**:
- `validate_stability(model, test_input: Any, num_runs: int = 5, time_interval: float = 1.0) -> TemporalStabilityResult`
- `generate_stability_report(result: TemporalStabilityResult) -> str`
- `validate_long_term_stability(model, test_cases: List[Any], duration_minutes: int = 30) -> Dict[str, Any]`

### DeploymentStressTester

```python
DeploymentStressTester(max_latency_ms: float = 5000, max_memory_mb: float = 8192)
```

**Methods**:
- `run_comprehensive_stress_test(model, test_input: Any = None, duration_seconds: int = 60, concurrent_requests: int = 10, request_rate: int = 5) -> StressTestResult`
- `generate_stress_report(result: StressTestResult) -> str`
- `run_spike_test(model, test_input: Any = None, spike_duration: int = 10) -> Dict[str, Any]`

---

For more information, see [TESTING_STANDARDS.md](TESTING_STANDARDS.md) for testing best practices and standards.
