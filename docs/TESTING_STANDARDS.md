# NGVT Testing Standards

## Overview

This document defines the testing standards, best practices, and quality benchmarks for the NGVT validation infrastructure. Following these standards ensures consistent, reliable, and production-ready testing.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Performance Benchmarks](#performance-benchmarks)
3. [Compliance Requirements](#compliance-requirements)
4. [Testing Workflow](#testing-workflow)
5. [CI/CD Integration](#cicd-integration)
6. [Quality Gates](#quality-gates)

## Testing Philosophy

### Core Principles

1. **Comprehensive Coverage**: Test all critical functionality
2. **Reproducibility**: Tests must produce consistent results
3. **Real-World Relevance**: Use realistic test scenarios
4. **Performance Awareness**: Monitor execution time and resources
5. **Clear Reporting**: Provide actionable feedback

### Testing Pyramid

```
        /\
       /  \      E2E Tests (5%)
      /----\     Integration Tests (15%)
     /------\    Unit Tests (80%)
    /--------\
```

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete workflows

## Performance Benchmarks

### ARC Validation Benchmarks

| Performance Tier | Accuracy | Classification |
|-----------------|----------|----------------|
| A+ | ≥85% | EXCEPTIONAL |
| A | 70-84% | EXCELLENT |
| B | 50-69% | GOOD |
| C | 30-49% | FAIR |
| D | 15-29% | BASIC |
| F | <15% | NEEDS IMPROVEMENT |

### Competitive Standing

| Tier | Percentile | Description |
|------|-----------|-------------|
| WORLD-CLASS | ≥95th | Top-tier performance |
| EXCELLENT | 85-94th | Industry-leading |
| STRONG | 70-84th | Above average |
| COMPETITIVE | 50-69th | Market competitive |
| DEVELOPING | 30-49th | Room for improvement |
| BASELINE | <30th | Below baseline |

### Stability Requirements

| Classification | Stability Score | Requirements |
|----------------|-----------------|--------------|
| PRODUCTION-READY | ≥95% | No drift, <5% variance |
| STABLE | 85-94% | Minimal drift, <10% variance |
| MODERATELY-STABLE | 70-84% | Some variance acceptable |
| UNSTABLE | 50-69% | Significant improvement needed |
| HIGHLY-UNSTABLE | <50% | Not production-ready |

### Deployment Readiness

| Classification | Stability | Error Rate | Latency |
|----------------|-----------|------------|---------|
| PRODUCTION-READY | ≥90% | <1% | <1s avg |
| BETA-READY | ≥75% | <5% | <2s avg |
| ALPHA-READY | ≥60% | <10% | <5s avg |
| DEVELOPMENT | ≥40% | <20% | Any |
| NOT-READY | <40% | Any | Any |

## Compliance Requirements

### ARC Standards Compliance

#### 1. Problem Structure

All ARC problems must conform to:

```python
{
    'train': [
        {'input': [[int]], 'output': [[int]]}  # 2D grid
    ],
    'test': [
        {'input': [[int]], 'output': [[int]]}  # 2D grid
    ]
}
```

Requirements:
- Values must be integers 0-9
- Grids must be 2D (list of lists)
- Both input and output required

#### 2. Validation Results

Validation results must include:
- Overall accuracy (0.0 to 1.0)
- Problems solved (count)
- Total problems (count)
- Per-problem accuracies (dict)
- Execution times (dict)
- Errors (dict)
- Timestamp

#### 3. Error Handling

Must gracefully handle:
- Model failures
- Timeout errors
- Memory errors
- Invalid inputs
- Missing data

### Testing Coverage Requirements

Minimum coverage standards:

| Component | Unit Test Coverage | Integration Coverage |
|-----------|-------------------|---------------------|
| Validators | ≥90% | ≥80% |
| Analyzers | ≥85% | ≥75% |
| Utilities | ≥80% | ≥70% |
| Overall | ≥85% | ≥75% |

## Testing Workflow

### 1. Development Testing

During active development:

```bash
# Quick unit tests (30 seconds)
python tests/test_performance.py

# Quick compliance tests (20 seconds)
python tests/test_compliance.py

# Run specific test class
python -m unittest tests.test_performance.TestRealWorldARCValidator
```

### 2. Pre-Commit Testing

Before committing changes:

```bash
# Run all tests
python tests/test_performance.py
python tests/test_compliance.py

# Check for errors
echo $?  # Should be 0
```

### 3. Pre-Release Testing

Before releasing to production:

```bash
# Full test suite
python tests/test_performance.py
python tests/test_compliance.py

# Integration testing
python examples/validation_example.py

# Stress testing
# Run deployment stress tests with production parameters
```

### 4. Continuous Testing

Regular automated testing:

```bash
# Daily validation runs
python -c "
from validation import RealWorldARCValidator
validator = RealWorldARCValidator()
result = validator.validate_system(model)
assert result.overall_accuracy >= 0.15  # Minimum threshold
"

# Weekly comprehensive testing
# Run full test suite + stress tests
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: NGVT Validation Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        cd torusScode
        pip install -r requirements_web.txt
    
    - name: Run performance tests
      run: python tests/test_performance.py
    
    - name: Run compliance tests
      run: python tests/test_compliance.py
    
    - name: Check minimum accuracy
      run: |
        python -c "
        from validation import RealWorldARCValidator
        # Quick validation check
        "
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

echo "Running NGVT validation tests..."

# Run quick tests
python tests/test_performance.py
PERF_RESULT=$?

python tests/test_compliance.py
COMP_RESULT=$?

if [ $PERF_RESULT -ne 0 ] || [ $COMP_RESULT -ne 0 ]; then
    echo "Tests failed! Commit aborted."
    exit 1
fi

echo "All tests passed!"
exit 0
```

## Quality Gates

### Gate 1: Unit Tests (Required)

**Criteria**:
- All unit tests pass
- Coverage ≥85%
- No critical errors

**Command**: `python tests/test_performance.py`

### Gate 2: Compliance Tests (Required)

**Criteria**:
- 100% ARC standards compliance
- All validation requirements met
- Proper error handling

**Command**: `python tests/test_compliance.py`

### Gate 3: Performance Validation (Required for Release)

**Criteria**:
- Accuracy ≥15% (minimum threshold)
- No critical performance regressions
- Execution time within acceptable limits

**Validation**:
```python
from validation import RealWorldARCValidator

validator = RealWorldARCValidator()
result = validator.validate_system(model)

# Must meet minimum threshold
assert result.overall_accuracy >= 0.15
```

### Gate 4: Stability Testing (Required for Production)

**Criteria**:
- Stability score ≥85%
- No drift detected
- Variance <10%

**Validation**:
```python
from validation import TemporalStabilityValidator

validator = TemporalStabilityValidator()
result = validator.validate_stability(model, test_input, num_runs=10)

# Must be stable
assert result.stability_score >= 0.85
assert not result.drift_detected
```

### Gate 5: Stress Testing (Required for Production)

**Criteria**:
- Readiness: BETA-READY or higher
- Error rate <5%
- Throughput meets requirements

**Validation**:
```python
from validation import DeploymentStressTester

tester = DeploymentStressTester()
result = tester.run_comprehensive_stress_test(
    model, test_input,
    duration_seconds=300,
    concurrent_requests=50
)

# Must be deployment ready
assert result.readiness_classification in ['PRODUCTION-READY', 'BETA-READY']
assert result.error_rate < 0.05
```

## Best Practices

### 1. Test Organization

```
tests/
├── __init__.py
├── test_performance.py      # Unit and performance tests
├── test_compliance.py        # Standards compliance tests
├── test_integration.py       # Integration tests (future)
└── fixtures/                 # Test data and fixtures
    ├── sample_problems.json
    └── test_models.py
```

### 2. Test Naming

```python
# Good
def test_validator_handles_invalid_input(self):
    """Test that validator gracefully handles invalid input"""
    
def test_accuracy_within_expected_range(self):
    """Test accuracy is between 0 and 1"""

# Bad  
def test1(self):
def test_stuff(self):
```

### 3. Assertion Messages

```python
# Good
self.assertGreaterEqual(
    result.accuracy, 0.0,
    f"Accuracy must be >= 0, got {result.accuracy}"
)

# Bad
self.assertGreaterEqual(result.accuracy, 0.0)
```

### 4. Test Fixtures

```python
class TestValidator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.validator = RealWorldARCValidator()
        self.test_model = MockModel(accuracy=0.7)
        self.test_problems = self.validator._create_sample_problems()
    
    def tearDown(self):
        """Clean up after tests"""
        # Clean up resources if needed
        pass
```

### 5. Mocking

```python
class MockModel:
    """Mock model for testing without real inference"""
    
    def __init__(self, accuracy=0.5, fail_rate=0.0):
        self.accuracy = accuracy
        self.fail_rate = fail_rate
    
    def predict(self, input_data):
        if random.random() < self.fail_rate:
            raise RuntimeError("Simulated failure")
        return self._generate_mock_output(input_data)
```

### 6. Performance Testing

```python
import time

def test_validation_performance(self):
    """Test validation completes within time limit"""
    start = time.time()
    result = self.validator.validate_system(
        self.model, 
        max_problems=10
    )
    elapsed = time.time() - start
    
    # Should complete in reasonable time
    self.assertLess(elapsed, 60.0,
                   f"Validation took {elapsed:.1f}s, expected <60s")
```

### 7. Error Testing

```python
def test_handles_model_errors(self):
    """Test graceful handling of model errors"""
    
    class BrokenModel:
        def predict(self, x):
            raise RuntimeError("Model error")
    
    broken = BrokenModel()
    result = self.validator.validate_system(broken, max_problems=2)
    
    # Should track errors, not crash
    self.assertGreater(len(result.errors), 0)
    self.assertEqual(result.solved_problems, 0)
```

## Reporting Standards

### Test Output Format

```
======================================================================
NGVT VALIDATION INFRASTRUCTURE - TEST SUITE
======================================================================

test_validator_initialization (tests.test_performance.TestValidator) ... ok
test_validate_system (tests.test_performance.TestValidator) ... ok
test_performance_tier (tests.test_performance.TestValidator) ... ok

----------------------------------------------------------------------
Ran 45 tests in 12.345s

OK

======================================================================
TEST SUMMARY
======================================================================
Tests run: 45
Successes: 45
Failures: 0
Errors: 0

✅ ALL TESTS PASSED!
```

### Validation Report Format

```
╔══════════════════════════════════════════════════════════════╗
║          ARC VALIDATION REPORT                               ║
╚══════════════════════════════════════════════════════════════╝

Performance Tier: B (GOOD)

Overall Results:
  • Accuracy: 55.23%
  • Problems Solved: 112/203
  • Success Rate: 112/203

Execution Metrics:
  • Total Problems: 203
  • Errors Encountered: 15
  • Average Time/Problem: 0.342s
```

## Continuous Improvement

### Performance Tracking

Track metrics over time:

```python
import json
from datetime import datetime

# Save results
results_history = []
result_data = {
    'timestamp': datetime.now().isoformat(),
    'accuracy': result.overall_accuracy,
    'tier': validator.get_performance_tier(result.overall_accuracy),
    'solved': result.solved_problems,
    'total': result.total_problems
}
results_history.append(result_data)

# Save to file
with open('validation_history.json', 'w') as f:
    json.dump(results_history, f, indent=2)
```

### Regression Detection

```python
def check_regression(current_accuracy, historical_accuracy):
    """Check for performance regression"""
    threshold = 0.05  # 5% regression threshold
    
    if current_accuracy < historical_accuracy - threshold:
        print(f"⚠️  REGRESSION DETECTED!")
        print(f"Current: {current_accuracy:.2%}")
        print(f"Previous: {historical_accuracy:.2%}")
        print(f"Drop: {(historical_accuracy - current_accuracy):.2%}")
        return True
    return False
```

---

## Summary Checklist

Before deploying to production, verify:

- [ ] All unit tests pass (test_performance.py)
- [ ] All compliance tests pass (test_compliance.py)
- [ ] Accuracy ≥15% (minimum threshold)
- [ ] Stability score ≥85%
- [ ] No performance drift detected
- [ ] Error rate <5%
- [ ] Stress test: BETA-READY or higher
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] CI/CD pipeline passing

Following these standards ensures NGVT maintains A+ quality and production readiness.
