"""
Temporal Stability Validator

Validates temporal consistency and stability of AI systems over time.
Tests for reproducibility, consistency, and drift detection.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TemporalStabilityResult:
    """Results from temporal stability validation"""
    stability_score: float
    consistency_score: float
    reproducibility_score: float
    drift_detected: bool
    variance: float
    test_runs: int
    timestamps: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    stability_tier: str = "UNKNOWN"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'stability_score': self.stability_score,
            'consistency_score': self.consistency_score,
            'reproducibility_score': self.reproducibility_score,
            'drift_detected': self.drift_detected,
            'variance': self.variance,
            'test_runs': self.test_runs,
            'timestamps': self.timestamps,
            'accuracy_history': self.accuracy_history,
            'stability_tier': self.stability_tier,
        }


class TemporalStabilityValidator:
    """
    Validates temporal stability and consistency
    
    Tests:
    - Reproducibility: Same inputs produce same outputs
    - Consistency: Performance remains stable over time
    - Drift detection: Identifies performance degradation
    - Variance analysis: Measures output variability
    """
    
    def __init__(self, max_variance: float = 0.05, min_runs: int = 3):
        """
        Initialize temporal stability validator
        
        Args:
            max_variance: Maximum acceptable variance in results
            min_runs: Minimum number of runs for stability assessment
        """
        self.max_variance = max_variance
        self.min_runs = min_runs
        self.test_history = defaultdict(list)
    
    def validate_stability(self,
                          model,
                          test_input: Any,
                          num_runs: int = 5,
                          time_interval: float = 1.0) -> TemporalStabilityResult:
        """
        Validate temporal stability of model
        
        Args:
            model: Model to test
            test_input: Input for testing
            num_runs: Number of test runs
            time_interval: Time between runs (seconds)
            
        Returns:
            TemporalStabilityResult with stability metrics
        """
        logger.info(f"Running temporal stability validation ({num_runs} runs)...")
        
        results = []
        timestamps = []
        
        for i in range(num_runs):
            start_time = time.time()
            
            try:
                # Run model
                if hasattr(model, 'predict'):
                    output = model.predict(test_input)
                elif hasattr(model, 'forward'):
                    output = model.forward(test_input)
                else:
                    # Fallback: use callable
                    output = model(test_input)
                
                # Convert to comparable format
                if hasattr(output, 'numpy'):
                    output = output.numpy()
                elif isinstance(output, (list, tuple)):
                    output = np.array(output)
                
                results.append(output)
                timestamps.append(start_time)
                
                logger.info(f"  Run {i+1}/{num_runs} completed")
                
                # Wait before next run
                if i < num_runs - 1:
                    time.sleep(time_interval)
                
            except Exception as e:
                logger.error(f"Error in run {i+1}: {e}")
                results.append(None)
                timestamps.append(start_time)
        
        # Filter out failed runs
        valid_results = [r for r in results if r is not None]
        
        if len(valid_results) < 2:
            logger.warning("Insufficient valid runs for stability analysis")
            return TemporalStabilityResult(
                stability_score=0.0,
                consistency_score=0.0,
                reproducibility_score=0.0,
                drift_detected=True,
                variance=1.0,
                test_runs=len(valid_results),
                timestamps=timestamps,
                accuracy_history=[],
                stability_tier="UNSTABLE"
            )
        
        # Calculate stability metrics
        stability_score = self._calculate_stability(valid_results)
        consistency_score = self._calculate_consistency(valid_results)
        reproducibility_score = self._calculate_reproducibility(valid_results)
        variance = self._calculate_variance(valid_results)
        drift_detected = self._detect_drift(valid_results)
        
        # Determine stability tier
        tier = self._get_stability_tier(stability_score)
        
        result = TemporalStabilityResult(
            stability_score=stability_score,
            consistency_score=consistency_score,
            reproducibility_score=reproducibility_score,
            drift_detected=drift_detected,
            variance=variance,
            test_runs=len(valid_results),
            timestamps=timestamps,
            accuracy_history=[],  # Can be populated by caller with actual accuracy measurements
            stability_tier=tier
        )
        
        logger.info(f"Stability validation complete: {tier} ({stability_score:.2%})")
        
        return result
    
    def _calculate_stability(self, results: List[np.ndarray]) -> float:
        """
        Calculate overall stability score
        
        Args:
            results: List of model outputs
            
        Returns:
            Stability score (0.0 to 1.0)
        """
        if len(results) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(results) - 1):
            for j in range(i + 1, len(results)):
                sim = self._calculate_similarity(results[i], results[j])
                similarities.append(sim)
        
        # Stability is average similarity
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_consistency(self, results: List[np.ndarray]) -> float:
        """
        Calculate consistency score (low variance = high consistency)
        
        Args:
            results: List of model outputs
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        try:
            # Stack results and calculate variance
            stacked = np.stack([r.flatten() for r in results])
            variance = np.var(stacked, axis=0).mean()
            
            # Convert variance to consistency (inverse relationship)
            # Lower variance = higher consistency
            consistency = 1.0 / (1.0 + variance)
            
            return min(consistency, 1.0)
        except Exception as e:
            logger.debug(f"Consistency calculation error: {e}")
            return 0.5
    
    def _calculate_reproducibility(self, results: List[np.ndarray]) -> float:
        """
        Calculate reproducibility score (exact matches)
        
        Args:
            results: List of model outputs
            
        Returns:
            Reproducibility score (0.0 to 1.0)
        """
        if len(results) < 2:
            return 0.0
        
        # Check how many results are identical to the first
        first = results[0]
        exact_matches = sum(
            1 for r in results[1:] if np.array_equal(r, first)
        )
        
        return exact_matches / (len(results) - 1)
    
    def _calculate_variance(self, results: List[np.ndarray]) -> float:
        """
        Calculate variance across results
        
        Args:
            results: List of model outputs
            
        Returns:
            Variance value
        """
        try:
            stacked = np.stack([r.flatten() for r in results])
            return float(np.var(stacked))
        except Exception:
            return 1.0
    
    def _calculate_similarity(self, result1: np.ndarray, result2: np.ndarray) -> float:
        """
        Calculate similarity between two results
        
        Args:
            result1: First result
            result2: Second result
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Ensure same shape
            if result1.shape != result2.shape:
                return 0.0
            
            # Calculate cosine similarity
            r1_flat = result1.flatten()
            r2_flat = result2.flatten()
            
            dot_product = np.dot(r1_flat, r2_flat)
            norm1 = np.linalg.norm(r1_flat)
            norm2 = np.linalg.norm(r2_flat)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert from [-1, 1] to [0, 1]
            return (similarity + 1) / 2
            
        except Exception as e:
            logger.debug(f"Similarity calculation error: {e}")
            return 0.5
    
    def _detect_drift(self, results: List[np.ndarray]) -> bool:
        """
        Detect if there's a drift in results over time
        
        Args:
            results: List of model outputs in chronological order
            
        Returns:
            True if drift detected, False otherwise
        """
        if len(results) < 3:
            return False
        
        try:
            # Calculate moving average of similarities
            similarities = []
            for i in range(len(results) - 1):
                sim = self._calculate_similarity(results[i], results[i + 1])
                similarities.append(sim)
            
            # Check if similarities are decreasing (indicating drift)
            if len(similarities) >= 3:
                # Simple linear trend check
                x = np.arange(len(similarities))
                coeffs = np.polyfit(x, similarities, 1)
                slope = coeffs[0]
                
                # Negative slope indicates drift
                return slope < -0.05
            
            return False
            
        except Exception:
            return False
    
    def _get_stability_tier(self, stability_score: float) -> str:
        """
        Get stability tier based on score
        
        Args:
            stability_score: Stability score
            
        Returns:
            Tier description
        """
        if stability_score >= 0.95:
            return "PRODUCTION-READY"
        elif stability_score >= 0.85:
            return "STABLE"
        elif stability_score >= 0.70:
            return "MODERATELY-STABLE"
        elif stability_score >= 0.50:
            return "UNSTABLE"
        else:
            return "HIGHLY-UNSTABLE"
    
    def generate_stability_report(self, result: TemporalStabilityResult) -> str:
        """
        Generate temporal stability report
        
        Args:
            result: Stability validation results
            
        Returns:
            Formatted report string
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║         TEMPORAL STABILITY REPORT                            ║
╚══════════════════════════════════════════════════════════════╝

Stability Tier: {result.stability_tier}

Metrics:
  • Overall Stability: {result.stability_score:.2%}
  • Consistency Score: {result.consistency_score:.2%}
  • Reproducibility: {result.reproducibility_score:.2%}
  • Variance: {result.variance:.6f}

Test Configuration:
  • Total Runs: {result.test_runs}
  • Drift Detected: {'Yes ⚠️' if result.drift_detected else 'No ✓'}

Assessment:
"""
        
        if result.stability_score >= 0.95:
            report += "  ✅ Excellent stability - Production ready\n"
        elif result.stability_score >= 0.85:
            report += "  ✓  Good stability - Suitable for most applications\n"
        elif result.stability_score >= 0.70:
            report += "  ⚠️  Moderate stability - May need improvement\n"
        else:
            report += "  ❌ Poor stability - Requires attention\n"
        
        if result.drift_detected:
            report += "  ⚠️  Performance drift detected over time\n"
        
        if result.variance > self.max_variance:
            report += f"  ⚠️  High variance ({result.variance:.4f}) exceeds threshold ({self.max_variance})\n"
        
        return report
    
    def validate_long_term_stability(self,
                                    model,
                                    test_cases: List[Any],
                                    duration_minutes: int = 30) -> Dict[str, Any]:
        """
        Validate stability over extended period
        
        Args:
            model: Model to test
            test_cases: List of test inputs
            duration_minutes: Duration of long-term test
            
        Returns:
            Long-term stability metrics
        """
        logger.info(f"Starting long-term stability test ({duration_minutes} minutes)...")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        results_over_time = []
        timestamps = []
        
        iteration = 0
        while time.time() < end_time:
            try:
                # Test on random case
                test_case = test_cases[iteration % len(test_cases)]
                
                # Run single validation
                result = self.validate_stability(model, test_case, num_runs=3, time_interval=0.5)
                
                results_over_time.append(result.stability_score)
                timestamps.append(time.time())
                
                iteration += 1
                
                # Sleep between iterations
                time.sleep(60)  # Test every minute
                
            except KeyboardInterrupt:
                logger.info("Long-term test interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
        
        # Analyze results
        if results_over_time:
            overall_stability = np.mean(results_over_time)
            stability_variance = np.var(results_over_time)
            trend_slope = np.polyfit(range(len(results_over_time)), results_over_time, 1)[0]
        else:
            overall_stability = 0.0
            stability_variance = 0.0
            trend_slope = 0.0
        
        return {
            'duration_minutes': (time.time() - start_time) / 60,
            'iterations': iteration,
            'overall_stability': overall_stability,
            'stability_variance': stability_variance,
            'trend_slope': trend_slope,
            'degradation_detected': trend_slope < -0.01,
            'results_over_time': results_over_time,
            'timestamps': timestamps,
        }
